"""Does rollout training give the free-run an attracting limit cycle? (before / k=8 / curriculum)

The forecast figures only free-run to the trial length (~7 cycles), over which a slowly decaying
stable focus is indistinguishable from a limit cycle. This diagnostic settles it: fit the best
config ONCE, then deep-copy the flow into three and free-run each autonomous flow for ~100 cycles:

  - before      : no rollout (one-step trained);
  - k=8         : the current fixed half-cycle rollout;
  - curriculum  : curriculum + mixture rollout over k in [8, 16, 32, 48] (up to 3 cycles).

A genuine limit cycle shows (a) per-cycle amplitude ratio -> ~1.0 (plateau, not decay to 0) and
(b) initial conditions inside / on / outside the cycle all converging onto ONE common ring. A
stable focus instead spirals every IC inward to a point (ratio < 1).

Run: uv run python -m experiments.v1_graf.rollout_diag
"""
from __future__ import annotations
import copy
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval
from experiments.v1_graf.forecast_video import filtered_path, T0, BINS_PER_CYCLE
from experiments.v1_graf.factor_order import decoded_variance_basis, project
from experiments.v1_graf.rollout import rollout_finetune
from experiments.v1_graf.rollout_test import test_accuracy
from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
N_CYC = 100                                     # free-run length (cycles) for the envelope / ratio
N_PORT = 40                                     # cycles shown in phase portraits (cleaner)
LADDER = [8, 16, 32, 48]                        # curriculum horizons (1/2, 1, 2, 3 cycles)
GOLD, CRIM, GRAY, BLUE = "#c8920a", "#d6336c", "#7a7a7a", "#1b6fd6"


@torch.no_grad()
def free_run(model, x0, n_step):
    """Autonomous free-run (deterministic mean flow) from x0 (L,) for n_step bins -> (n_step+1, L)."""
    x, _ = model.forecast(torch.as_tensor(x0[None].astype(np.float32)), n_step=n_step)
    return x.detach().cpu().numpy()[:, 0, :]


def cycle_amplitude(z):
    """Per-cycle peak radius in the (factor1, factor2) plane (max over each 16-bin window)."""
    r = np.hypot(z[:, 0], z[:, 1])
    nc = len(r) // BINS_PER_CYCLE
    return np.array([r[c * BINS_PER_CYCLE:(c + 1) * BINS_PER_CYCLE].max() for c in range(nc)])


def per_cycle_ratio(amp):
    return (amp[-1] / amp[0]) ** (1.0 / max(1, len(amp) - 1))


def main():
    torch.set_default_dtype(torch.float32)
    data = prepare_single_dir_data(direction=225.0, n_val=10)
    test, psth = data["test_trials"], data["psth_counts"]
    ybar = float(data["test_counts"].mean())

    # Fit the best config ONCE (L=4, 1600 RBF, lambda=0); rollout only touches the flow weights,
    # so deep-copies share the readout/decoder -> all free-runs live in ONE coordinate frame.
    res = _train_eval(data["train_trials"], test, data["test_counts"], ybar, epochs=100,
                      latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
                      flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
                      dyn_noise=0.0, smooth_lambda=0.0, psth_counts=psth, return_model=True)
    model0, ro = res["model"], res["ro"]
    print(f"fitted one-step: test PLL {res['pll']:.3f} (ceil {data['pll_psth']:.3f}), "
          f"n_basis {res['n_basis']}", flush=True)

    models = {"before": copy.deepcopy(model0)}
    print(f"rollout curriculum+mixture k in {LADDER}, single-trial targets (40 epochs) ...", flush=True)
    models["filtered"] = rollout_finetune(copy.deepcopy(model0), ro, data["train_trials"],
                                          epochs=40, k_ladder=LADDER, target_mode="filtered", verbose=True)
    print(f"rollout curriculum+mixture k in {LADDER}, trial-average orbit (400 epochs) ...", flush=True)
    models["trialavg"] = rollout_finetune(copy.deepcopy(model0), ro, data["train_trials"],
                                          epochs=400, k_ladder=LADDER, target_mode="trialavg", verbose=True)

    # ONE decoded-variance basis from the before-model trial-average (shared reference frame).
    allt = list(data["train_trials"]) + list(data["val_trials"]) + list(test)
    xbar = np.stack([filtered_path(models["before"], ro, t)[0] for t in allt], 0).mean(0)
    C = model0.decoder.decode.weight.detach().cpu().numpy()
    W, fve, ctr = decoded_variance_basis(xbar, C)
    pbar = project(xbar, W, ctr)
    bar_amp = cycle_amplitude(pbar)
    x_t0 = filtered_path(models["before"], ro, test[0])[0][T0]    # common free-run start

    # long free-run + amplitude envelope + forecast accuracy for each model
    runs, amps, stats = {}, {}, {}
    for tag, m in models.items():
        z = project(free_run(m, x_t0, N_CYC * BINS_PER_CYCLE), W, ctr)
        amp = cycle_amplitude(z)
        s, kd = test_accuracy(m, ro, test, psth)
        runs[tag], amps[tag] = z, amp
        stats[tag] = dict(ratio=per_cycle_ratio(amp), end_data=amp[-1] / bar_amp.mean(),
                          S=s, vp=(kd[8]["vs_persist"], kd[16]["vs_persist"], kd[32]["vs_persist"]))
        print(f"{tag:11s}: amp {amp[0]:.2f}->{amp[-1]:.2f} | ratio/cyc {stats[tag]['ratio']:.4f} "
              f"| end/data {stats[tag]['end_data']:.2f} | S {s:+.4f} "
              f"| vs-persist k8/16/32 {stats[tag]['vp'][0]:+.3f}/{stats[tag]['vp'][1]:+.3f}/{stats[tag]['vp'][2]:+.3f}",
              flush=True)

    # multi-IC for the trial-average model (limit-cycle attraction test)
    ics = {"on cycle (t0)": x_t0, "inside (0.4x)": ctr + 0.4 * (x_t0 - ctr),
           "outside (1.8x)": ctr + 1.8 * (x_t0 - ctr)}
    multi = {t: project(free_run(models["trialavg"], ic, N_PORT * BINS_PER_CYCLE), W, ctr)
             for t, ic in ics.items()}

    # ---------------- figure: 2 x 3 ----------------
    set_style()
    fig, ax = plt.subplots(2, 3, figsize=(FW(1.0), FW(1.0) * 0.62))
    np_ = N_PORT * BINS_PER_CYCLE

    def portrait(a, z, color, title):
        z = z[:np_]
        for i in range(0, len(z) - 1, 4):                       # fade older segments
            a.plot(z[i:i + 5, 0], z[i:i + 5, 1], color=color, lw=0.6, alpha=0.12 + 0.88 * i / len(z))
        a.plot(pbar[:, 0], pbar[:, 1], color=GOLD, lw=1.6, alpha=0.9, zorder=1)
        a.scatter(*z[0, :2], color="k", s=16, zorder=5)
        a.set_title(title, fontsize=8); a.set_aspect("equal", "box")
        a.set_xlabel("factor 1"); a.set_ylabel("factor 2")

    portrait(ax[0, 0], runs["before"], GRAY, f"before  (ratio {stats['before']['ratio']:.3f}/cyc)")
    portrait(ax[0, 1], runs["filtered"], BLUE, f"single-trial tgt  (ratio {stats['filtered']['ratio']:.3f}/cyc)")
    portrait(ax[0, 2], runs["trialavg"], CRIM, f"trial-avg tgt  (ratio {stats['trialavg']['ratio']:.3f}/cyc)")

    a = ax[1, 0]                                                # amplitude envelope (log-y)
    a.axhspan(bar_amp.min(), bar_amp.max(), color=GOLD, alpha=0.18, label="data orbit band")
    for tag, col in [("before", GRAY), ("filtered", BLUE), ("trialavg", CRIM)]:
        a.semilogy(np.arange(len(amps[tag])), amps[tag], color=col, lw=1.6, label=tag)
    a.set_xlabel("cycle"); a.set_ylabel("free-run amplitude")
    a.set_title("Per-cycle amplitude envelope", fontsize=8); a.legend(fontsize=6)

    a = ax[1, 1]                                                # curriculum multi-IC
    a.plot(pbar[:, 0], pbar[:, 1], color=GOLD, lw=1.6, alpha=0.9, label="data orbit")
    for (tag, z), col in zip(multi.items(), [CRIM, "#1b9e9a", BLUE]):
        a.plot(z[:, 0], z[:, 1], color=col, lw=0.6, alpha=0.7, label=tag)
        a.scatter(*z[0, :2], color=col, s=14, zorder=5)
    a.set_title("Curriculum: 3 initial conditions", fontsize=8); a.legend(fontsize=5.5)
    a.set_xlabel("factor 1"); a.set_ylabel("factor 2"); a.set_aspect("equal", "box")

    a = ax[1, 2]                                                # factor-1 long time series
    t = np.arange(np_)
    a.plot(t / BINS_PER_CYCLE, runs["before"][:np_, 0], color=GRAY, lw=0.8, alpha=0.8, label="before")
    a.plot(t / BINS_PER_CYCLE, runs["trialavg"][:np_, 0], color=CRIM, lw=0.8, alpha=0.9, label="trial-avg tgt")
    a.set_xlabel("cycle"); a.set_ylabel("factor 1 (free-run)")
    a.set_title("Factor 1 over time", fontsize=8); a.legend(fontsize=6)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"rollout_diag.{ext}"), dpi=200)
    plt.close(fig)
    print(f"diag -> {FIGS}/rollout_diag.png", flush=True)


if __name__ == "__main__":
    main()
