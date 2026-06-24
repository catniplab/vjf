"""Sweep rollout configs for one that gives the free-run a genuine attracting limit cycle.

Fit the best one-step config ONCE, then for each rollout config deep-copy the flow, fine-tune,
and score the autonomous free-run by the TRUE limit-cycle test (not the forecast metric):

  - per-cycle amplitude ratio of the on-cycle free-run -> ~1.0 (sustained, not decaying to 0);
  - initial conditions inside (0.4x) and outside (1.8x) the cycle converge to a COMMON nonzero
    tail amplitude (attracting ring) rather than both collapsing to the center (focus).

The hypothesis under test: a long, exactly-periodic target (``target='cycle'``, the folded+tiled
trial-average orbit) plus a multi-cycle rollout horizon turns the slow per-cycle decay into a
large enough penalty to pin the amplitude mode to marginal stability.

Writes results JSON + an amplitude-envelope gallery. Run: uv run python -m experiments.v1_graf.rollout_sweep
"""
from __future__ import annotations
import copy
import json
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
from experiments.v1_graf.rollout_diag import free_run, cycle_amplitude, per_cycle_ratio
from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
OUT = os.path.join(HERE, "rollout_sweep_results.json")
N_CYC = 80                                      # free-run length (cycles) for scoring
TAIL = 20                                       # cycles averaged for the tail amplitude

# Escalating periodic-target rollouts (the long, clean target is what can penalize slow decay).
CONFIGS = [
    dict(name="before", rollout=False),
    dict(name="trialavg k48", target_mode="trialavg", k_ladder=[8, 16, 32, 48], clip=1.0, epochs=300),
    dict(name="cycle k48 c1", target_mode="cycle", k_ladder=[16, 32, 48], clip=1.0, epochs=300),
    dict(name="cycle k96 c1", target_mode="cycle", k_ladder=[24, 48, 96], clip=1.0, epochs=250),
    dict(name="cycle k96 c10", target_mode="cycle", k_ladder=[24, 48, 96], clip=10.0, epochs=250),
    dict(name="cycle k160 c1", target_mode="cycle", k_ladder=[32, 64, 128, 160], clip=1.0, epochs=250),
    dict(name="cycle k160 c10", target_mode="cycle", k_ladder=[32, 64, 128, 160], clip=10.0, epochs=250),
    dict(name="cycle k160 pure", target_mode="cycle", k_ladder=[160], clip=10.0, epochs=200),
]


def main():
    torch.set_default_dtype(torch.float32)
    data = prepare_single_dir_data(direction=225.0, n_val=10)
    test, psth = data["test_trials"], data["psth_counts"]
    ybar = float(data["test_counts"].mean())

    res = _train_eval(data["train_trials"], test, data["test_counts"], ybar, epochs=100,
                      latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
                      flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
                      dyn_noise=0.0, smooth_lambda=0.0, psth_counts=psth, return_model=True)
    model0, ro = res["model"], res["ro"]
    print(f"fitted one-step: test PLL {res['pll']:.3f} (ceil {data['pll_psth']:.3f})", flush=True)

    allt = list(data["train_trials"]) + list(data["val_trials"]) + list(test)
    xbar = np.stack([filtered_path(model0, ro, t)[0] for t in allt], 0).mean(0)
    C = model0.decoder.decode.weight.detach().cpu().numpy()
    W, fve, ctr = decoded_variance_basis(xbar, C)
    data_amp = float(cycle_amplitude(project(xbar, W, ctr)).mean())
    x_t0 = filtered_path(model0, ro, test[0])[0][T0]
    ics = {"on": x_t0, "inside": ctr + 0.4 * (x_t0 - ctr), "outside": ctr + 1.8 * (x_t0 - ctr)}
    n_long = N_CYC * BINS_PER_CYCLE

    def amps_for(model):
        out = {}
        for tag, ic in ics.items():
            out[tag] = cycle_amplitude(project(free_run(model, ic, n_long), W, ctr))
        return out

    results = []
    for cfg in CONFIGS:
        m = copy.deepcopy(model0)
        if cfg.get("rollout", True):
            print(f"[{cfg['name']}] rollout ...", flush=True)
            rollout_finetune(m, ro, data["train_trials"], epochs=cfg["epochs"],
                             k_ladder=cfg["k_ladder"], target_mode=cfg["target_mode"],
                             clip=cfg["clip"], lr=1e-3, verbose=True)
        amps = amps_for(m)
        s, kd = test_accuracy(m, ro, test, psth)
        ratio = per_cycle_ratio(amps["on"])
        in_tail, out_tail, on_tail = (float(amps[t][-TAIL:].mean()) for t in ("inside", "outside", "on"))
        rel_gap = abs(in_tail - out_tail) / (0.5 * (in_tail + out_tail) + 1e-9)
        converged = bool(rel_gap < 0.25 and min(in_tail, out_tail) > 0.3 * data_amp)
        rec = dict(name=cfg["name"], ratio=ratio, on_tail=on_tail, in_tail=in_tail, out_tail=out_tail,
                   rel_gap=rel_gap, converged=converged, tail_over_data=on_tail / data_amp,
                   S=s, vp8=kd[8]["vs_persist"], vp16=kd[16]["vs_persist"], vp32=kd[32]["vs_persist"],
                   amps={t: amps[t].tolist() for t in amps})
        results.append(rec)
        print(f"  {cfg['name']:16s} ratio/cyc {ratio:.4f} | tail on/in/out "
              f"{on_tail:.2f}/{in_tail:.2f}/{out_tail:.2f} (data {data_amp:.2f}) | "
              f"{'LIMIT CYCLE' if converged else 'focus'} gap {rel_gap:.2f} | S {s:+.4f}", flush=True)

    with open(OUT, "w") as fh:
        json.dump(dict(data_amp=data_amp, results=[{k: v for k, v in r.items() if k != "amps"}
                                                    for r in results]), fh, indent=2)

    # gallery: amplitude envelope per config (on/inside/outside ICs), log-y
    set_style()
    ncol = 4
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FW(1.0), 1.7 * nrow + 0.3), sharex=True, squeeze=False)
    for ci, r in enumerate(results):
        ax = axes[ci // ncol][ci % ncol]
        ax.axhline(data_amp, color="#c8920a", lw=1.4, alpha=0.8)
        for tag, col in [("on", "#d6336c"), ("inside", "#1b9e9a"), ("outside", "#1b6fd6")]:
            ax.semilogy(np.arange(len(r["amps"][tag])), r["amps"][tag], color=col, lw=1.0, label=tag)
        verdict = "LIMIT CYCLE" if r["converged"] else "focus"
        ax.set_title(f"{r['name']}\nratio {r['ratio']:.3f} | {verdict} | S{r['S']:+.3f}", fontsize=6.5)
        if ci == 0:
            ax.legend(fontsize=5.5, loc="lower left")
    for j in range(len(results), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Rollout sweep: free-run amplitude vs cycle (gold = data orbit; converge to a "
                 "common nonzero band = limit cycle)", fontsize=8)
    fig.savefig(os.path.join(FIGS, "rollout_sweep.png"), dpi=200)
    plt.close(fig)
    print(f"\nsweep -> {FIGS}/rollout_sweep.png ; {OUT}", flush=True)
    winners = [r["name"] for r in results if r["converged"]]
    print(f"limit-cycle configs: {winners or 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
