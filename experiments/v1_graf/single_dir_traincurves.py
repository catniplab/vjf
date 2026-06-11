"""Training-curve diagnostics for the single-direction sVJF -- focus: how much does
the readout subspace move as it learns (the drift behind the stale RBF centers)?

Logs, over the online training stream: the readout C subspace principal angle (per
refresh, and cumulative from the seeded C), C condition number, ELBO components, the
filtered-latent centroid/spread, and the distance from the filtered latent to its
nearest (fixed) RBF center.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask, kmeans_centers
from experiments.v1_graf.figstyle import set_style, ema
from vjf.model import VJF
from vjf.readout import OnlineReadout, _principal_angle
from vjf.realtime import online_filter_trials

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
BIN_MS, T_MAX_MS, SEED = 10.0, 1400.0, 20260609
EPOCHS, WARMUP_TRIALS, N_RBF, HIDDEN, REFRESH_K = 50, 8, 100, [64, 64], 500


def main(grow=False, grow_thresh=0.5, max_rbf=None, grow_min_gap=None, latent_dim=2,
         column_norm="eig", tag="", rbf_base=25, width_scale=0.5):
    os.makedirs(FIGS, exist_ok=True)
    set_style()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    arr = load_array(5)
    cycle_bins = 1000.0 / (arr["tf"] * BIN_MS)               # bins per grating cycle (EMA timescale)
    counts = bin_spikes(arr["spk_times"], bin_ms=BIN_MS)
    dirs = arr["ori"]
    mask, _ = well_tuned_mask(counts, dirs)
    counts = counts[:, :, mask]
    N = counts.shape[2]
    nt = int(round(T_MAX_MS / BIN_MS))
    keep = np.unique(dirs)
    d_star = float(max(keep, key=lambda d: counts[dirs == d][:, :nt, :].mean()))
    idx = np.where(dirs == d_star)[0]; rng.shuffle(idx)
    train_trials = [counts[i][:nt] for i in idx[10:]]
    steps_per_epoch = len(train_trials) * nt

    cover = np.concatenate(train_trials[:WARMUP_TRIALS], 0)
    # RBF budget scales O(2^d) with latent dim (curse of dimensionality); seed a small
    # basis and grow via novelty up to the cap over the first epoch.
    max_rbf_eff = max_rbf if max_rbf is not None else rbf_base * (2 ** latent_dim)
    n_seed = int(min(max(8, 2 ** latent_dim), cover.shape[0])) if grow else \
        int(min(max_rbf_eff, cover.shape[0]))
    wscale = width_scale if grow else 1.0                  # narrow seed only when growing
    model = VJF.make_model(ydim=N, xdim=latent_dim, udim=0, n_rbf=n_seed, hidden_sizes=HIDDEN,
                           likelihood="poisson", transition_flow="srrls", encoder="projection")
    ro = OnlineReadout(N, latent_dim, smooth_tau=8.0, refresh_K=REFRESH_K, link="log",
                       column_norm=column_norm)
    Cw, bw = ro.warm_start(cover)
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cw))
        model.decoder.decode.bias.copy_(torch.as_tensor(bw.reshape(-1)))

    # rate-limit growth so the (cap - seed) adds spread across the whole first epoch
    # (default), instead of saturating the cap in the first ~1.5 trials.
    gap = (grow_min_gap if grow_min_gap is not None
           else max(1, steps_per_epoch // max(1, max_rbf_eff - n_seed)))
    if grow:
        model.transition.grow_rbf = True
        model.transition.grow_thresh = grow_thresh
        model.transition.max_rbf = max_rbf_eff
        model.transition.grow_min_gap = gap
        print(f"grow: seed {n_seed} -> cap {max_rbf_eff}, min_gap {gap} steps "
              f"(~{(max_rbf_eff - n_seed) * gap / steps_per_epoch:.2f} epochs to fill)")

    rep = train_trials * EPOCHS
    C_snaps, C_steps = [], []
    # loss components logged every step (per-bin ELBO is grating-phase + Poisson noisy
    # -> smoothed over one cycle below); latent/basis logged coarsely.
    lstep, loss, recon, dyn, ent = [], [], [], [], []
    mu_log, mu_steps, nb_log = [], [], []
    add_steps, prev_nb = [], model.transition.velocity.feature.n_basis
    for res in online_filter_trials(model, rep, readout=ro, warmup_trials=WARMUP_TRIALS,
                                    seed_centers=lambda s: kmeans_centers(s, n_seed, width_scale=wscale)):
        nb_now = model.transition.velocity.feature.n_basis
        if nb_now > prev_nb:                                 # a center was added this step
            add_steps.append(res.step); prev_nb = nb_now
        if res.refreshed:
            C_snaps.append(model.decoder.decode.weight.detach().cpu().numpy().copy())
            C_steps.append(res.step)
        if not res.diverged:
            lstep.append(res.step); loss.append(res.loss); recon.append(res.recon)
            dyn.append(res.dynamics); ent.append(res.entropy)
        if res.step % 50 == 0 and not res.diverged:
            mu_log.append(res.mean.copy()); mu_steps.append(res.step)
            nb_log.append(model.transition.velocity.feature.n_basis)

    centers = model.transition.velocity.feature.centroid.detach().cpu().numpy()  # fixed after seed
    C_steps = np.asarray(C_steps)
    ang_consec = np.array([np.degrees(_principal_angle(C_snaps[i], C_snaps[i - 1]))
                           for i in range(1, len(C_snaps))])
    ang_cumul = np.array([np.degrees(_principal_angle(C_snaps[i], C_snaps[0]))
                          for i in range(len(C_snaps))])
    cond = np.array([np.linalg.cond(C) for C in C_snaps])
    mu = np.asarray(mu_log); mu_steps = np.asarray(mu_steps)
    # distance from filtered latent to nearest fixed RBF center
    d2c = np.array([np.linalg.norm(centers - m, axis=1).min() for m in mu])
    epoch = lambda s: np.asarray(s, dtype=float) / steps_per_epoch

    fig, ax = plt.subplots(2, 3, figsize=(12.0, 6.6))

    # --- readout subspace drift: cumulative (from seed) is the story; consecutive recedes
    ax[0, 0].plot(epoch(C_steps), ang_cumul, "-", color="C0", label="from seed")
    ax[0, 0].plot(epoch(C_steps[1:]), ang_consec, "-", color="0.6", lw=0.8, label="per refresh")
    ax[0, 0].set_title("readout C principal angle"); ax[0, 0].set_xlabel("epoch")
    ax[0, 0].set_ylabel("degrees"); ax[0, 0].legend()

    # --- RBF basis size, with the centers' add-events and the end of the first epoch
    ax[0, 1].plot(epoch(mu_steps), nb_log, color="0.2")
    ax[0, 1].set_title("RBF basis size"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("# centers")

    # --- ELBO components: per-bin (faint) under a one-grating-cycle EMA (bold)
    comps = [("-ELBO", np.asarray(loss), "0.15"), ("recon", np.asarray(recon), "C0"),
             ("dynamics", np.asarray(dyn), "C2"), ("entropy", np.asarray(ent), "C3")]
    ep_l = epoch(lstep)
    stride = max(1, len(ep_l) // 6000)                       # thin the raw trace for a light PDF
    for name, y, col in comps:
        ax[0, 2].plot(ep_l[::stride], y[::stride], color=col, lw=0.4, alpha=0.2)
        ax[0, 2].plot(ep_l[::stride], ema(y, cycle_bins)[::stride], color=col, lw=1.4, label=name)
    allc = np.concatenate([c[1] for c in comps])
    lo, hi = np.percentile(allc, [1, 99]); pad = 0.06 * (hi - lo + 1e-9)
    ax[0, 2].set_ylim(lo - pad, hi + pad)                    # clip onset transients off the scale
    ax[0, 2].set_title(f"ELBO (faint: per-bin; bold: EMA ~{cycle_bins:.0f}-bin cycle)")
    ax[0, 2].set_xlabel("epoch"); ax[0, 2].legend(ncol=2)

    # --- filtered latent, each dimension
    for d in range(mu.shape[1]):
        ax[1, 0].plot(epoch(mu_steps), mu[:, d], lw=0.6, alpha=0.85, label=f"x{d + 1}")
    ax[1, 0].set_title("filtered latent"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].legend(ncol=mu.shape[1])

    # --- latent spread (std over a sliding ~1-epoch window)
    w = max(10, steps_per_epoch // 50)
    run_std = np.array([mu[max(0, i - w):i + 1].std(0).mean() for i in range(len(mu))])
    ax[1, 1].plot(epoch(mu_steps), run_std, color="0.2")
    ax[1, 1].set_title("latent spread (running std)"); ax[1, 1].set_xlabel("epoch")

    # --- distance from the filtered latent to its nearest RBF center
    ax[1, 2].plot(epoch(mu_steps), d2c, color="0.2", lw=0.8)
    ax[1, 2].set_title("latent -> nearest RBF center"); ax[1, 2].set_xlabel("epoch")
    ax[1, 2].set_ylabel("latent distance")

    for axp in (ax[0, 1], ax[1, 2]):                         # add-events + end of first epoch
        for s in add_steps:
            axp.axvline(epoch(s), color="C3", lw=0.3, alpha=0.08)
        axp.axvline(1.0, color="0.5", ls=":", lw=0.9)
    if add_steps:
        ax[1, 2].plot([], [], color="C3", lw=1.2, label=f"RBF added (n={len(add_steps)})")
        ax[1, 2].plot([], [], color="0.5", ls=":", lw=0.9, label="end epoch 1")
        ax[1, 2].legend()

    tag2 = (f"grow (cap {max_rbf_eff}, w x{width_scale})" if grow else "fixed RBF")
    fig.suptitle(f"single-dir sVJF  -  dir {d_star:.0f} deg, L={latent_dim}, "
                 f"{column_norm}-norm C, {EPOCHS} ep  -  {tag2}", fontsize=10)
    sfx = ("_" + tag) if tag else ("_grow" if grow else "")
    fig.savefig(os.path.join(FIGS, f"single_dir_traincurves{sfx}.png"))
    fig.savefig(os.path.join(FIGS, f"single_dir_traincurves{sfx}.pdf")); plt.close(fig)

    print(f"dir={d_star:.0f}  grow={grow}  n_basis {nb_log[0]} -> {nb_log[-1]}")
    print(f"subspace drift from seed={ang_cumul[-1]:.1f} deg (last consec {ang_consec[-1]:.2f}); "
          f"cond C {cond[0]:.1f}->{cond[-1]:.1f}")
    print(f"dist-to-center: start {d2c[:50].mean():.3g} -> end {d2c[-50:].mean():.3g}")
    print("fig ->", os.path.join(FIGS, f"single_dir_traincurves{sfx}.png"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grow", action="store_true")
    ap.add_argument("--grow-thresh", type=float, default=0.5)
    ap.add_argument("--max-rbf", type=int, default=None)     # None -> rbf_base * 2^latent_dim
    ap.add_argument("--grow-min-gap", type=int, default=None)  # None -> spread adds across epoch 1
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--column-norm", type=str, default="eig", choices=["eig", "unit"])
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--rbf-base", type=int, default=25)      # cap = rbf_base * 2^latent_dim
    ap.add_argument("--width-scale", type=float, default=0.5)
    a = ap.parse_args()
    main(grow=a.grow, grow_thresh=a.grow_thresh, max_rbf=a.max_rbf, grow_min_gap=a.grow_min_gap,
         latent_dim=a.latent_dim, column_norm=a.column_norm, tag=a.tag,
         rbf_base=a.rbf_base, width_scale=a.width_scale)
