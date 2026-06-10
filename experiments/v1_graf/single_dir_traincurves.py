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
from vjf.model import VJF
from vjf.readout import OnlineReadout, _principal_angle
from vjf.realtime import online_filter_trials

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
BIN_MS, T_MAX_MS, SEED = 10.0, 1400.0, 20260609
EPOCHS, WARMUP_TRIALS, N_RBF, HIDDEN, REFRESH_K = 50, 8, 100, [64, 64], 500
matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 130, "savefig.bbox": "tight"})


def main(grow=False, grow_thresh=0.5, max_rbf=400, grow_min_gap=1):
    os.makedirs(FIGS, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    arr = load_array(5)
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
    n_rbf = min(N_RBF, cover.shape[0])
    model = VJF.make_model(ydim=N, xdim=2, udim=0, n_rbf=n_rbf, hidden_sizes=HIDDEN,
                           likelihood="poisson", transition_flow="srrls", encoder="projection")
    ro = OnlineReadout(N, 2, smooth_tau=8.0, refresh_K=REFRESH_K, link="log")
    Cw, bw = ro.warm_start(cover)
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cw))
        model.decoder.decode.bias.copy_(torch.as_tensor(bw.reshape(-1)))

    if grow:
        model.transition.grow_rbf = True
        model.transition.grow_thresh = grow_thresh
        model.transition.max_rbf = max_rbf
        model.transition.grow_min_gap = grow_min_gap

    rep = train_trials * EPOCHS
    C_snaps, C_steps = [], []
    st, loss, recon, dyn, ent, mu_log, mu_steps, nb_log = [], [], [], [], [], [], [], []
    for res in online_filter_trials(model, rep, readout=ro, warmup_trials=WARMUP_TRIALS,
                                    seed_centers=lambda s: kmeans_centers(s, n_rbf)):
        if res.refreshed:
            C_snaps.append(model.decoder.decode.weight.detach().cpu().numpy().copy())
            C_steps.append(res.step)
        if res.step % 50 == 0 and not res.diverged:
            st.append(res.step); loss.append(res.loss); recon.append(res.recon)
            dyn.append(res.dynamics); ent.append(res.entropy)
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
    epoch = lambda s: s / steps_per_epoch

    fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.0))
    ax[0, 0].plot(epoch(C_steps[1:]), ang_consec, ".-", ms=3, label="consecutive (per refresh)")
    ax[0, 0].plot(epoch(C_steps), ang_cumul, ".-", ms=3, label="cumulative (from seed)")
    ax[0, 0].set_title("Readout C subspace principal angle"); ax[0, 0].set_xlabel("epoch")
    ax[0, 0].set_ylabel("degrees"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].plot(epoch(np.array(mu_steps)), nb_log, lw=1.2, color="C3")
    ax[0, 1].set_title("RBF basis size (n_basis)"); ax[0, 1].set_xlabel("epoch")
    ax[0, 1].set_ylabel("# centers")
    ax[0, 2].plot(epoch(np.array(st)), loss, lw=0.7, label="-ELBO")
    ax[0, 2].plot(epoch(np.array(st)), recon, lw=0.7, label="recon")
    ax[0, 2].plot(epoch(np.array(st)), dyn, lw=0.7, label="dynamics")
    ax[0, 2].plot(epoch(np.array(st)), ent, lw=0.7, label="entropy")
    ax[0, 2].set_title("ELBO components"); ax[0, 2].set_xlabel("epoch"); ax[0, 2].legend(fontsize=7)
    ax[1, 0].plot(epoch(mu_steps), mu[:, 0], lw=0.5, label="latent 1")
    ax[1, 0].plot(epoch(mu_steps), mu[:, 1], lw=0.5, label="latent 2")
    ax[1, 0].set_title("Filtered latent (per step)"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].legend(fontsize=7)
    # running spread (std over a sliding window of ~1 epoch of samples)
    w = max(10, steps_per_epoch // 50)
    run_std = np.array([mu[max(0, i - w):i + 1].std(0).mean() for i in range(len(mu))])
    ax[1, 1].plot(epoch(mu_steps), run_std, lw=0.8)
    ax[1, 1].set_title("Latent spread (running std)"); ax[1, 1].set_xlabel("epoch")
    ax[1, 2].plot(epoch(mu_steps), d2c, lw=0.5)
    ax[1, 2].set_title("Dist: filtered latent -> nearest RBF center"); ax[1, 2].set_xlabel("epoch")
    ax[1, 2].set_ylabel("latent distance")
    tag = (f"GROW RBF (thresh {grow_thresh}, cap {max_rbf})" if grow else "fixed RBF")
    fig.suptitle(f"Single-direction sVJF training curves (dir {d_star:.0f} deg, L=2, "
                 f"{EPOCHS} epochs, K={REFRESH_K}) -- {tag}")
    sfx = "_grow" if grow else ""
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
    ap.add_argument("--max-rbf", type=int, default=400)
    ap.add_argument("--grow-min-gap", type=int, default=1)
    a = ap.parse_args()
    main(grow=a.grow, grow_thresh=a.grow_thresh, max_rbf=a.max_rbf, grow_min_gap=a.grow_min_gap)
