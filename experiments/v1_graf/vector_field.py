"""Visualize the learned RBF flow (velocity field) of the single-direction sVJF.

For the trained model, f(x) = per-step velocity = transition.velocity mean over a grid
of the 2-D latent. Streamlines of f, with the condition-averaged inferred cycle
(colored by time), the RBF centers, and an autonomous free-run overlaid. Two arms:
baseline (readout refresh on -> latent inflates) vs frozen readout (no refresh).
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask, kmeans_centers
from experiments.v1_graf.run_m1 import _infer_latent_paths
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter_trials

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
BIN_MS, T_MAX_MS, SEED, EPOCHS, WARMUP, NRBF = 10.0, 1400.0, 20260609, 20, 8, 100
matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 130, "savefig.bbox": "tight"})


def _train(train_trials, N, refresh_k):
    torch.manual_seed(SEED)
    rep = train_trials * EPOCHS
    cover = np.concatenate(train_trials[:WARMUP], 0)
    n_rbf = min(NRBF, cover.shape[0])
    model = VJF.make_model(ydim=N, xdim=2, udim=0, n_rbf=n_rbf, hidden_sizes=[64, 64],
                           likelihood="poisson", transition_flow="srrls", encoder="projection")
    ro = OnlineReadout(N, 2, smooth_tau=8.0, refresh_K=refresh_k, link="log")
    Cw, bw = ro.warm_start(cover)
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cw))
        model.decoder.decode.bias.copy_(torch.as_tensor(bw.reshape(-1)))
    for _ in online_filter_trials(model, rep, readout=ro, warmup_trials=WARMUP,
                                  seed_centers=lambda s: kmeans_centers(s, n_rbf)):
        pass
    return model, ro


@torch.no_grad()
def _field(model, lo, hi, n=28):
    gx = np.linspace(lo[0], hi[0], n); gy = np.linspace(lo[1], hi[1], n)
    GX, GY = np.meshgrid(gx, gy)
    grid = torch.as_tensor(np.stack([GX.ravel(), GY.ravel()], 1).astype(np.float32))
    vel = model.transition._velocity_mean(grid).cpu().numpy()    # per-step displacement f(x)
    U = vel[:, 0].reshape(n, n); V = vel[:, 1].reshape(n, n)
    return GX, GY, U, V


def main():
    os.makedirs(FIGS, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
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
    test_idx, train_idx = idx[:10], idx[10:]
    train_trials = [counts[i][:nt] for i in train_idx]
    test_trials = [counts[i][:nt] for i in test_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    for ax, (label, rk) in zip(axes, [("baseline (refresh on)", 500),
                                       ("frozen readout (no refresh)", 10 ** 9)]):
        model, ro = _train(train_trials, N, rk)
        paths = _infer_latent_paths(model, ro, test_trials)
        allp = np.concatenate(paths, 0)
        cyc = np.mean([p for p in paths], 0)                     # condition-avg cycle (nt,2)
        rng_ = np.ptp(allp, 0)
        lo = allp.min(0) - 0.1 * (rng_ + 1e-6)
        hi = allp.max(0) + 0.1 * (rng_ + 1e-6)
        GX, GY, U, V = _field(model, lo, hi)
        spd = np.hypot(U, V)
        ax.streamplot(GX, GY, U, V, color=spd, cmap="Greys", density=1.2, linewidth=0.7, arrowsize=0.8)
        t = np.arange(cyc.shape[0])
        ax.scatter(cyc[:, 0], cyc[:, 1], c=t, cmap="viridis", s=10, zorder=3, label="inferred cycle (time)")
        cen = model.transition.velocity.feature.centroid.detach().cpu().numpy()
        ax.plot(cen[:, 0], cen[:, 1], "rx", ms=3, alpha=0.4, zorder=2)
        # autonomous free-run from the cycle's first point (~3 grating cycles)
        with torch.no_grad():
            fr, _ = model.forecast(torch.as_tensor(cyc[0:1].astype(np.float32)), n_step=48)
        fr = fr.cpu().numpy()[:, 0, :]
        ax.plot(fr[:, 0], fr[:, 1], color="magenta", lw=1.3, zorder=4, label="free-run (48 steps)")
        ax.plot(fr[0, 0], fr[0, 1], "mo", ms=5, zorder=5)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel("latent 1"); ax.set_ylabel("latent 2")
        ax.set_title(f"{label}, n_basis={model.transition.velocity.feature.n_basis}")
        ax.legend(fontsize=7, loc="upper left")
        print(f"{label}: latent extent x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}]")
    fig.suptitle(f"Learned RBF flow f(x) (per-step velocity), single direction {d_star:.0f} deg, L=2, "
                 f"{EPOCHS} epochs")
    fig.savefig(os.path.join(FIGS, "vector_field.png"))
    fig.savefig(os.path.join(FIGS, "vector_field.pdf")); plt.close(fig)
    print("fig ->", os.path.join(FIGS, "vector_field.png"))


if __name__ == "__main__":
    main()
