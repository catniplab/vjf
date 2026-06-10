"""Data-validation: 3D PCA of the Graf V1 population (no VJF).

sqrt-transform + mild temporal Gaussian smoothing, PCA fit on the condition
averages, directions 0-180 deg only, circular colormap. Renders single-trial
trajectories and condition-averaged trajectories (one per direction).
"""
from __future__ import annotations
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
BIN_MS = 10.0
SMOOTH_BINS = 2.5          # ~25 ms Gaussian sigma over time
T_MAX_MS = 1400.0          # signal interval (vLGP Fig 8)
matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 130, "savefig.bbox": "tight"})


def _color(d):
    return cm.hsv((float(d) % 180.0) / 180.0)   # circular over orientation (180-periodic)


def main():
    os.makedirs(FIGS, exist_ok=True)
    arr = load_array(5)
    counts = bin_spikes(arr["spk_times"], bin_ms=BIN_MS)
    dirs = arr["ori"]
    mask, _ = well_tuned_mask(counts, dirs)
    counts = counts[:, :, mask]
    N = counts.shape[2]
    nt = int(round(T_MAX_MS / BIN_MS))
    counts = counts[:, :nt, :]

    # sqrt (variance-stabilize) + mild temporal smoothing, per trial/neuron over time
    x = gaussian_filter1d(np.sqrt(counts.astype(np.float64)), SMOOTH_BINS, axis=1)

    dir_all = np.unique(dirs)
    sel = dir_all[(dir_all >= 0) & (dir_all < 180)]          # 0,5,...,175 deg
    # condition-averaged PSTH per selected direction
    psth = np.stack([x[dirs == d].mean(0) for d in sel], 0)  # (D, nt, N)

    # PCA fit on condition averages -> top 3 PCs
    M = psth.reshape(-1, N)
    mu = M.mean(0)
    _, _, vt = np.linalg.svd(M - mu, full_matrices=False)
    P = vt[:3].T
    proj_avg = (psth - mu) @ P                               # (D, nt, 3)

    sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(0, 180))

    # ---- condition-averaged trajectories ----
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for i, d in enumerate(sel):
        t = proj_avg[i]
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=_color(d), lw=1.4)
        ax.scatter(*t[0], color=_color(d), s=14)             # stimulus onset
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.view_init(elev=22, azim=-60)
    ax.set_title(f"Condition-averaged population PCA trajectories\n"
                 f"directions 0-175 deg, {N} well-tuned neurons, sqrt+smooth, 0-1400 ms")
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.10); cb.set_label("direction (deg)")
    fig.savefig(os.path.join(FIGS, "datacheck_avg_3d.png"))
    fig.savefig(os.path.join(FIGS, "datacheck_avg_3d.pdf")); plt.close(fig)

    # ---- single-trial trajectories (a few trials per direction) ----
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(20260610)
    for d in sel:
        idx = np.where(dirs == d)[0]
        for j in rng.choice(idx, size=min(3, len(idx)), replace=False):
            t = (x[j] - mu) @ P
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=_color(d), lw=0.5, alpha=0.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.view_init(elev=22, azim=-60)
    ax.set_title(f"Single-trial PCA trajectories (3 trials/direction)\n"
                 f"directions 0-175 deg, {N} well-tuned neurons, sqrt+smooth, 0-1400 ms")
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.10); cb.set_label("direction (deg)")
    fig.savefig(os.path.join(FIGS, "datacheck_single_3d.png"))
    fig.savefig(os.path.join(FIGS, "datacheck_single_3d.pdf")); plt.close(fig)

    var = (np.cumsum(np.linalg.svd(M - mu, compute_uv=False) ** 2) /
           (np.linalg.svd(M - mu, compute_uv=False) ** 2).sum())[:3]
    print(f"N well-tuned={N}, directions={len(sel)} (0-175 deg), bins={nt} (0-1400 ms)")
    print(f"PCA var explained (cum, top3) = {np.round(var, 3)}")
    print("figs ->", FIGS)


if __name__ == "__main__":
    main()
