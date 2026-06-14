"""PCA trajectories of the Graf V1 population over the analyzed (stimulus) window.

PCA is fit on the per-direction trial-averaged log-rate feature log(smoothed counts + c)
-- the same link-matched feature the sVJF readout's CCIPCA tracks -- over the
\\SI{0}{}--\\SI{1280}{ms} stimulus window. Shows the cyclic trajectory the model is meant to
capture: the PC1-PC2 limit cycle for the strongest direction, the components over time
(the 6.25 Hz oscillation), and the per-direction trajectories (the orientation structure).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask
from experiments.v1_graf.figstyle import set_style, FW, dir_color

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
BIN_MS, STIM_MS = 10.0, 1280.0


def main():
    set_style()
    arr = load_array(5); dirs = arr["ori"]
    counts = bin_spikes(arr["spk_times"], bin_ms=BIN_MS)
    mask, _ = well_tuned_mask(counts, dirs); counts = counts[:, :, mask]
    nt = int(round(STIM_MS / BIN_MS))
    udir = np.unique(dirs)
    # per-direction trial-averaged log-rate feature over the stimulus window
    psth = np.stack([counts[dirs == d][:, :nt, :].mean(0) for d in udir], 0)   # (D, nt, N)
    feat = gaussian_filter1d(np.log(psth + 1e-2), sigma=1.0, axis=1)           # mild temporal smoothing
    M = feat.reshape(-1, feat.shape[-1])                                       # (D*nt, N)
    mu = M.mean(0)
    _, S, Vt = np.linalg.svd(M - mu, full_matrices=False)
    P = Vt[:3].T                                                               # (N, 3) top-3 PCs
    var3 = (S[:3] ** 2 / (S ** 2).sum())
    Z = (feat - mu) @ P                                                        # (D, nt, 3) trajectories
    d_star = int(np.argmax(psth[:, :, :].mean((1, 2))))                        # strongest direction index
    t = np.arange(nt) * BIN_MS

    fig, ax = plt.subplots(1, 3, figsize=(FW(1.0), 2.8))
    # (a) PC1-PC2 limit cycle for the strongest direction, colored by time
    z = Z[d_star]
    ax[0].scatter(z[:, 0], z[:, 1], c=t, cmap="viridis", s=6)
    ax[0].plot(z[:, 0], z[:, 1], color="0.6", lw=0.5, zorder=0)
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    ax[0].set_title(f"dir {udir[d_star]:.0f} deg cycle")
    # (b) PCs over time (the 6.25 Hz oscillation) -- time axis
    for k, c in zip(range(3), ("C0", "C1", "C2")):
        ax[1].plot(t, z[:, k], color=c, lw=1.0, label=f"PC{k+1}")
    ax[1].set_xlabel("time in trial (ms)"); ax[1].set_ylabel("score"); ax[1].legend(fontsize=6.5, ncol=3)
    ax[1].set_title("PCs over time")
    # (c) per-direction PC1-PC2 trajectories (orientation structure)
    for di in range(0, len(udir), 12):                                        # 6 well-separated directions
        ax[2].plot(Z[di][:, 0], Z[di][:, 1], lw=1.0, alpha=0.85,
                   color=dir_color(udir[di]), label=f"{udir[di]:.0f}")
    ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2"); ax[2].set_title("per-direction")
    ax[2].legend(fontsize=5.5, ncol=2, title="deg", title_fontsize=5.5)
    fig.suptitle(f"population PCA over the stimulus window "
                 f"(var PC1-3 = {var3[0]:.2f}/{var3[1]:.2f}/{var3[2]:.2f})", fontsize=9)
    out = os.path.join(FIGS, "pca_traj.png")
    fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
    print(f"PCA var(PC1-3)={np.round(var3,3)}; strongest dir={udir[d_star]:.0f}; fig -> {out}")


if __name__ == "__main__":
    main()
