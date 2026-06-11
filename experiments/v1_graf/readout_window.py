"""Is the warm-start readout C onset-dominated, and does the window matter?

For the chosen direction, compute the readout's link feature (log EMA, per trial) and
PCA-2 it over different windows:
  - full window, 8 trials   (== the actual warm-start C)
  - full window, 40 trials  (more trials -> tests window COUNT)
  - sustained only (0.3-1.4 s, onset excluded) -> tests onset DOMINANCE
  - onset only (0-0.3 s)
Reports principal angles + the fraction of SUSTAINED-period feature variance each
subspace captures, and plots the population feature norm over time (the onset spike).
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask
from vjf.readout import OnlineReadout, _principal_angle

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
BIN_MS, T_MAX_MS, SEED = 10.0, 1400.0, 20260609
ONSET_MS = 300.0
matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 130, "savefig.bbox": "tight"})


def main():
    os.makedirs(FIGS, exist_ok=True)
    rng = np.random.default_rng(SEED)
    arr = load_array(5)
    counts = bin_spikes(arr["spk_times"], bin_ms=BIN_MS)
    dirs = arr["ori"]
    mask, _ = well_tuned_mask(counts, dirs)
    counts = counts[:, :, mask]
    N = counts.shape[2]
    nt = int(round(T_MAX_MS / BIN_MS))
    n_on = int(round(ONSET_MS / BIN_MS))
    keep = np.unique(dirs)
    d_star = float(max(keep, key=lambda d: counts[dirs == d][:, :nt, :].mean()))
    idx = np.where(dirs == d_star)[0]; rng.shuffle(idx)

    ro = OnlineReadout(N, 2, smooth_tau=8.0, link="log")

    def feat_trial(tc):
        ro.nu = np.zeros(N)
        return np.stack([ro._feat(tc[t]) for t in range(tc.shape[0])], 0)   # (nt, N) log-rate

    F = np.stack([feat_trial(counts[i][:nt]) for i in idx], 0)              # (50, nt, N)

    def pca2(mat):
        m = mat - mat.mean(0)
        return np.linalg.svd(m, full_matrices=False)[2][:2]                 # (2, N)

    C_full8 = pca2(F[:8].reshape(-1, N))
    C_full40 = pca2(F.reshape(-1, N))
    C_sust = pca2(F[:, n_on:].reshape(-1, N))
    C_onset = pca2(F[:, :n_on].reshape(-1, N))

    sust = F[:, n_on:].reshape(-1, N)
    sust = sust - sust.mean(0)

    def varcap(C):                                                          # frac of SUSTAINED var in subspace C
        p = sust @ C.T
        return float((p ** 2).sum() / ((sust ** 2).sum() + 1e-12))

    ang = lambda a, b: float(np.degrees(_principal_angle(a.T, b.T)))
    print(f"direction {d_star:.0f} deg, N={N}, onset = first {n_on} bins ({ONSET_MS:.0f} ms)")
    print(f"principal angle  full8 vs full40   = {ang(C_full8, C_full40):.1f} deg  (window COUNT effect)")
    print(f"principal angle  full8 vs sustained = {ang(C_full8, C_sust):.1f} deg  (onset DOMINANCE)")
    print(f"principal angle  onset vs sustained = {ang(C_onset, C_sust):.1f} deg")
    print(f"sustained-variance captured by: full8={varcap(C_full8):.3f}  full40={varcap(C_full40):.3f}"
          f"  sustained={varcap(C_sust):.3f}  onset={varcap(C_onset):.3f}")

    # population feature norm over time (trial-averaged), shows the onset transient
    psth = F.mean(0)                                                        # (nt, N)
    norm_t = np.linalg.norm(psth - psth.mean(0), axis=1)
    t_ms = np.arange(nt) * BIN_MS

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax[0].plot(t_ms, norm_t, lw=1.2); ax[0].axvspan(0, ONSET_MS, color="orange", alpha=0.15, label="onset window")
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("population feature norm (trial-avg)")
    ax[0].set_title("Onset transient dominates the variance"); ax[0].legend(fontsize=7)
    labels = ["full\n(8 tr)", "full\n(40 tr)", "sustained\n(40 tr)", "onset\n(40 tr)"]
    vals = [varcap(C_full8), varcap(C_full40), varcap(C_sust), varcap(C_onset)]
    ax[1].bar(range(4), vals, color=["#c44", "#e88", "#4a4", "#88c"])
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("frac of SUSTAINED variance captured"); ax[1].set_ylim(0, 1.0)
    ax[1].set_title("Warm-start C (full) under-captures the sustained response")
    fig.suptitle(f"Readout-window analysis (dir {d_star:.0f} deg, L=2)")
    fig.savefig(os.path.join(FIGS, "readout_window.png"))
    fig.savefig(os.path.join(FIGS, "readout_window.pdf")); plt.close(fig)
    print("fig ->", os.path.join(FIGS, "readout_window.png"))


if __name__ == "__main__":
    main()
