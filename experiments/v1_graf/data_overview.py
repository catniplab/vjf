"""Data-characterization figures for the Graf V1 dataset (array_5): a single-trial raster
with the rough SNR, the trial-averaged population PSTH over a trial, a per-neuron PSTH
heatmap (time axis), and tuning curves for selected neurons.

Rough SNR via neuroFisherSNR: fit a rank-d log-linear Poisson model exp(Cx+b) to the
trial-averaged log-rate (the stimulus-locked PSTH) and report the instantaneous Fisher
SNR bound (how well the d-dim latent is estimable from the spikes).
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import neurofisherSNR as nf

from experiments.v1_graf.graf_loader import (
    load_array, bin_spikes, well_tuned_mask, tuning_curve, _von_mises2)
from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
DATA = os.path.join(HERE, "report_m1", "data")
BIN_MS, STIM_MS, TRIAL_MS = 10.0, 1280.0, 2560.0


def rough_snr(psth_stim, dims=(2, 3)):
    """rank-d log-linear-Poisson Fisher SNR bound of the trial-averaged stim-window rate."""
    l = np.log(psth_stim + 1e-2); b = l.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(l - b, full_matrices=False)
    out = {}
    for d in dims:
        x = U[:, :d] * S[:d]
        snr = float(nf.SNR_bound_instantaneous(x - x.mean(0), Vt[:d], b))
        out[d] = {"snr": snr, "dB": float(nf.power_to_dB(snr)),
                  "R2": float(nf.powerDb_to_R2(nf.power_to_dB(snr)))}
    return out


def fit_vm2(axis, y):
    """Refit the sum-of-two-von-Mises tuning model (same init/bounds as well_tuned_mask).
    Returns (popt, R2_of_the_tuning_fit) or (None, 0.0) on failure."""
    amp = float(y.max() - y.min()); pk = float(axis[int(np.argmax(y))])
    p0 = [float(y.min()), amp, pk, 2.0, 0.5 * amp, (pk + 180) % 360, 2.0]
    bounds = ([0, 0, 0, 0, 0, 0, 0], [np.inf, np.inf, 360, 20, np.inf, 360, 20])
    try:
        popt, _ = curve_fit(_von_mises2, axis, y, p0=p0, bounds=bounds, maxfev=10000)
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ((y - _von_mises2(axis, *popt)) ** 2).sum() / (ss + 1e-12) if ss > 1e-9 else 0.0
        return popt, float(r2)
    except (RuntimeError, ValueError):
        return None, 0.0


def main():
    set_style(); os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(20260614)
    arr = load_array(5)
    spk = arr["spk_times"]; dirs = arr["ori"]
    counts = bin_spikes(spk, bin_ms=BIN_MS)
    mask, r2 = well_tuned_mask(counts, dirs)
    counts_m = counts[:, :, mask]; spk_m = spk[mask]; r2_m = r2[mask]
    N = counts_m.shape[2]
    n_full = int(round(TRIAL_MS / BIN_MS)); n_stim = int(round(STIM_MS / BIN_MS))
    d_star = float(max(np.unique(dirs), key=lambda d: counts_m[dirs == d][:, :n_stim, :].mean()))
    idx = np.where(dirs == d_star)[0]

    psth = counts_m[idx].mean(0)                              # (n_full, N) trial-avg counts/bin
    snr = rough_snr(psth[:n_stim])
    with open(os.path.join(DATA, "snr.json"), "w") as fh:
        json.dump({"array": 5, "direction_deg": d_star, "n_neurons": N, "snr": snr,
                   "mean_rate_hz": float(psth[:n_stim].mean() / (BIN_MS / 1000)),
                   "peak_rate_hz": float(psth.max() / (BIN_MS / 1000))}, fh, indent=2)
    snr3 = snr[3]
    t = np.arange(n_full) * BIN_MS

    # --- Figure 1: single-trial raster + population PSTH + per-neuron PSTH heatmap (time axis)
    order = np.argsort(psth[:n_stim].argmax(0))              # sort neurons by PSTH peak time
    fig, ax = plt.subplots(3, 1, figsize=(FW(1.0), 5.2), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 2]})
    trial = int(idx[0])                                      # one example trial
    raster = [np.ravel(spk_m[n, trial]) for n in order]
    ax[0].eventplot(raster, colors="0.15", lineoffsets=np.arange(N), linelengths=0.8, linewidths=0.4)
    ax[0].set_ylim(-1, N); ax[0].set_ylabel("neuron (PSTH-peak order)")
    ax[0].set_title(f"array 5, dir {d_star:.0f} deg, $N{{=}}{N}$ well-tuned  |  rough SNR "
                    f"{snr3['dB']:.1f} dB ($R^2{{\\approx}}{snr3['R2']:.2f}$, $d{{=}}3$)")
    ax[1].plot(t, psth.mean(1) / (BIN_MS / 1000), color="C0", lw=1.0)
    ax[1].set_ylabel("pop. rate (Hz)")
    im = ax[2].imshow((psth[:, order] / (BIN_MS / 1000)).T, aspect="auto", origin="lower",
                      extent=[0, TRIAL_MS, 0, N], cmap="magma", interpolation="nearest")
    ax[2].set_ylabel("neuron"); ax[2].set_xlabel("time in trial (ms)")
    fig.colorbar(im, ax=ax[2], pad=0.01, label="rate (Hz)")
    for a in ax:                                             # shade the stimulus (drifting grating) window
        a.axvspan(0, STIM_MS, color="C1", alpha=0.06, lw=0)
    fig.savefig(os.path.join(FIGS, "data_overview.png"))
    fig.savefig(os.path.join(FIGS, "data_overview.pdf")); plt.close(fig)

    # --- Figure 2: tuning curves for selected well-tuned neurons
    tc, tc_axis = tuning_curve(counts_m, dirs, bin_ms=BIN_MS)  # tc (N, n_dir) Hz; tc_axis (n_dir,) deg
    sel = np.argsort(r2_m)[::-1][:6]                         # the 6 best-fit (most tuned) neurons
    fine = np.linspace(0, 360, 361)
    fig2, ax2 = plt.subplots(2, 3, figsize=(FW(1.0), 3.4), sharex=True)
    for a, nidx in zip(ax2.ravel(), sel):
        popt, r2fit = fit_vm2(tc_axis, tc[nidx])
        a.plot(tc_axis, tc[nidx], "o", ms=2.5, color="0.35", label="data")
        if popt is not None:                                # overlay the double-von-Mises fit
            a.plot(fine, _von_mises2(fine, *popt), color="C3", lw=1.2, label="von Mises$_2$ fit")
        a.set_title(f"neuron {int(nidx)} (tuning-fit $R^2{{=}}{r2fit:.2f}$)", fontsize=8)
    ax2.ravel()[0].legend(fontsize=6, loc="upper right")
    for a in ax2[-1]:
        a.set_xlabel("direction (deg)"); a.set_xticks([0, 180, 360])
    for a in ax2[:, 0]:
        a.set_ylabel("rate (Hz)")
    fig2.savefig(os.path.join(FIGS, "tuning_curves.png"))
    fig2.savefig(os.path.join(FIGS, "tuning_curves.pdf")); plt.close(fig2)

    print(f"dir={d_star:.0f} N={N} SNR(d=3)={snr3['dB']:.1f}dB R2={snr3['R2']:.2f} "
          f"SNR(d=2)={snr[2]['dB']:.1f}dB; figs -> data_overview.png, tuning_curves.png")


if __name__ == "__main__":
    main()
