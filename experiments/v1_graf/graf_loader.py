"""Load the Graf et al. (2011) macaque V1 drifting-grating dataset.

array_{1..5}.mat hold spk_times (object (N,3600) cells of spike times in ms),
ori (3600,) grating direction in deg, tf_tot (temporal freq, 6.25 Hz),
neur_param (N,2). Trial = 2560 ms (first 1280 ms stimulus, then blank).
"""
from __future__ import annotations
import os
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit

N_REP, N_ORI, T_TOTAL_MS, STIM_MS = 50, 72, 2560.0, 1280.0
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def load_array(array_num: int, data_dir: str = DATA_DIR) -> dict:
    """Return raw fields for one array: spk_times (N,3600) object, ori (3600,),
    tf (float), neur_param (N,2)."""
    if array_num not in range(1, 6):
        raise ValueError(f"array_num must be in 1..5, got {array_num}")
    m = loadmat(os.path.join(data_dir, f"array_{array_num}.mat"))
    return {
        "spk_times": m["spk_times"],
        "ori": np.ravel(m["ori"]).astype(float),
        "tf": float(np.ravel(m["tf_tot"])[0]),
        "neur_param": m["neur_param"],
    }


def bin_spikes(spk_times: np.ndarray, bin_ms: float = 10.0,
               t_total_ms: float = T_TOTAL_MS) -> np.ndarray:
    """Histogram spike-time cells into counts (trial, bin, neuron), float32.
    Spikes are clipped to [0, t_total_ms)."""
    N, n_trial = spk_times.shape
    n_bin = int(round(t_total_ms / bin_ms))
    edges = np.arange(n_bin + 1) * bin_ms
    counts = np.zeros((n_trial, n_bin, N), dtype=np.float32)
    # one-time load cost (~4-6 s for array_5: N=148, 3600 trials); vectorize if it ever matters
    for i in range(N):
        for c in range(n_trial):
            t = np.asarray(spk_times[i, c], dtype=float).ravel()
            t = t[(t >= 0.0) & (t < t_total_ms)]
            counts[c, :, i] = np.histogram(t, bins=edges)[0]
    return counts


def tuning_curve(counts: np.ndarray, ori: np.ndarray, *, bin_ms: float = 10.0,
                 stim_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Mean spike rate (Hz) per neuron per direction. counts shape: (trial, bin, neuron).
    stim_only restricts to the first STIM_MS of each trial."""
    n_bin = counts.shape[1]
    sl = slice(0, int(round(STIM_MS / bin_ms))) if stim_only else slice(None)
    win_s = (n_bin if sl.stop is None else sl.stop) * bin_ms / 1000.0
    per_trial = counts[:, sl, :].sum(1)                       # (trial, neuron) counts
    axis = np.unique(ori)
    tc = np.stack([per_trial[ori == d].mean(0) for d in axis], axis=1)  # (neuron, n_ori)
    return tc / win_s, axis                                   # Hz


def _von_mises2(theta, b, a1, mu1, k1, a2, mu2, k2):
    t = np.deg2rad(theta)
    return (b + a1 * np.exp(k1 * (np.cos(t - np.deg2rad(mu1)) - 1))
              + a2 * np.exp(k2 * (np.cos(t - np.deg2rad(mu2)) - 1)))


def well_tuned_mask(counts: np.ndarray, ori: np.ndarray, *, bin_ms: float = 10.0,
                    r2_thresh: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Keep neurons whose direction tuning is fit (R^2 >= thresh) by a sum of two
    von Mises bumps (~180 deg apart), mirroring vLGP's selection."""
    tc, axis = tuning_curve(counts, ori, bin_ms=bin_ms, stim_only=True)
    N = tc.shape[0]
    r2 = np.zeros(N)
    # bounds: b>=0, a1>=0, mu1 in [0,360), k1>=0, a2>=0, mu2 in [0,360), k2>=0
    bounds = ([0, 0, 0, 0, 0, 0, 0], [np.inf, np.inf, 360, 20, np.inf, 360, 20])
    for n in range(N):
        y = tc[n]
        if y.max() <= 0:
            continue
        amp = float(y.max() - y.min())
        pk = axis[int(np.argmax(y))]
        # amplitude initial guess is range above baseline, not the raw max
        p0 = [float(y.min()), amp, float(pk), 2.0, 0.5 * amp, float((pk + 180) % 360), 2.0]
        try:
            popt, _ = curve_fit(_von_mises2, axis, y, p0=p0, bounds=bounds, maxfev=10000)
            yhat = _von_mises2(axis, *popt)
            ss = ((y - y.mean()) ** 2).sum()
            if ss < 1e-9:
                r2[n] = 0.0
                continue
            r2[n] = 1.0 - ((y - yhat) ** 2).sum() / (ss + 1e-12)
        except (RuntimeError, ValueError):
            r2[n] = 0.0
    return r2 >= r2_thresh, r2


def signal_metric(counts: np.ndarray, *, t_total_ms: float = T_TOTAL_MS) -> dict:
    """Per-array signal proxy for strongest-first ordering.
    mean_rate_hz is the per-neuron, per-trial mean firing rate."""
    n_trial, _, N = counts.shape
    return {"N": N, "total_spikes": float(counts.sum()),
            "mean_rate_hz": float(counts.sum() / (N * n_trial) / (t_total_ms / 1000.0))}


def kmeans_centers(states: np.ndarray, n_rbf: int, *, width_scale: float = 1.0,
                   seed: int = 20260609):
    """Place n_rbf RBF centers by k-means on visited latent states (states: (T, xdim)),
    widths = width_scale * median nearest-center distance. Returns (centers, logwidths)
    as float32 numpy arrays for RBFDS.initialize(rbf_centers=..., rbf_logwidths=...)."""
    from sklearn.cluster import KMeans
    if states.shape[0] < n_rbf:
        raise ValueError(
            f"need >= n_rbf states to seed {n_rbf} centers, got {states.shape[0]}")
    km = KMeans(n_clusters=n_rbf, n_init=4, random_state=seed).fit(states)
    c = km.cluster_centers_.astype(np.float32)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn_dist = d.min(1)
    w = (width_scale * np.median(nn_dist)).astype(np.float32)
    logw = np.log(np.full(n_rbf, max(float(w), 1e-3), dtype=np.float32))
    return c, logw
