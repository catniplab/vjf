"""Load the Graf et al. (2011) macaque V1 drifting-grating dataset.

array_{1..5}.mat hold spk_times (object (N,3600) cells of spike times in ms),
ori (3600,) grating direction in deg, tf_tot (temporal freq, 6.25 Hz),
neur_param (N,2). Trial = 2560 ms (first 1280 ms stimulus, then blank).
"""
from __future__ import annotations
import os
import numpy as np
from scipy.io import loadmat

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
