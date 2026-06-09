import numpy as np
import pytest
from experiments.v1_graf.graf_loader import bin_spikes
from experiments.v1_graf.graf_loader import tuning_curve, well_tuned_mask, signal_metric
from experiments.v1_graf.graf_loader import kmeans_centers


def _fake_spk(N=3, n_trial=4, seed=20260609):
    rng = np.random.default_rng(seed)
    spk = np.empty((N, n_trial), dtype=object)
    for i in range(N):
        for c in range(n_trial):
            spk[i, c] = (np.sort(rng.uniform(0, 2560, size=rng.integers(0, 20)))
                         .reshape(-1, 1))
    return spk


def test_bin_spikes_shape_and_counts():
    spk = _fake_spk()
    counts = bin_spikes(spk, bin_ms=10.0, t_total_ms=2560.0)
    assert counts.shape == (4, 256, 3)          # (trial, bin, neuron)
    assert counts.dtype == np.float32
    # total binned spikes == total spike times within [0, t_total)
    N, n_trial = spk.shape
    raw = sum(np.asarray(spk[i, c]).size for i in range(N) for c in range(n_trial))
    assert int(counts.sum()) == raw


def test_tuning_and_quality_mask():
    rng = np.random.default_rng(20260609)
    ori_axis = np.arange(0, 360, 5.0)
    dirs = np.repeat(ori_axis, 50)                 # (3600,) sorted; 50 reps
    rad = np.deg2rad(dirs)
    rate_tuned = 1.0 + 0.9 * np.cos(2 * (rad - np.deg2rad(40)))   # orientation (180-periodic)
    spk = np.empty((2, dirs.size), dtype=object)
    for c in range(dirs.size):
        spk[0, c] = np.sort(rng.uniform(0, 1280, int(rng.poisson(8 * rate_tuned[c])))).reshape(-1, 1)
        spk[1, c] = np.sort(rng.uniform(0, 1280, int(rng.poisson(8 * 1.0)))).reshape(-1, 1)
    counts = bin_spikes(spk, bin_ms=10.0)
    tc, axis = tuning_curve(counts, dirs, stim_only=True)
    assert tc.shape == (2, 72) and axis.shape == (72,)
    mask, r2 = well_tuned_mask(counts, dirs, r2_thresh=0.75)
    assert mask[0] and not mask[1]                  # tuned kept, flat rejected
    assert r2[0] > 0.75 and r2[1] < 0.5


def test_signal_metric_monotone():
    rng = np.random.default_rng(20260609)
    N, n_trial, n_bin = 2, 8, 256
    # low-rate array: ~1 spike/bin per neuron per trial on average
    counts_lo = rng.binomial(1, 0.05, size=(n_trial, n_bin, N)).astype(np.float32)
    # high-rate array: same shape, clearly more spikes per bin
    counts_hi = rng.binomial(1, 0.40, size=(n_trial, n_bin, N)).astype(np.float32)

    sm_lo = signal_metric(counts_lo)
    sm_hi = signal_metric(counts_hi)

    # key/shape assertions
    assert set(sm_lo) >= {"N", "mean_rate_hz", "total_spikes"}
    assert sm_lo["N"] == N

    # monotonicity: higher spike probability -> higher mean_rate_hz
    assert sm_hi["mean_rate_hz"] > sm_lo["mean_rate_hz"]


def test_kmeans_centers_shapes():
    rng = np.random.default_rng(20260609)
    states = rng.standard_normal((500, 3)).astype(np.float32)
    c, logw = kmeans_centers(states, n_rbf=20)
    assert c.shape == (20, 3) and logw.shape == (20,)
    assert np.isfinite(logw).all()


def test_kmeans_centers_requires_enough_states():
    # fewer visited states than requested centers should fail clearly, not silently
    # clamp the cluster count (which would return < n_rbf centers).
    rng = np.random.default_rng(20260609)
    states = rng.standard_normal((10, 3)).astype(np.float32)
    with pytest.raises(ValueError):
        kmeans_centers(states, n_rbf=20)
