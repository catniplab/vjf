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


from experiments.v1_graf.eval import predictive_ll_bits_per_spike
def test_pll_perfect_vs_baseline():
    rng = np.random.default_rng(20260609)
    ybar = 0.1
    y = rng.poisson(ybar, size=(5, 100, 8)).astype(np.float32)
    # model that predicts the true generating rate equals the mean-rate baseline -> ~0 bits/spike
    lam_true = np.full_like(y, ybar)
    pll = predictive_ll_bits_per_spike(y, lam_true, ybar)
    assert abs(pll) < 0.05
    lam_better = np.clip(y.mean(0, keepdims=True).repeat(5, 0), 1e-3, None)  # PSTH predictor
    assert predictive_ll_bits_per_spike(y, lam_better, ybar) >= -0.05


import torch
from vjf.model import VJF
from vjf.readout import OnlineReadout
from experiments.v1_graf.eval import leave_one_neuron_rates

def test_leave_one_neuron_rates_emastate_isolated():
    rng = np.random.default_rng(20260609)
    torch.manual_seed(0)
    m = VJF.make_model(8, 2, 0, 8, hidden_sizes=[8, 8], likelihood="poisson",
                       transition_flow="srrls", encoder="projection")
    ro = OnlineReadout(8, 2, refresh_K=1000)
    win = rng.poisson(0.3, size=(15, 8)).astype(np.float32)
    Cp, bp = ro.warm_start(win)
    with torch.no_grad():
        m.decoder.decode.weight.copy_(torch.as_tensor(Cp))
        m.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1)))
    ro.nu = np.abs(rng.standard_normal(8))       # non-trivial EMA entry state (positive to avoid log NaN)
    nu_before = ro.nu.copy()
    trial = rng.poisson(0.3, size=(10, 8)).astype(np.float32)
    lam1 = leave_one_neuron_rates(m, ro, trial)
    assert lam1.shape == (10, 8)
    assert np.allclose(ro.nu, nu_before)         # restored on exit (no side effect)
    lam2 = leave_one_neuron_rates(m, ro, trial)
    assert np.allclose(lam1, lam2)               # deterministic / order-independent


from experiments.v1_graf.eval import orientation_decode_acc, torus_embedding
def test_decode_and_torus_shapes():
    rng = np.random.default_rng(20260609)
    dirs = np.repeat(np.arange(0, 360, 5.0), 4)
    lat = np.stack([np.cos(np.deg2rad(dirs)), np.sin(np.deg2rad(dirs)),
                    rng.standard_normal(dirs.size) * 0.01], 1).astype(np.float32)
    acc = orientation_decode_acc(lat, dirs, n_splits=4, seed=20260609)
    assert acc > 0.5
    emb, axis = torus_embedding(lat, dirs)
    assert emb.shape[0] == 72 and emb.shape[1] == 3
