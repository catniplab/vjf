import numpy as np
import pytest
from experiments.v1_graf.graf_loader import bin_spikes


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
