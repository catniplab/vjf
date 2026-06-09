import math

import numpy as np
import pytest
import torch

from vjf import synthetic as syn
from vjf.distribution import Gaussian
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter, online_filter_trials

YDIM, XDIM, N_RBF = 12, 2, 8


def _counts(T=30, seed=20260605):
    return np.random.default_rng(seed).poisson(0.3, size=(T, YDIM)).astype("float32")


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    return VJF.make_model(YDIM, XDIM, 0, N_RBF, hidden_sizes=[8, 8],
                          transition_flow="srrls", **kw)


def _proj_model(seed=0, refresh_K=10, counts=None):
    """A projection-encoder model with its readout warm-started into the decoder."""
    m = _model(seed, encoder="projection")
    ro = OnlineReadout(YDIM, XDIM, refresh_K=refresh_K)
    Cp, bp = ro.warm_start(counts[:20] if counts is not None else _counts()[:20])
    with torch.no_grad():
        m.decoder.decode.weight.copy_(torch.as_tensor(Cp))
        m.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1)))
    return m, ro


def test_raw_spike_equivalence_and_fields():
    # online_filter is a faithful shell over a manual filter loop (warm throughout),
    # and exposes the ELBO components + pred_mean.
    counts = _counts()
    m1, m2 = _model(0), _model(0)            # identical params (same build seed)

    torch.manual_seed(7)
    q, manual = None, []
    for y in counts:
        q, loss = m1.filter(torch.as_tensor(y), None, q, sgd=True, update=True,
                            verbose=False, warm_up=True)
        manual.append((q.mean.detach().numpy()[0].copy(), float(loss.detach())))

    torch.manual_seed(7)
    res = list(online_filter(m2, counts, warmup_steps=10**9))  # warm whole run

    assert len(res) == len(manual)
    for (mu, ll), r in zip(manual, res):
        assert np.allclose(mu, r.mean, atol=1e-5)
        assert abs(ll - r.loss) < 1e-4
        assert np.isfinite([r.logvar.sum(), r.recon, r.dynamics, r.entropy]).all()
    assert res[0].pred_mean is None and res[1].pred_mean is not None  # prediction needs a prior post.


def test_projection_supplies_valid_y_enc():
    counts = _counts()
    m, ro = _proj_model(counts=counts)
    seen = []
    orig = m.filter
    def spy(y, u=None, qs=None, **kw):
        seen.append(kw.get("y_enc"))
        return orig(y, u, qs, **kw)
    m.filter = spy

    res = list(online_filter(m, counts, readout=ro, warmup_steps=10**9))
    assert all(np.isfinite(r.mean).all() for r in res)
    assert all(e is not None for e in seen)                  # the loop supplies y_enc
    assert seen[0].shape[-1] == XDIM                          # an xdim subspace feature, not raw spikes
    with pytest.raises(ValueError, match="projection"):       # bare call still guards
        m.filter(torch.as_tensor(counts[0]), None, None)


def test_projection_without_readout_raises():
    # a projection model driven without a readout has no y_enc -> the guard fires.
    counts = _counts()
    m = _model(0, encoder="projection")
    with pytest.raises(ValueError, match="projection"):
        list(online_filter(m, counts, warmup_steps=10**9))


def test_fixed_readout_does_not_drift():
    counts = _counts()
    z = syn.limit_cycle(40, seed=1)
    C, b = syn.poisson_readout(z, YDIM, seed=1)
    m = _model(0, encoder="projection")
    ro = OnlineReadout(YDIM, XDIM)
    ro.set_fixed(C, b)
    with torch.no_grad():
        m.decoder.decode.weight.copy_(torch.as_tensor(C))
        m.decoder.decode.bias.copy_(torch.as_tensor(b.reshape(-1)))
    mb0 = ro.mean_b.copy()

    res = list(online_filter(m, counts, readout=ro, adapt_readout=False, warmup_steps=10**9))
    assert np.array_equal(ro.mean_b, mb0)                     # running mean frozen
    assert not any(r.refreshed for r in res)                  # no Procrustes refresh when fixed


def test_refresh_fires_after_t0():
    counts = _counts(T=40)
    m, ro = _proj_model(refresh_K=10, counts=counts)
    res = list(online_filter(m, counts, readout=ro, warmup_steps=5, rbf_width_scale=0.5))
    assert not res[0].refreshed                               # never refresh at t==0
    assert any(r.refreshed for r in res)                      # but it does fire on later K-multiples


def test_warmup_boundary_initializes_once_with_aligned_args():
    counts = _counts(T=20)
    m = _model(0)
    rec = {"n": 0, "xt": None, "xs": None, "lw_post": None}
    orig = m.transition.initialize
    def spy(xt, xs, ut=None):
        rec["n"] += 1
        out = orig(xt, xs, ut)
        rec["xt"], rec["xs"] = xt.detach().clone(), xs.detach().clone()
        rec["lw_post"] = m.transition.velocity.feature.logwidth.detach().clone()
        return out
    m.transition.initialize = spy

    W = 10
    res = list(online_filter(m, counts, warmup_steps=W, rbf_width_scale=0.5))
    assert rec["n"] == 1                                                  # exactly once
    assert [r.warming_up for r in res] == [True] * W + [False] * (len(res) - W)
    means = np.stack([res[i].mean for i in range(W)])                     # buffered warm means
    assert np.allclose(rec["xt"].numpy(), means[1:], atol=1e-5)           # xt = means[1:]
    assert np.allclose(rec["xs"].numpy(), means[:-1], atol=1e-5)          # xs = means[:-1]
    assert torch.allclose(m.transition.velocity.feature.logwidth.detach(),
                          rec["lw_post"] + math.log(0.5), atol=1e-5)      # widths narrowed after init


def test_divergence_exception_path_resets_q():
    counts = _counts(T=4)
    m = _model(0)
    seen_qs, calls = [], {"n": 0}
    orig = m.filter
    def flaky(y, u=None, qs=None, **kw):
        calls["n"] += 1
        seen_qs.append(qs)
        if calls["n"] == 2:
            raise AssertionError("boom")
        return orig(y, u, qs, **kw)
    m.filter = flaky

    res = list(online_filter(m, counts, warmup_steps=10**9))
    assert not res[0].diverged and res[1].diverged and not res[2].diverged
    assert np.allclose(res[1].mean, res[0].mean)            # repeats last good mean
    assert math.isnan(res[1].loss)
    assert seen_qs[2] is None                               # posterior reset after divergence


def test_divergence_magnitude_guard_path():
    counts = _counts(T=4)
    m = _model(0)
    calls = {"n": 0}
    orig = m.filter
    def huge(y, u=None, qs=None, **kw):
        calls["n"] += 1
        out = orig(y, u, qs, **kw)
        if calls["n"] == 2:                                 # return a wildly large posterior mean
            qt = out[0]
            return (Gaussian(torch.full_like(qt.mean, 1e9), qt.logvar),) + tuple(out[1:])
        return out
    m.filter = huge

    res = list(online_filter(m, counts, warmup_steps=10**9, max_abs_state=1e3))
    assert res[1].diverged and not res[2].diverged          # guard trips, then recovers
    assert np.allclose(res[1].mean, res[0].mean)


def test_determinism_across_warmup_boundary():
    counts = _counts(T=30)
    def run():
        m = _model(0)
        torch.manual_seed(123)
        return [r.mean.copy() for r in online_filter(m, counts, warmup_steps=10, rbf_width_scale=0.5)]
    a, b = run(), run()
    assert len(a) == len(b) == len(counts)
    assert all(np.allclose(x, y) for x, y in zip(a, b))


def test_synthetic_calibration_and_reproducibility():
    z = syn.limit_cycle(400, seed=1)
    C, b = syn.poisson_readout(z, 40, mean_rate=0.1, peak_rate=0.5, seed=1)
    rate = syn.rate_at(z, C, b)
    assert abs(rate.mean(0).mean() - 0.1) < 0.02                          # per-neuron mean ~ target
    ratio = np.median(rate.max(0) / rate.mean(0))
    assert abs(ratio - 5.0) < 1.5                                         # median peak/mean ~ 0.5/0.1
    # determinism of data + stream
    z2 = syn.limit_cycle(400, seed=1)
    C2, b2 = syn.poisson_readout(z2, 40, mean_rate=0.1, peak_rate=0.5, seed=1)
    assert np.array_equal(z, z2) and np.array_equal(C, C2) and np.array_equal(b, b2)
    s1 = np.array(list(syn.stream(z, C, b, seed=2)))
    s2 = np.array(list(syn.stream(z, C, b, seed=2)))
    assert np.array_equal(s1, s2) and s1.shape == (400, 40)


def test_trials_reset_posterior_each_trial():
    counts = _counts(T=20)
    trials = [counts[:10], counts[10:]]            # two 10-step trials
    m = _model(encoder="spikes")
    results = list(online_filter_trials(m, trials, warmup_trials=1, readout=None))
    # one result per sample, with trial index and in-trial step
    assert len(results) == 20
    assert [r.trial for r in results[:10]] == [0]*10
    assert [r.trial for r in results[10:]] == [1]*10
    # first sample of each trial starts from the prior: pred_mean is None
    assert results[0].pred_mean is None and results[10].pred_mean is None
    # within a trial, later samples have a one-step prediction
    assert results[5].pred_mean is not None
