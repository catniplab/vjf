import numpy as np
import torch

from vjf.readout import OnlineReadout, _procrustes
from vjf.model import VJF


def _principal_angle(a, b):
    """Largest principal angle (rad) between the column spaces of a and b."""
    qa = np.linalg.qr(a)[0]
    qb = np.linalg.qr(b)[0]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(s.min(), -1.0, 1.0))


def test_warm_start_shapes():
    n, m = 20, 2
    rng = np.random.default_rng(20260602)
    ro = OnlineReadout(n, m, link='log')
    counts = rng.poisson(0.3, size=(100, n))
    C, b = ro.warm_start(counts)
    assert C.shape == (n, m) and b.shape == (n,)
    assert ro.project(ro.feature(counts[0])).shape == (m,)


def test_incremental_pca_recovers_subspace():
    # identity link: y = C_true x + noise; CCIPCA should track C_true's column space.
    rng = np.random.default_rng(20260602)
    n, m, T = 30, 2, 6000
    C_true = rng.standard_normal((n, m))
    X = rng.standard_normal((T, m))
    Y = X @ C_true.T + 0.1 * rng.standard_normal((T, n))

    ro = OnlineReadout(n, m, link='identity', refresh_K=0)
    ro.warm_start(Y[:200])
    ang0 = _principal_angle(ro.C, C_true)
    for t in range(200, T):
        g = ro.feature(Y[t])
        ro.update(g)
    ang1 = _principal_angle(ro._scaled_C(), C_true)
    assert ang1 < ang0
    assert ang1 < np.deg2rad(10)


def test_procrustes_aligns():
    rng = np.random.default_rng(20260602)
    n, m = 15, 2
    C = rng.standard_normal((n, m))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    C_rot = C @ R
    C_back = _procrustes(C_rot, C)
    assert np.allclose(C_back, C, atol=1e-5)


def test_maybe_refresh_writes_decoder_only_on_K():
    n, m = 12, 2
    rng = np.random.default_rng(20260602)
    ro = OnlineReadout(n, m, link='log', refresh_K=5)
    ro.warm_start(rng.poisson(0.3, size=(50, n)))
    model = VJF.make_model(n, m, 0, 8, hidden_sizes=[8, 8], encoder='projection')
    decoder = model.decoder

    w0 = decoder.decode.weight.detach().clone()
    assert ro.maybe_refresh(decoder, step=3) is None        # no refresh off a K-multiple
    assert torch.equal(decoder.decode.weight, w0)
    # advance the estimator a bit so the refreshed C differs, then refresh on a multiple of K
    for _ in range(20):
        ro.update(ro.feature(rng.poisson(0.3, size=n)))
    metrics = ro.maybe_refresh(decoder, step=10)            # refresh -> drift-metrics dict
    assert isinstance(metrics, dict) and metrics["step"] == 10
    assert decoder.decode.weight.shape == (n, m)
    assert torch.allclose(decoder.decode.bias.detach(),
                          torch.as_tensor(ro.mean_b.astype(np.float32)))
