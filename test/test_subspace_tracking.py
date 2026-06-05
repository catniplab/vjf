import numpy as np
import torch

from vjf.model import VJF
from vjf.readout import OnlineReadout, apply_latent_rotation

N, M, N_RBF = 20, 2, 16


def _velocity(transition, X):
    """f(x) = E[x_{t+1}] - x for a batch X (rows are states)."""
    out = transition(X, None, sampling=False)
    mean = out.mean if isinstance(out, tuple) else out
    return (mean - X).detach().numpy()


def _setup():
    torch.manual_seed(20260605)
    m = VJF.make_model(N, M, 0, N_RBF, hidden_sizes=[8, 8],
                       transition_flow="srrls", encoder="projection")
    tr = m.transition
    tr.velocity.feature.centroid.data = torch.randn(N_RBF, M)
    tr.velocity.feature.logwidth.data = torch.zeros(N_RBF)        # unit width
    tr.velocity.w_mean = torch.randn(tr.velocity.w_mean.shape[0], M)  # nontrivial flow (srrls: plain tensor)
    C = np.random.default_rng(0).standard_normal((N, M)).astype("float32")
    with torch.no_grad():
        m.decoder.decode.weight.copy_(torch.as_tensor(C))
    ro = OnlineReadout(N, M)
    ro.set_fixed(C, np.zeros(N, dtype="float32"))
    return m, tr, ro, C


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype="float32")


def test_imposed_rotation_with_tracking_is_equivariant():
    # C <- C Q rotates the latent to x' = Q^T x; with subspace-tracking the flow must satisfy
    # f_new(Q^T x) = Q^T f_old(x), i.e. (rows) velocity(X @ Q) == f_before @ Q.
    m, tr, ro, C = _setup()
    X = torch.randn(64, M)
    f_before = _velocity(tr, X)
    Q = _rot(0.6)
    ro.impose_rotation(m.decoder, Q, transition=tr, track_subspace=True)
    f_after = _velocity(tr, torch.as_tensor(X.numpy() @ Q))
    assert np.allclose(f_after, f_before @ Q, atol=1e-4)
    # and the readout was actually rotated
    assert np.allclose(m.decoder.decode.weight.detach().numpy(), C @ Q, atol=1e-4)


def test_imposed_rotation_without_tracking_breaks_equivariance():
    m, tr, ro, C = _setup()
    X = torch.randn(64, M)
    f_before = _velocity(tr, X)
    Q = _rot(0.6)
    ro.impose_rotation(m.decoder, Q, transition=tr, track_subspace=False)
    f_after = _velocity(tr, torch.as_tensor(X.numpy() @ Q))
    assert not np.allclose(f_after, f_before @ Q, atol=1e-2)   # flow did NOT follow the frame


def test_maybe_refresh_returns_drift_metrics():
    rng = np.random.default_rng(1)
    counts = rng.poisson(0.3, size=(80, N)).astype("float32")
    m, tr, ro, C = _setup()
    ro2 = OnlineReadout(N, M, refresh_K=10)
    Cp, bp = ro2.warm_start(counts[:40])
    with torch.no_grad():
        m.decoder.decode.weight.copy_(torch.as_tensor(Cp))
        m.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1)))
    for y in counts[40:]:
        ro2.update(ro2.feature(y))
    out = ro2.maybe_refresh(m.decoder, 10, transition=tr, track_subspace=True, oracle_C=C)
    assert out is not None
    for k in ("step", "refresh_rot_deg", "angle_prev_deg", "angle_oracle_deg", "frame_rot_deg", "cond"):
        assert k in out
    assert out["step"] == 10 and np.isfinite(out["angle_oracle_deg"])
    # no refresh off a K-multiple
    assert ro2.maybe_refresh(m.decoder, 13) is None
