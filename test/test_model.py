import torch

from vjf.model import RBFDS, VJF
from vjf.recognition import Recognition
from vjf.distribution import Gaussian


def test_RBFLDS():
    n_rbf = 10
    xdim, udim = 3, 1
    lds = RBFDS(n_rbf, xdim, udim)

    N = 20
    x = torch.randn(N, xdim)

    lds.loss(x, x)
    lds.update(x, x, torch.randn(N, udim))


def test_rbfds_seeded_centers_require_srrls():
    # rbf_centers/rbf_logwidths are honored only by the srrls flow; passing them to
    # any other flow_learner must fail loudly rather than silently ignore them.
    import pytest
    xt = torch.randn(20, 2)
    xs = torch.randn(20, 2)
    centers = torch.zeros(5, 2)
    rls_ds = RBFDS(n_rbf=5, xdim=2, udim=0, flow_learner='rls')
    with pytest.raises(NotImplementedError):
        rls_ds.initialize(xt, xs, rbf_centers=centers)
    srrls_ds = RBFDS(n_rbf=5, xdim=2, udim=0, flow_learner='srrls')
    srrls_ds.initialize(xt, xs, rbf_centers=centers)   # must NOT raise


def test_Recognition():
    ydim = 10
    xdim = 3
    udim = 2
    recog = Recognition(ydim, xdim, udim, [5, 5])
    N = 20
    y = torch.randn(N, ydim)
    x = torch.randn(N, xdim)
    u = torch.randn(N, udim)
    q = Gaussian(x, torch.zeros_like(x))
    mean, logvar = recog(y, q, u)
    assert mean.shape == (N, xdim) and logvar.shape == (N, xdim)


def test_VJF():
    ydim = 10
    xdim = 3
    udim = 1
    n_rbf = 10
    N = 100
    y = torch.randn(N, ydim)
    x = torch.randn(N, xdim)
    u = torch.randn(N, udim)

    model = VJF.make_model(ydim, xdim, udim, n_rbf, hidden_sizes=[5, 5])
    model.fit(y, u, max_iter=1)
    model.forecast(x[0, ...], u, n_step=N)


def test_spikes_encoder_backward_compat():
    ydim, xdim, n_rbf = 12, 2, 8
    m = VJF.make_model(ydim, xdim, 0, n_rbf, hidden_sizes=[8, 8])  # default encoder='spikes'
    assert m.recognition.mlp[0].in_features == ydim + 2 * xdim
    q, loss = m.filter(torch.randint(0, 2, (ydim,)).float(), None, None)
    assert q.mean.shape[-1] == xdim and torch.isfinite(loss.detach())


def test_projection_encoder_requires_y_enc():
    # encoder='projection' must fail early (clear error), not crash with a shape mismatch,
    # when the recognition feature is not supplied (e.g. via VJF.fit / raw-y path).
    import pytest
    ydim, xdim, n_rbf = 12, 2, 8
    m = VJF.make_model(ydim, xdim, 0, n_rbf, hidden_sizes=[8, 8], encoder='projection')
    y = torch.randint(0, 2, (5, ydim)).float()
    with pytest.raises(ValueError, match="projection"):
        m.fit(y, max_iter=1)
    with pytest.raises(ValueError, match="projection"):
        m.filter(y[0], None, None)  # no y_enc


def test_projection_encoder():
    # recognition reads an xdim-dim feature via y_enc; decoder/likelihood read y.
    ydim, xdim, n_rbf = 12, 2, 8
    m = VJF.make_model(ydim, xdim, 0, n_rbf, hidden_sizes=[8, 8], encoder='projection')
    assert m.recognition.mlp[0].in_features == 3 * xdim  # xdim + udim(0) + 2*xdim
    y = torch.randint(0, 2, (ydim,)).float()
    y_enc = torch.randn(xdim)
    q, loss = m.filter(y, None, None, y_enc=y_enc)
    assert q.mean.shape[-1] == xdim and torch.isfinite(loss.detach())
