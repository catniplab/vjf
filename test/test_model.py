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


def test_rbfds_seeded_centers_applied():
    # Data-driven rbf_centers are honored by the srrls AND sgd flows: they are written
    # into the RBF centroids (growth then extends the basis from there).
    xt = torch.randn(20, 2)
    xs = torch.randn(20, 2)
    centers = torch.zeros(5, 2)
    for flow in ('srrls', 'sgd'):
        ds = RBFDS(n_rbf=5, xdim=2, udim=0, flow_learner=flow)
        ds.initialize(xt, xs, rbf_centers=centers)            # must NOT raise
        assert torch.allclose(ds.velocity.feature.centroid.data, centers)


def test_grow_basis_sgd_keeps_parameter():
    # grow_basis on the sgd flow appends a center and keeps w_mean a trainable Parameter
    # (so VJF can re-point the optimizer at it); the zero row leaves predictions unchanged.
    from torch.nn import Parameter
    ds = RBFDS(n_rbf=4, xdim=2, udim=0, flow_learner='sgd')
    nb0 = ds.velocity.feature.n_basis
    assert isinstance(ds.velocity.w_mean, Parameter)
    x = torch.randn(7, 2)
    pred0 = ds.velocity(x, sampling=False)
    ds.velocity.grow_basis(torch.zeros(1, 2), logwidth=0.0)
    assert ds.velocity.feature.n_basis == nb0 + 1
    assert isinstance(ds.velocity.w_mean, Parameter) and ds.velocity.w_mean.shape[0] == nb0 + 1
    assert torch.allclose(ds.velocity(x, sampling=False), pred0, atol=1e-5)  # zero new weight


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
