import torch

from vjf.model import DS, VJF
from vjf.recognition import Recognition


def test_RBFLDS():
    n_rbf = 10
    xdim, udim = 3, 1
    lds = DS(n_rbf, xdim, udim)

    N = 20
    u = torch.randn(N, udim)
    xu = torch.randn(N, xdim + udim)
    x = torch.randn(N, xdim)

    lds.loss(x, x)
    lds.update(x, x, u)


def test_Recognition():
    ydim = 10
    xdim = 3
    recog = Recognition(ydim, xdim, [5, 5])
    N = 20
    y = torch.randn(N, ydim)
    x = torch.randn(N, xdim)
    mean, logvar = recog(y, x)
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
