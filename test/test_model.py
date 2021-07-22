import torch

from vjf.model import RBFLDS, Recognition, VJF


def test_RBFLDS():
    n_rbf = 10
    xdim, udim = 3, 1
    lds = RBFLDS(n_rbf, xdim, udim)

    N = 20
    xu = torch.randn(N, xdim + udim)
    x = torch.randn(N, xdim)

    lds.loss(x, x)
    lds.update(None, xu, None, None, x, None)


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
    udim = 0
    n_rbf = 10

    VJF.make_model(ydim, xdim, udim, n_rbf, hidden_sizes=[5, 5])
