import torch

from vjf.module import RBF, LinearRegression, RBFN


def test_RBF():
    n_dim, n_basis = 3, 10
    rbf = RBF(n_dim, n_basis)
    blr = LinearRegression(rbf, n_dim)

    N = 20
    x = torch.randn(N, n_dim)
    y = torch.randn(N, n_dim)
    blr(x)
    blr.kalman(y, x, 1.)


def test_RBFN():
    n_dim, n_basis = 3, 10
    rbfn = RBFN(n_dim, n_dim, n_basis)

    N = 20
    x = torch.randn(N, n_dim)
    y = torch.randn(N, n_dim)
    
    rbfn(x)
    