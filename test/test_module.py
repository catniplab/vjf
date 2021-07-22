import torch

from vjf.module import RBF, bLinReg


def test_RBF():
    n_dim, n_basis = 3, 10
    rbf = RBF(n_dim, n_basis)
    blr = bLinReg(rbf, n_dim)

    N = 20
    x = torch.randn(N, n_dim)
    y = torch.randn(N, n_dim)
    blr(x)
    blr.update(y, x, 1.)
