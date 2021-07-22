import torch
from torch import Tensor, cdist

# from .module import RBFN


def rbf(x: Tensor, c: Tensor, w: Tensor) -> Tensor:
    """
    Radial basis functions
    :param x: input, (batch, dim)
    :param c: centroids, (basis, dim)
    :param w: width (scale), (basis)
    :return:
    """
    d = cdist(x, c)  # ||x - c||, (batch, basis)
    d /= w  # ||x - c||/w
    return torch.exp(-.5 * d.pow(2))

#
# def test_rbf():
#     N, D, B = 100, 5, 10
#     x = torch.randn(N, D)
#     c = torch.randn(B, D)
#     w = torch.rand(B)
#     r1 = rbf(x, c, w)
#
#     f = RBFN(D, B, c, w[:, None].log())
#     r2 = f(x)
#
#     assert torch.allclose(r1, r2)


def gaussian_entropy(logvar):
    return 0.5 * torch.mean(torch.sum(logvar, dim=-1))