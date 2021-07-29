"""
Utility functions to improve numerical stability
"""
import torch
from torch import Tensor


def positivize(a: Tensor, eps: float = 1e-6) -> Tensor:
    assert a.ndim == 2
    # U = a.triu(1)
    # d = a.diagonal().clamp(min=0) + eps
    # a = U.t() + U + torch.diag(d)
    # a = a.triu() + a.triu(1).t()  # symmetrize
    w, v = torch.linalg.eigh(a)
    s = w.clamp(min=eps).sqrt()
    sqrt = v.mm(s.diag_embed())
    a = sqrt.mm(sqrt.t())
    return a


def symmetrize(a: Tensor) -> Tensor:
    # return (a + a.transpose(-1, -2)) / 2
    return a.triu() + a.triu(1).transpose(-1, -2)


def test_symmetrize():
    a = torch.randn(2, 3, 3)
    symmetrize(a)
