"""
Utility functions to improve numerical stability
"""
import torch
from torch import Tensor


def positivize(a: Tensor, eps: float = 1e-3) -> Tensor:
    assert a.ndim == 3
    a = symmetrize(a)
    d1 = a.diagonal(dim1=-2, dim2=-1)
    d2 = d1.clamp(min=eps)
    return a - torch.diag_embed(d1) + torch.diag_embed(d2)


def symmetrize(a: Tensor) -> Tensor:
    # return (a + a.transpose(-1, -2)) / 2
    return a.triu() + a.triu(1).transpose(-1, -2)


def test_symmetrize():
    a = torch.randn(2, 3, 3)
    symmetrize(a)
