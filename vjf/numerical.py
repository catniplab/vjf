"""
Utility functions to improve numerical stability
"""
import torch
from torch import Tensor


def PSD(a: Tensor, eps: float = 1e-5) -> Tensor:
    a = symmetrize(a)
    d1 = a.diagonal(dim1=-2, dim2=-1)
    d2 = d1.clamp(min=eps)
    return a - torch.diag_embed(d1) + torch.diag_embed(d2)


def symmetrize(a: Tensor) -> Tensor:
    return (a + a.transpose(dim0=-2, dim1=-1)) / 2
