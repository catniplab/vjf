from typing import Tuple

import torch
from torch import Tensor


def complete_shape(a: Tensor) -> Tensor:
    a = torch.atleast_2d(a)
    if a.ndim < 3:
        a = a[None, ...]
    return a


def reparametrize(q: Tuple[Tensor, Tensor]) -> Tensor:
    mean, logvar = q
    return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
