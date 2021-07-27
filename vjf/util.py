from typing import Tuple

import torch
from torch import Tensor


def reparametrize(q: Tuple[Tensor, Tensor]) -> Tensor:
    mean, logvar = q
    return mean + torch.randn_like(mean) * torch.exp(.5 * logvar)


def symmetric(a: Tensor) -> bool:
    return torch.allclose(a, a.t())
