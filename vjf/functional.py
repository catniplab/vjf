from typing import Tuple

import torch
from torch import Tensor, cdist


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


def gaussian_entropy(q: Tuple[Tensor, Tensor]) -> Tensor:
    _, logvar = q
    return 0.5 * torch.mean(torch.sum(logvar, dim=-1))
