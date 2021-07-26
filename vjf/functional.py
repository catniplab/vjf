from typing import Tuple

import torch
from torch import Tensor, cdist
from torch.nn import functional


def rbf(x: Tensor, c: Tensor, w: Tensor) -> Tensor:
    """
    Radial basis functions
    :param x: input, (batch, dim)
    :param c: centroids, (basis, dim)
    :param w: width (scale), (basis)
    :return:
    """
    d = cdist(x, c)  # ||x - c||, (batch, basis)
    d = d / w  # ||x - c||/w
    return torch.exp(-.5 * d.pow(2))


def gaussian_entropy(q: Tuple[Tensor, Tensor]) -> Tensor:
    _, logvar = q
    assert logvar.ndim == 2
    return 0.5 * logvar.sum(-1).mean()


def gaussian_loss(x, mu, logvar):
    x = torch.atleast_2d(x)
    mu = torch.atleast_2d(mu)

    p = torch.exp(-.5 * logvar)
    mse = functional.mse_loss(x * p, mu * p, reduction='none')
    assert mse.ndim == 2

    nll = .5 * (mse + logvar)
    return nll.sum(-1).mean()
