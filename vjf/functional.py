from typing import Union

import torch
from torch import Tensor, cdist
from torch.nn import functional

from .distribution import Gaussian
from .util import at_least2d


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


def gaussian_entropy(q: Gaussian) -> Tensor:
    """Gaussian entropy"""
    _, logvar = q
    assert logvar.ndim == 2
    return 0.5 * logvar.sum(-1).mean()


def gaussian_loss(a: Union[Tensor, Gaussian], b: Union[Tensor, Gaussian], logvar: Tensor) -> Tensor:
    """
    Negative Gaussian log-likelihood
    E_{a,b} [ 0.5 * (1/sigma^2 ||a - b||_2^2 + 2 * log(sigma)) ]
    :param a: Tensor or Tuple
    :param b: Tensor or Tuple
    :param logvar: 2*log(sigma)
    :return:
        (expected) Gaussian loss
    """
    a = at_least2d(a)
    b = at_least2d(b)

    if isinstance(a, Tensor):
        m1, logv1 = a, None
    else:
        m1, logv1 = a

    if isinstance(b, Tensor):
        m2, logv2 = b, None
    else:
        m2, logv2 = b

    p = torch.exp(-.5 * logvar)

    # print(mu, logv)
    mse = functional.mse_loss(m1 * p, m2 * p, reduction='none')
    assert mse.ndim == 2
    assert torch.all(torch.isfinite(mse)), mse

    nll = .5 * (mse + logvar)

    if logv1 is None and logv2 is None:
        trace = 0.
    elif logv2 is None:
        trace = torch.exp(logv1 - logvar)
    elif logv1 is None:
        trace = torch.exp(logv2 - logvar)
    else:
        trace = torch.exp(logv1 + logv2 - logvar)

    nll = nll + .5 * trace

    return nll.sum(-1).mean()
