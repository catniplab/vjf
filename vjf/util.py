from functools import reduce
from typing import Tuple, Union
import operator

import torch
from torch import Tensor

from .distribution import Gaussian


def reparametrize(q: Tuple[Tensor, Tensor]) -> Tensor:
    mean, logvar = q
    return mean + torch.randn_like(mean) * torch.exp(.5 * logvar)


def symmetric(a: Tensor) -> bool:
    return torch.allclose(a, a.transpose(-1, -2))


def running_var(acc_var, acc_size, new_var, new_size, *, size_cap=1000):
    """Running Variance
    :param acc_var:
    :param acc_size:
    :param new_var:
    :param new_size:
    :param size_cap:
    :return:
    """
    acc_size = min(acc_size, size_cap)
    tot_size = acc_size + new_size
    f1 = acc_size / tot_size
    f2 = new_size / tot_size
    acc_var = f1 * acc_var + f2 * new_var
    # print(acc_var.item(), tot_size)
    return acc_var, tot_size


def nonecat(a: Tensor, u: Tensor):
    """Concatenation allowing None input
    :param a: 1st Tensor
    :param u: 2st Tensor or None
    """
    au = torch.atleast_2d(a)
    if u is not None:
        udim = u.shape[-1]
        if udim > 0:
            u = torch.atleast_2d(u)
            au = torch.cat((au, u), -1)
    return au


def at_least2d(a: Union[Tensor, Gaussian]) -> Union[Tensor, Gaussian]:
    """
    See torch.at_least2d
    :param a: Tensor or Tuple
    :return:
    """
    if isinstance(a, Tensor):
        return torch.atleast_2d(a)
    elif isinstance(a, Gaussian):
        return Gaussian(torch.atleast_2d(a.mean), torch.atleast_2d(a.logvar))
    else:
        raise TypeError(a.__class__)


def flat2d(a):
    if a is None:
        return None
    if a.ndim <= 2:
        return at_least2d(a)
    else:
        shape = a.shape
        if 0 == shape[-1]:
            return a.reshape(prod(shape[:-1]), 0)
        else:
            return a.reshape(-1, a.shape[-1])


def prod(a):
    return reduce(operator.mul, a, 1)
