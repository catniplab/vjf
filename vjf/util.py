from typing import Tuple

import torch
from torch import Tensor


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
    au = torch.atleast_2d(a)
    if u is not None:
        u = torch.atleast_2d(u)
        au = torch.cat((au, u), -1)
    return au
