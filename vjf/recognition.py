from collections import namedtuple
from typing import Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import Module, Linear, Tanh, Sequential, ReLU, Dropout, Parameter, GRUCell
from torch.nn.functional import hardtanh, tanhshrink

from .distribution import Gaussian
from .util import nonecat


__all__ = ['Recognition']


class Recognition(Module):
    def __init__(self, ydim: int, xdim: int, udim: int, hidden_sizes: Sequence[int], activation=Tanh):
        super().__init__()

        layers = [Linear(ydim + udim + 2 * xdim, hidden_sizes[0]), activation()]  # input layer
        for k in range(len(hidden_sizes) - 1):
            # layers.append(Dropout(p=0.5))
            layers.append(Linear(hidden_sizes[k], hidden_sizes[k + 1]))
            layers.append(activation())

        self.add_module('mlp', Sequential(*layers))
        self.add_module('mean', Linear(hidden_sizes[-1], xdim, bias=False))
        self.add_module('logvar', Linear(hidden_sizes[-1], xdim, bias=True))
        # nn.init.zeros_(self.input_x.weight)

    def forward(self, y: Tensor, xs: Union[Tensor, Gaussian], u: Tensor = None) -> Gaussian:
        yu = nonecat(y, u)
        if isinstance(xs, Tensor):
            inputs = torch.cat((yu, xs), dim=-1)
        elif isinstance(xs, Gaussian):
            inputs = torch.cat((yu, *xs), dim=-1)
        else:
            raise TypeError
        output = self.mlp(inputs)
        mean = self.mean(output)
        logvar = self.logvar(output)
        return Gaussian(mean, logvar)


class GRURecognition(Module):
    def __init__(self, ydim: int, xdim: int, udim: int, hidden_size: int, activation=Tanh):
        super().__init__()
        
        self.add_module('gru', GRUCell(ydim + udim, hidden_size))
        self.add_module('mean', Linear(hidden_size, xdim, bias=True))
        self.add_module('logvar', Linear(hidden_size, xdim, bias=True))
        
        self.register_parameter('h', Parameter(torch.empty(1, hidden_size), requires_grad=True))
        nn.init.zeros_(self.h)

    def forward(self, y: Tensor, h: Tensor, u: Tensor = None) -> Gaussian:
        batch = y.shape[0]
        if h is None:
            h = self.h.tile((batch, 1))
        yu = nonecat(y, u)
        h = self.gru(yu, h)
        mean = self.mean(h)
        logvar = self.logvar(h)
        return Gaussian(mean, logvar), h
