from collections import namedtuple
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import Module, Linear, Tanh, Sequential, ReLU, Dropout, Parameter
from torch.nn.functional import hardtanh, tanhshrink


__all__ = ['Gaussian', 'DumbRecognition', 'Recognition']

Gaussian = namedtuple('Gaussian', ['mean', 'logvar'])  # 2nd field is logvar or cov


class DumbRecognition(Module):
    def __init__(self, ydim: int, xdim: int, hidden_sizes: Sequence[int]):
        super().__init__()

        # self.add_module('input_y', Linear(ydim, hidden_sizes[0]))
        # self.add_module('input_x', Linear(xdim, hidden_sizes[0], bias=False))

        layers = [Linear(ydim, hidden_sizes[0]), ReLU()]  # input layer
        for k in range(len(hidden_sizes) - 1):
            layers.append(Linear(hidden_sizes[k], hidden_sizes[k + 1]))
            layers.append(ReLU())
        layers.append(Linear(hidden_sizes[-1], xdim * 2))
        self.add_module('mlp', Sequential(*layers))

        # nn.init.zeros_(self.input_x.weight)

    def forward(self, y: Tensor, pt: Tensor) -> Gaussian:
        output = self.mlp(y)
        mean, logvar = output.chunk(2, dim=-1)
        return Gaussian(mean, logvar)


class Recognition(Module):
    def __init__(self, ydim: int, xdim: int, hidden_sizes: Sequence[int], activation=Tanh):
        super().__init__()

        layers = [Linear(ydim + 2 * xdim, hidden_sizes[0]), activation()]  # input layer
        for k in range(len(hidden_sizes) - 1):
            # layers.append(Dropout(p=0.5))
            layers.append(Linear(hidden_sizes[k], hidden_sizes[k + 1]))
            layers.append(activation())

        self.add_module('mlp', Sequential(*layers))
        self.add_module('mean', Linear(hidden_sizes[-1], xdim, bias=False))
        self.add_module('logvar', Linear(hidden_sizes[-1], xdim, bias=True))
        # nn.init.zeros_(self.input_x.weight)

    def forward(self, y: Tensor, pt: Tensor) -> Gaussian:
        if isinstance(pt, Tensor):
            inputs = torch.cat((y, pt), dim=-1)
        else:
            inputs = torch.cat((y, *pt), dim=-1)
        output = self.mlp(inputs)
        # mean, logvar = output.chunk(2, dim=-1)
        mean = self.mean(output)
        logvar = self.logvar(output)
        # mean = torch.tanh(mean)
        # mean = hardtanh(mean)
        # mean = tanhshrink(mean)
        # mean = mean + pt
        return Gaussian(mean, logvar)
