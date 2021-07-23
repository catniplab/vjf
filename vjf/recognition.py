from collections import namedtuple
from typing import Sequence

from torch import Tensor
from torch.nn import Module, Linear, Tanh, Sequential, ReLU, Dropout

__all__ = ['DiagonalGaussian', 'DumbRecognition', 'Recognition']

DiagonalGaussian = namedtuple('DiagonalGaussian', ['mean', 'logvar'])


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

    def forward(self, y: Tensor, pt: Tensor) -> DiagonalGaussian:
        output = self.mlp(y)
        mean, logvar = output.chunk(2, dim=-1)
        return DiagonalGaussian(mean, logvar)


class Recognition(Module):
    def __init__(self, ydim: int, xdim: int, hidden_sizes: Sequence[int]):
        super().__init__()

        # self.add_module('input_y', Linear(ydim, hidden_sizes[0]))
        # self.add_module('input_x', Linear(xdim, hidden_sizes[0], bias=False))

        layers = [Linear(ydim, hidden_sizes[0]), ReLU()]  # input layer
        for k in range(len(hidden_sizes) - 1):
            layers.append(Dropout(p=0.2))
            layers.append(Linear(hidden_sizes[k], hidden_sizes[k + 1]))
            layers.append(ReLU())
        layers.append(Linear(hidden_sizes[-1], xdim * 2))
        self.add_module('mlp', Sequential(*layers))

        # nn.init.zeros_(self.input_x.weight)

    def forward(self, y: Tensor, pt: Tensor) -> DiagonalGaussian:
        # y = self.input_y(y)
        # x = self.input_x(pt)
        # output = self.mlp(y + x)
        # output = self.mlp(torch.cat((y, pt), dim=-1))
        output = self.mlp(y)
        mean, logvar = output.chunk(2, dim=-1)
        mean = mean + pt
        return DiagonalGaussian(mean, logvar)
