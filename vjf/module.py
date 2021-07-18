import math
import torch
from torch import nn
from torch.nn import Parameter


class RBFN(nn.Module):
    def __init__(self, xdim, rdim, center=None, logwidth=None):
        super().__init__()

        # centers
        self.register_parameter(
            "c",
            Parameter(torch.empty(rdim, xdim, dtype=torch.float), requires_grad=False),
        )
        if center is not None:
            self.c.data = torch.tensor(center, dtype=torch.float)
        else:
            nn.init.uniform_(self.c, -0.5, 0.5)

        # kernel widths
        self.register_parameter(
            "logwidth",
            Parameter(torch.full((rdim, 1),
                                 fill_value=0.5*math.log(xdim) - 0.5*math.log(rdim),
                                 dtype=torch.float),
                      requires_grad=False),
        )
        if logwidth is not None:
            self.logwidth.data = torch.tensor(logwidth, dtype=torch.float)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (?, xdim) -> (?, 1, xdim)
        c = torch.unsqueeze(self.c, dim=0)  # (1, rdim, xdim)
        iw = torch.unsqueeze(torch.exp(-self.logwidth), dim=0)  # (1, rdim, 1)
        # (?, 1, xdim) - (1, rdim, xdim) -> (?, rdim, xdim) -> (?, rdim)
        return torch.exp(-0.5 * torch.sum((x * iw - c * iw) ** 2, dim=-1))


class SGP(nn.Module):
    def __init__(self, in_features, out_features, inducing_points, mean, cov):
        super().__init__()

    def forward(self, *input):
        pass


class IGRU(nn.Module):
    def __init__(self, in_features, hidden, mapping):
        super().__init__()

        # reset gate
        self.add_module(
            "i2r", nn.Linear(in_features=in_features, out_features=hidden, bias=False)
        )
        self.add_module(
            "h2r", nn.Linear(in_features=hidden, out_features=hidden, bias=True)
        )
        # update gate
        self.add_module(
            "i2u", nn.Linear(in_features=in_features, out_features=hidden, bias=False)
        )
        self.add_module(
            "h2u", nn.Linear(in_features=hidden, out_features=hidden, bias=True)
        )
        # hidden state
        self.add_module(
            "i2g", nn.Linear(in_features=in_features, out_features=mapping, bias=False)
        )
        self.add_module(
            "h2g", nn.Linear(in_features=hidden, out_features=mapping, bias=True)
        )
        self.add_module(
            "g2h", nn.Linear(in_features=mapping, out_features=hidden, bias=False)
        )

    def forward(self, hidden, inputs):
        u = torch.sigmoid(self.i2u(inputs) + self.h2u(hidden))
        r = torch.sigmoid(self.i2r(inputs) + self.h2r(hidden))
        g = torch.tanh(self.i2g(inputs) + self.h2g(r * hidden))
        h = u * hidden + (1 - u) * self.g2h(g)
        return h
