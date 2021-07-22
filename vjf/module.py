import math
import torch
from torch import nn, Tensor
from torch.nn import Parameter, Module, functional

from .functional import rbf


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


class RBF(Module):
    """Radial basis functions"""
    def __init__(self, n_dim, n_basis):
        super().__init__()
        self.n_basis = n_basis
        self.register_parameter('centroid', Parameter(torch.rand(n_basis, n_dim) - .5, requires_grad=False))
        self.register_parameter('logwidth', Parameter(torch.zeros(n_basis), requires_grad=False))

    @property
    def n_feature(self):
        return self.n_basis

    def forward(self, x):
        return rbf(x, self.centroid, self.logwidth.exp())


class bLinReg(Module):
    """Bayesian linear regression"""
    def __init__(self, feature: Module, n_output: int, logvar: float):
        super().__init__()
        self.feature = feature
        self.n_output = n_output
        # self.bias = torch.zeros(n_outputs)
        self.w_mean = torch.zeros(self.feature.n_feature, n_output)
        self.w_precision = torch.eye(self.feature.n_feature)
        # It turns out that the precision is independent of the output, and hence only one matrix is needed.

        self.register_parameter('logvar', Parameter(torch.tensor(logvar)))

    def forward(self, x, u=None) -> Tensor:
        if u is not None:
            x = torch.cat((x, u), dim=-1)
        feat = self.feature(x)
        return functional.linear(feat, self.w_mean.t())  # do we need the intercept?

    def update(self, target, x, u=None):
        """
        :param target: (sample, dim)
        :param x: (sample, dim)
        :param u: (sample, dim) or None
        :return:
        """
        precision = torch.exp(-self.logvar)

        if u is not None:
            x = torch.cat((x, u), dim=-1)

        feat = self.feature(x)  # (sample, feature)
        scaled_feat = precision * feat
        G = self.w_precision @ self.w_mean + scaled_feat.t() @ target
        # (feature, feature) (feature, output) + (feature, sample) (sample, output) => (feature, output)
        self.w_precision = self.w_precision + scaled_feat.t() @ feat
        # (feature, feature) + (feature, sample) (sample, feature) => (feature, feature)
        self.w_precision = .5 * (self.w_precision + self.w_precision.t())  # make sure symmetric
        L = torch.linalg.cholesky(self.w_precision)
        self.w_mean = G.cholesky_solve(L)  # (feature, feature) (feature, output) => (feature, output)


def test_RBF():
    n_dim, n_basis = 3, 10
    rbf = RBF(n_dim, n_basis)
    blr = bLinReg(rbf, n_dim, 0.)

    N = 20
    x = torch.randn(N, n_dim)
    y = torch.randn(N, n_dim)
    blr(x)
    blr.update(y, x)

