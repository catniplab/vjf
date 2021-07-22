from typing import Union

import torch
from torch import Tensor
from torch.nn import Parameter, Module, functional

from .functional import rbf


class RBF(Module):
    """Radial basis functions"""
    def __init__(self, n_dim: int, n_basis: int):
        super().__init__()
        self.n_basis = n_basis
        self.register_parameter('centroid', Parameter(torch.rand(n_basis, n_dim) - .5, requires_grad=False))
        self.register_parameter('logwidth', Parameter(torch.zeros(n_basis), requires_grad=False))

    @property
    def n_feature(self):
        return self.n_basis

    def forward(self, x: Tensor) -> Tensor:
        return rbf(x, self.centroid, self.logwidth.exp())


class bLinReg(Module):
    """Bayesian linear regression"""
    def __init__(self, feature: Module, n_output: int):
        super().__init__()
        self.add_module('feature', feature)
        self.n_output = n_output
        # self.bias = torch.zeros(n_outputs)
        self.w_mean = torch.zeros(self.feature.n_feature, n_output)
        self.w_precision = torch.eye(self.feature.n_feature)
        # It turns out that the precision is independent of the output, and hence only one matrix is needed.

    def forward(self, x: Tensor) -> Tensor:
        feat = self.feature(x)
        return functional.linear(feat, self.w_mean.t())  # do we need the intercept?

    def update(self, target: Tensor, x: Tensor, precision: Union[Tensor, float]):
        """
        :param target: (sample, dim)
        :param x: (sample, dim)
        :param precision: observation precision
        :return:
        """
        feat = self.feature(x)  # (sample, feature)
        scaled_feat = precision * feat
        g = self.w_precision @ self.w_mean + scaled_feat.t() @ target  # what's it called, gain?
        # (feature, feature) (feature, output) + (feature, sample) (sample, output) => (feature, output)
        self.w_precision = self.w_precision + scaled_feat.t() @ feat
        # (feature, feature) + (feature, sample) (sample, feature) => (feature, feature)
        self.w_precision = .5 * (self.w_precision + self.w_precision.t())  # make sure symmetric
        L = torch.linalg.cholesky(self.w_precision)
        self.w_mean = g.cholesky_solve(L)  # (feature, feature) (feature, output) => (feature, output)
