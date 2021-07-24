from typing import Union

import torch
from torch import Tensor
from torch.nn import Parameter, Module, functional

from .functional import rbf


class RBF(Module):
    """Radial basis functions"""
    def __init__(self, n_dim: int, n_basis: int, intercept: bool = False):
        super().__init__()
        self.n_basis = n_basis
        self.intercept = intercept
        self.register_parameter('centroid', Parameter(torch.randn(n_basis, n_dim), requires_grad=False))
        self.register_parameter('logwidth', Parameter(torch.zeros(n_basis), requires_grad=False))

    @property
    def n_feature(self):
        if self.intercept:
            return self.n_basis + 1
        else:
            return self.n_basis

    def forward(self, x: Tensor) -> Tensor:
        output = rbf(x, self.centroid, self.logwidth.exp())
        if self.intercept:
            output = torch.column_stack((torch.ones(output.shape[0]), output))
        return output


class bLinReg(Module):
    """Bayesian linear regression"""
    def __init__(self, feature: Module, n_output: int):
        super().__init__()
        self.add_module('feature', feature)
        self.n_output = n_output
        # self.bias = torch.zeros(n_outputs)
        self.w_mean = torch.zeros(self.feature.n_feature, n_output)
        self.w_precision = torch.eye(self.feature.n_feature)
        self.w_cholesky = torch.linalg.cholesky(self.w_precision)
        # It turns out that the precision is independent of the output, and hence only one matrix is needed.

    def forward(self, x: Tensor, sampling=False) -> Tensor:
        feat = self.feature(x)
        w = self.w_mean
        if sampling:
            w = w + torch.randn_like(w).cholesky_solve(self.w_cholesky)  # sampling
        return functional.linear(feat, w.t())  # do we need the intercept?

    def update(self, x: Tensor, target: Tensor, precision: Union[Tensor, float], jitter: float = 1e-5):
        """
        :param x: (sample, dim)
        :param target: (sample, dim)
        :param precision: observation precision
        :param jitter: for numerical stability
        :return:
        """
        feat = self.feature(x)  # (sample, feature)
        scaled_feat = precision * feat
        g = self.w_precision @ self.w_mean + scaled_feat.t() @ target  # what's it called, gain?
        # (feature, feature) (feature, output) + (feature, sample) (sample, output) => (feature, output)
        self.w_precision = self.w_precision + scaled_feat.t() @ feat
        assert torch.allclose(self.w_precision, self.w_precision.t())
        # (feature, feature) + (feature, sample) (sample, feature) => (feature, feature)
        # self.w_precision = .5 * (self.w_precision + self.w_precision.t())  # make sure symmetric
        self.w_cholesky = torch.linalg.cholesky(self.w_precision.double())
        self.w_mean = g.double().cholesky_solve(self.w_cholesky).float()
        self.w_cholesky = self.w_cholesky.float()
        # (feature, feature) (feature, output) => (feature, output)
        # print(self.w_precision.diagonal())


# TODO: Kalman filter for weight estimation
# w(t) = w(t-1) + e
# y(t) = X(t) w(t) + v
