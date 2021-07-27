from typing import Union

import torch
from torch import Tensor
from torch.nn import Parameter, Module, functional

from .functional import rbf
from . import kalman
from .util import symmetric


class RBF(Module):
    """Radial basis functions"""
    def __init__(self, n_dim: int, n_basis: int, intercept: bool = False):
        super().__init__()
        self.n_basis = n_basis
        self.intercept = intercept
        self.register_parameter('centroid', Parameter(torch.rand(n_basis, n_dim) - 0.5, requires_grad=False))
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


class LinearRegression(Module):
    """Bayesian linear regression"""
    def __init__(self, feature: Module, n_output: int):
        super().__init__()
        self.add_module('feature', feature)
        self.n_output = n_output
        # self.bias = torch.zeros(n_outputs)
        self.w_mean = torch.zeros(self.feature.n_feature, n_output)
        self.w_cov = torch.eye(self.feature.n_feature)
        self.w_chol = torch.eye(self.feature.n_feature)
        # self.w_precision = torch.eye(self.feature.n_feature)
        self.Q = torch.eye(self.feature.n_feature)  # for Kalman

    def forward(self, x: Tensor, sampling=False) -> Tensor:
        feat = self.feature(x)
        w = self.w_mean
        if sampling:
            # w = w + torch.randn_like(w).cholesky_solve(self.w_cholesky)  # sampling
            w = w + self.w_chol.mm(torch.randn_like(w))  # sampling
        return functional.linear(feat, w.t())  # do we need the intercept?

    # def update(self, x: Tensor, target: Tensor, precision: Union[Tensor, float], jitter: float = 1e-5):
    #     """
    #     :param x: (sample, dim)
    #     :param target: (sample, dim)
    #     :param precision: observation precision
    #     :param jitter: for numerical stability
    #     :return:
    #     """
    #     feat = self.feature(x)  # (sample, feature)
    #     scaled_feat = precision * feat
    #     g = self.w_precision @ self.w_mean + scaled_feat.t() @ target  # what's it called, gain?
    #     # (feature, feature) (feature, output) + (feature, sample) (sample, output) => (feature, output)
    #     self.w_precision = self.w_precision + scaled_feat.t() @ feat
    #     assert torch.allclose(self.w_precision, self.w_precision.t())
    #     # (feature, feature) + (feature, sample) (sample, feature) => (feature, feature)
    #     # self.w_precision = .5 * (self.w_precision + self.w_precision.t())  # make sure symmetric
    #     self.w_cholesky = torch.linalg.cholesky(self.w_precision.double())
    #     self.w_mean = g.double().cholesky_solve(self.w_cholesky).float()
    #     self.w_cholesky = self.w_cholesky.float()
    #     # (feature, feature) (feature, output) => (feature, output)

    @torch.no_grad()
    def update(self, x: Tensor, target: Tensor, v: Union[Tensor, float]):
        A = torch.eye(self.w_mean.shape[0])  # (feature, feature)
        H = self.feature(x)  # (sample, feature)
        R = torch.eye(H.shape[0]) * v  # (feature, feature)
        m = self.w_mean
        S = self.w_cov
        yhat, mhat, Phat = kalman.predict(m, S, A, self.Q, H, R)
        assert symmetric(Phat)
        m, S = kalman.update(target, yhat, mhat, Phat, H, R)
        assert symmetric(S)
        self.w_mean = m
        self.w_cov = S
        self.w_chol = torch.linalg.cholesky(S)
        # U, s, Vh = torch.linalg.svd(S)
        # self.w_chol = U.mm(torch.diag(s.sqrt())).mm(Vh)

    @torch.no_grad()
    def reset(self):
        self.w_mean = torch.zeros_like(self.w_mean)
        self.w_cov = torch.eye(self.feature.n_feature)
        self.w_chol = torch.eye(self.feature.n_feature)
