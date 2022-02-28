import math
import warnings
from typing import Union, Tuple

import torch
from torch import Tensor, linalg, nn
from torch.nn import Module, Parameter, Linear, functional

from vjf.util import reparametrize

from .functional import rbf


__all__ = ['RBFFeature', 'LinearModel', 'RFFFeature']


class RBFFeature(Module):
    def __init__(self, center, logscale, *, normalized=False) -> None:
        super().__init__()
        self.center = center
        self.logscale = logscale
        self.normalized = normalized

    @property
    def ndim(self):
        if hasattr(self.center, 'shape'):
            return self.center.shape[0]
        else:
            return 0

    def forward(self, x, eps = 1e-8):
        h = rbf(x, self.center, self.logscale.exp())
        if self.normalized:
            h = h / (h.sum(-1, keepdim=True) + eps)
        return h

    @torch.no_grad()
    def train(self, x):
        cidx = torch.multinomial(torch.ones(self.ndim), self.ndim)
        center = x[cidx]
        pdist = functional.pdist(center)
        self.center = center
        self.logscale.fill_(torch.log(pdist.max() / math.sqrt(self.ndim)))


class RFFFeature(Module):
    """Random Fourier Features"""
    def __init__(self, in_features: int, out_features: int, ndim: int, logscale: float=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ndim = ndim
        self.logscale = logscale
        self.omega = torch.randn(ndim, in_features) * math.sqrt(2./ndim)  # requires_grad should always be False
        self.b = 2 * math.pi * torch.rand(ndim)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(functional.linear(x / math.exp(self.logscale), self.omega, self.b))

    @torch.no_grad()
    def train(self, x):
        pass
        # cidx = torch.multinomial(torch.ones(self.ndim), self.ndim)
        # center = x[cidx]
        # pdist = functional.pdist(center)
        # self.logscale.fill_(torch.log(pdist.max() / math.sqrt(self.ndim)))


class LinearModel(Module):
    """
    phi = phi(x)
    p(y|x, w, beta) = N(y|w'.phi, 1/beta I)
    p(w) = N(w|0, 1/alpha I)
    p(w|y) = N(w|w, S)
    w = beta S Phi'y
    inv(S) = alpha I + beta Phi'Phi
    p(y*|x*, w, S, alpha, beta) = N(y|w'phi*, sigma^2)
    sigma^2 = 1/beta (only for new x) + phi*' S phi*
    """
    def __init__(self, feature: Module, n_output: int, *, bias: bool=False, alpha: float=1., beta: float=1.):
        super().__init__()

        self.add_module('feature', feature)
        self.n_output = n_output

        self.alpha = torch.tensor(alpha)  # prior precision
        self.beta = torch.tensor(beta) # noise precision
        self.w = torch.zeros(self.feature.ndim, n_output)  # posterior mean
        self.P = torch.eye(self.feature.ndim) * alpha  # posterior precision, it turns out the same for all output dimensions
    
    @torch.no_grad()
    def train(self, x: Tensor, y: Tensor):
        self.feature.train(x)
        phi = self.feature(x)  # (batch, feature)
        G = phi.T.mm(phi)  # (feature, batch) (batch, feature) => (feature, feature)
        I = torch.eye(self.feature.ndim) # identity
        P = self.alpha * I + self.beta * G
        P = .5 * (P + P.T)  # ensure symmetry
        phiy = phi.T.mm(y)  # (feature, batch) (batch, output) => (feature, output)
        w = self.beta * linalg.solve(P, phiy)  # (feature, output)
        self.P = P
        self.w = w

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        output_shape = (x.shape[0], self.n_output)
        phi = self.feature(x) # (batch, feature)
        w = self.w # (feature, output)
        m = torch.matmul(phi, w)  # predictive mean, (batch, feature) (feature, output) => (batch, output)
        P = self.P  # (feature, feature)
        Sphi = linalg.solve(P, phi.T)  # (feature, feature) (feature, batch) => (feature, batch)
        s = torch.matmul(phi, Sphi)  # predictive variance, (batch, feature) (feature, batch) => (batch, batch)
        s = s.diagonal().unsqueeze(1).expand_as(m)  # (batch, batch) => (batch,) => (batch, 1) => (batch, output)
        assert m.shape == output_shape
        assert s.shape == output_shape
        return m, s

    def sample(self, x: Tensor) -> Tensor:
        m, s = self.forward(x)
        return reparametrize((m, s))
