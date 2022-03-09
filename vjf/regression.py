import math
import warnings
from typing import Union, Tuple

import torch
from torch import Tensor, linalg, nn
from torch.nn import Module, Parameter, Linear, functional

from vjf.util import reparametrize

from .functional import rbf


__all__ = ['RBFFeature', 'RFFFeature', 'LinearModel', 'RFFFeature2', 'LinearModel2']


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
    def update(self, x):
        cidx = torch.multinomial(torch.ones(self.ndim), self.ndim)
        center = x[cidx]
        pdist = functional.pdist(center)
        self.center = center
        self.logscale.fill_(torch.log(pdist.max() / math.sqrt(2 * self.ndim)))


class RFFFeature(Module):
    """Random Fourier Features"""
    def __init__(self, in_features: int, out_features: int, ndim: int, logscale: float=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ndim = ndim
        self.logscale = logscale
        self.omega = torch.randn(ndim, in_features) * math.sqrt(2./ndim)
        self.b = 2 * math.pi * torch.rand(ndim)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(functional.linear(x / math.exp(self.logscale), self.omega, self.b))

    @torch.no_grad()
    def update(self, x):
        n_sample = x.shape[0]
        pdist = functional.pdist(x)
        self.logscale = math.log(pdist.mean().item() / math.sqrt(n_sample))


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
    def update(self, x: Tensor, y: Tensor):
        self.feature.update(x)
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


### Distinct features for multi-output
class RFFFeature2(Module):
    """Random Fourier Features"""
    def __init__(self, input_ndim: int, output_ndim: int, ndim: int, logscale: float=0.):
        super().__init__()
        self.in_features = input_ndim
        self.out_features = output_ndim
        self.ndim = ndim
        self.logscale = logscale
        self.omega = torch.randn(output_ndim, input_ndim, ndim) * math.sqrt(2./ndim)  # (O, F, D)
        self.b = 2 * math.pi * torch.rand(output_ndim, 1, ndim)

    def forward(self, x: Tensor) -> Tensor:
        """
        param x: (B, D)
        """
        omega = self.omega  # (O, D, F)
        b = self.b  # (O, 1, F)
        x = x.unsqueeze(0)  # (B, D) => (1, B, D)
        mm = torch.matmul(x / math.exp(self.logscale), omega)  # (1, B, D) (O, D, F) => (O, B, F)
        return torch.cos(mm + b)  # (O, B, F)

    @torch.no_grad()
    def update(self, x):
        n_sample = x.shape[0]
        dist = functional.pdist(x).topk(n_sample, largest=True).values.mean()
        self.logscale = math.log(dist.item() / math.sqrt(2 * n_sample))


def eye(m, n):
    """
    Great a 3D tensor (m, n, n). Each slice is an identity matrix.
    """
    return torch.diag_embed(torch.ones(m, n))


class LinearModel2(Module):
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
        self.w = torch.zeros(n_output, self.feature.ndim, 1)  # posterior mean
        self.P = eye(n_output, self.feature.ndim) * alpha  # posterior precision, it turns out the same for all output dimensions
        self.I = eye(n_output, self.feature.ndim)
    
    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor):
        self.feature.update(x)
        F = self.feature(x)  # (O, B, F)
        Ft = F.transpose(-1, -2)  # (O, B, F) => (O, F, B)
        G = torch.matmul(Ft, F)  # (O, F, B) (O, B, F) => (O, F, F)
        I = self.I  # identity, (O, F, F)
        P = self.alpha * I + self.beta * G  # (O, F, F) (O, F, F) => (O, F, F)
        P = .5 * (P + P.transpose(-2, -1))  # ensure symmetry
        y = y.T.unsqueeze(-1)  # (B, O) => (O, B) => (O, B, 1)
        phiy = torch.matmul(Ft, y)  # (O, F, B) (O, B, 1) => (O, F, 1)
        w = self.beta * linalg.solve(P, phiy)  # (O, F, F) (O, F, 1) => (O, F, 1)
        self.P = P
        self.w = w

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        output_shape = (x.shape[0], self.n_output)
        F = self.feature(x)  # (O, B, F)
        w = self.w  # (O, F, 1)
        m = torch.matmul(F, w).squeeze(-1).T  # predictive mean, (O, B, F) (O, F, 1) => (O, B, 1) => (O, B) => (B, O)
        P = self.P  # (O, F, F)
        Ft = F.transpose(-1, -2)  # (O, B, F) => (O, F, B)
        SFt = linalg.solve(P, Ft)  # (O, F, F) (O, F, B) => (O, F, B)
        s = torch.matmul(F, SFt)  # predictive variance, (O, B, F) (O, F, B) => (O, B, B)
        s = s.diagonal(0, dim1=-2, dim2=-1).t()  # (O, B, B) => (O, B) => (B, O)
        assert m.shape == output_shape
        assert s.shape == output_shape
        return m, s

    def sample(self, x: Tensor) -> Tensor:
        m, s = self.forward(x)
        return reparametrize((m, s))
