"""
Bayesian Linear Regression
"""
import math
import warnings
from typing import Union, Tuple

import torch
from torch import Tensor, linalg, nn
from torch.nn import functional

from .util import reparametrize, eye
from .functional import rbf, gaussian_kl


__all__ = [
    'RBFFeature', 'RFFFeature', 'LinearModel', 'RBFFeature2', 'RFFFeature2', 'LinearModel2', 'LinearModelVI'
]


# Same feature for multi-output
class RBFFeature(nn.Module):
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

    def forward(self, x, eps=1e-8):
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


class RFFFeature(nn.Module):
    """Random Fourier Features"""
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ndim: int,
                 logscale: float = 0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ndim = ndim
        self.logscale = logscale
        self.omega = torch.randn(ndim, in_features) * math.sqrt(2. / ndim)
        self.b = 2 * math.pi * torch.rand(ndim)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(
            functional.linear(x / math.exp(self.logscale), self.omega, self.b))

    @torch.no_grad()
    def update(self, x):
        n_sample = x.shape[0]
        pdist = functional.pdist(x)
        self.logscale = math.log(pdist.mean().item() / math.sqrt(n_sample))


class LinearModel(nn.Module):
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
    def __init__(self,
                 feature: nn.Module,
                 n_output: int,
                 *,
                 bias: bool = False,
                 alpha: float = 1.,
                 beta: float = 1.):
        super().__init__()

        self.add_module('feature', feature)
        self.n_output = n_output

        self.alpha = torch.tensor(alpha)  # prior precision
        self.beta = torch.tensor(beta)  # noise precision
        self.w = torch.zeros(self.feature.ndim, n_output)  # posterior mean
        self.P = torch.eye(
            self.feature.ndim
        ) * alpha  # posterior precision, it turns out the same for all output dimensions

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor):
        self.feature.update(x)
        phi = self.feature(x)  # (batch, feature)
        G = phi.t().mm(
            phi)  # (feature, batch) (batch, feature) => (feature, feature)
        I = torch.eye(self.feature.ndim)  # identity
        P = self.alpha * I + self.beta * G
        P = .5 * (P + P.t())  # ensure symmetry
        phiy = phi.t().mm(
            y)  # (feature, batch) (batch, output) => (feature, output)
        w = self.beta * linalg.solve(P, phiy)  # (feature, output)
        self.P = P
        self.w = w

    def forward(self, x: Tensor, variance: bool=False) -> Tuple[Tensor, Tensor]:
        output_shape = (x.shape[0], self.n_output)
        phi = self.feature(x)  # (batch, feature)
        w = self.w  # (feature, output)
        m = torch.matmul(
            phi, w
        )  # predictive mean, (batch, feature) (feature, output) => (batch, output)
        P = self.P  # (feature, feature)
        Sphi = linalg.solve(
            P,
            phi.t())  # (feature, feature) (feature, batch) => (feature, batch)
        s = torch.matmul(
            phi, Sphi
        )  # predictive variance, (batch, feature) (feature, batch) => (batch, batch)
        s = s.diagonal().unsqueeze(1).expand_as(
            m)  # (batch, batch) => (batch,) => (batch, 1) => (batch, output)
        assert m.shape == output_shape
        assert s.shape == output_shape

        if variance:
            return m, s
        else:
            return m

    def sample(self, x: Tensor) -> Tensor:
        m, s = self.forward(x, True)
        return reparametrize((m, s))


### Different features for multi-output
class RBFFeature2(nn.Module):
    def __init__(self, center: Tensor, logscale: Tensor, output_ndim: int, *, normalized=False) -> None:
        super().__init__()
        self.ndim = center.shape[0]
        self.output_ndim = output_ndim
        self.center = center
        self.logscale = logscale
        self.normalized = normalized

    def forward(self, x, eps=1e-8):
        h = rbf(x, self.center, self.logscale.exp())
        if self.normalized:
            h = h / (h.sum(-1, keepdim=True) + eps)
        h = h.expand(self.output_ndim, -1, -1)
        return h

    @torch.no_grad()
    def update(self, x):
        cidx = torch.multinomial(torch.ones(self.ndim), self.ndim)
        center = x[cidx]
        pdist = functional.pdist(center)
        self.center = center
        self.logscale.fill_(torch.log(pdist.max() / math.sqrt(2 * self.ndim)))


class RFFFeature2(nn.Module):
    """Random Fourier Features"""
    def __init__(self,
                 input_ndim: int,
                 output_ndim: int,
                 ndim: int,
                 logscale: float = 0.):
        super().__init__()
        self.in_features = input_ndim
        self.out_features = output_ndim
        self.ndim = ndim
        self.logscale = logscale
        self.omega = torch.randn(output_ndim, input_ndim, ndim) * math.sqrt(
            2. / ndim)  # (O, F, D)
        self.b = 2 * math.pi * torch.rand(output_ndim, 1, ndim)
        # omega = torch.randn(1, input_ndim, ndim) * math.sqrt(
        #     2. / ndim)  # (O, F, D)
        # b = 2 * math.pi * torch.rand(1, 1, ndim)
        # self.omega = omega.expand(output_ndim, -1, -1)
        # self.b = b.expand(output_ndim, -1, -1)

    def forward(self, x: Tensor) -> Tensor:
        """
        param x: (B, D)
        """
        omega = self.omega  # (O, D, F)
        b = self.b  # (O, 1, F)
        x = x.unsqueeze(0)  # (B, D) => (1, B, D)
        mm = torch.matmul(x / math.exp(self.logscale),
                          omega)  # (1, B, D) (O, D, F) => (O, B, F)
        return torch.cos(mm + b)  # (O, B, F)

    @torch.no_grad()
    def update(self, x):
        n_sample = x.shape[0]
        # dist = functional.pdist(x).topk(n_sample, largest=True).values.mean()
        dist = functional.pdist(x).max()
        self.logscale = math.log(dist.item() / math.sqrt(2 * n_sample))


class LinearModel2(nn.Module):
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
    def __init__(self,
                 feature: nn.Module,
                 n_output: int,
                 *,
                 bias: bool = False,
                 alpha: float = 1.,
                 beta: float = 1.):
        super().__init__()

        self.add_module('feature', feature)
        self.n_output = n_output

        self.alpha = torch.tensor(alpha)  # prior precision
        self.beta = torch.tensor(beta)  # noise precision
        self.w = torch.zeros(n_output, self.feature.ndim, 1)  # posterior mean
        self.P = eye(
            n_output, self.feature.ndim
        ) * alpha  # posterior precision, it turns out the same for all output dimensions
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
        y = y.t().unsqueeze(-1)  # (B, O) => (O, B) => (O, B, 1)
        phiy = torch.matmul(Ft, y)  # (O, F, B) (O, B, 1) => (O, F, 1)
        w = self.beta * linalg.solve(P,
                                     phiy)  # (O, F, F) (O, F, 1) => (O, F, 1)
        self.P = P
        self.w = w

    def forward(self, x: Tensor, variance=False) -> Tuple[Tensor, Tensor]:
        output_shape = (x.shape[0], self.n_output)
        F = self.feature(x)  # (O, B, F)
        w = self.w  # (O, F, 1)
        m = torch.matmul(F, w).squeeze(
            -1
        ).t()  # predictive mean, (O, B, F) (O, F, 1) => (O, B, 1) => (O, B) => (B, O)
        P = self.P  # (O, F, F)
        Ft = F.transpose(-1, -2)  # (O, B, F) => (O, F, B)
        SFt = linalg.solve(P, Ft)  # (O, F, F) (O, F, B) => (O, F, B)
        s = torch.matmul(
            F, SFt)  # predictive variance, (O, B, F) (O, F, B) => (O, B, B)
        s = s.diagonal(0, dim1=-2,
                       dim2=-1).t()  # (O, B, B) => (O, B) => (B, O)
        assert m.shape == output_shape
        assert s.shape == output_shape
        if variance:
            return m, s
        else:
            return m

    def sample(self, x: Tensor) -> Tensor:
        m, s = self.forward(x, True)
        return reparametrize((m, s))


class LinearModelVI(nn.Module):
    def __init__(self, feature, n_outputs):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__()
        
        feat_ndim = feature.ndim
        self.add_module('feature', feature)

        # self.register_parameter('b_mean', nn.Parameter(torch.zeros(input_ndim)))
        # self.register_parameter('b_logvar', nn.Parameter(torch.zeros(input_ndim)))
        # self.register_parameter('b_prior_mean', nn.Parameter(torch.zeros(input_ndim), requires_grad=False))
        # self.register_parameter('b_prior_logvar', nn.Parameter(torch.zeros(input_ndim), requires_grad=False))

        self.register_parameter('w_mean', nn.Parameter(torch.zeros(n_outputs, feat_ndim)))
        self.register_parameter('w_logvar', nn.Parameter(torch.zeros(n_outputs, feat_ndim)))
        self.register_parameter('w_prior_mean', nn.Parameter(torch.zeros(n_outputs, feat_ndim), requires_grad=False))
        self.register_parameter('w_prior_logvar', nn.Parameter(torch.zeros(n_outputs, feat_ndim), requires_grad=False))

        # self.register_parameter('noise_mean', nn.Parameter(torch.tensor(0.)))
        # self.register_parameter('noise_logvar', nn.Parameter(torch.tensor(0.)))
        # self.register_parameter('noise_prior_mean', nn.Parameter(torch.tensor(0.), requires_grad=False))
        # self.register_parameter('noise_prior_logvar', nn.Parameter(torch.tensor(0.), requires_grad=False))

    @torch.no_grad()
    def update(self, x, y):
        self.feature.update(x)

    def forward(self, x: Tensor, sample: bool=True) -> Tensor:
        if sample:
            w = reparametrize((self.w_mean, self.w_logvar))
        else:
            w = self.w_mean
        feat = self.feature(x)
        return functional.linear(feat, w)

    def kl(self):
        kl_w = gaussian_kl((self.w_mean, self.w_logvar), (self.w_prior_mean, self.w_prior_logvar))
        # kl_noise = gaussian_kl((self.noise_mean, self.noise_logvar), (self.noise_prior_mean, self.noise_prior_logvar))
        kl_noise = 0
        return kl_w + kl_noise
