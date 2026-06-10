import math
import warnings
from typing import Union

import torch
from torch import Tensor, linalg, nn
from torch.nn import Parameter, Module, functional

from . import kalman
from .functional import rbf
from .distribution import Gaussian


class RBF(Module):
    """Radial basis functions"""
    def __init__(self, n_dim: int, n_basis: int, intercept: bool = False, requires_grad: bool = False):
        super().__init__()
        self.n_basis = n_basis
        self.intercept = intercept
        self.register_parameter('centroid', Parameter(torch.rand(n_basis, n_dim) * 4 - 2., requires_grad=requires_grad))
        self.register_parameter('logwidth', Parameter(torch.zeros(n_basis), requires_grad=requires_grad))

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
    def __init__(self, feature: Module, n_output: int, bayes=True):
        super().__init__()

        self.bayes = bayes
        self.add_module('feature', feature)
        self.n_output = n_output
        # self.bias = torch.zeros(n_outputs)
        w_mean = torch.zeros(self.feature.n_feature, n_output)
        if not bayes:
            self.register_parameter('w_mean', Parameter(w_mean))
        else:
            self.w_mean = w_mean
        # self.w_cov = torch.eye(self.feature.n_feature)
        self.w_chol = torch.eye(self.feature.n_feature)
        self.w_precision = torch.eye(self.feature.n_feature)
        self.w_pchol = linalg.cholesky(self.w_precision)

    def forward(self, x: Tensor, sampling=True) -> Union[Tensor, Gaussian]:
        """
        Predictive distribution or sample given predictor
        :param x: predictor, supposed to be [x, u].
        :param sampling: return a sample if True, default=True
        :return:
            predictive distribution or sample given sampling
        """
        feat = self.feature(x)
        w = self.w_mean
        
        if not self.bayes:
            return functional.linear(feat, w.t())

        if sampling:
            w = w + self.w_chol.mm(torch.randn_like(w))  # sampling
            # w = w + torch.randn_like(w).cholesky_solve(self.w_pchol)
            return functional.linear(feat, w.t())
        else:
            FL = feat.mm(self.w_chol)
            logvar = FL.mm(FL.t()).diagonal().log().tile((w.shape[-1], 1)).t()
            return Gaussian(functional.linear(feat, w.t()), logvar)
    
    @torch.no_grad()
    def rls(self, x: Tensor, target: Tensor, v: Union[Tensor, float], shrink: float = 1., ridge: float = 0.):
        """RLS weight update
        :param x: (sample, dim)
        :param target: (sample, dim)
        :param v: observation noise
        :param shrink: forgetting factor, 1 meaning no forgetfulness. 0.98 ~ 1
        :param ridge: regularization floor combined with forgetting. With shrink<1,
            adds (1-shrink)*ridge*I to the precision each step so unexcited feature
            directions floor at `ridge` instead of decaying to zero (prevents the
            covariance wind-up / blow-up that plain forgetting causes with sparse
            features). 0 (default) reproduces the original update.
        :return:
        """
        # eye = torch.eye(self.w_precision.shape[0])
        P = self.w_precision
        feat = self.feature(x)  # (sample, feature)
        s = torch.as_tensor(v, dtype=feat.dtype, device=feat.device).sqrt()
        scaled_feat = feat / s
        scaled_target = target / s
        # Forgetting on the gain (g <- shrink*g_prev + new info); the ridge floor
        # (1-shrink)*ridge*I is added to P only (not g), so it both keeps P
        # well-conditioned AND lightly shrinks w toward 0 -- preventing both the
        # precision blow-up (shrink=1) and the wind-up that plain forgetting causes.
        g = (P * shrink).mm(self.w_mean) + scaled_feat.t().mm(scaled_target)  # what's it called, gain?
        # (feature, feature) (feature, output) + (feature, sample) (sample, output) => (feature, output)
        P = P * shrink + scaled_feat.t().mm(scaled_feat)
        if ridge > 0. and shrink < 1.:
            P = P + (1. - shrink) * ridge * torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
        # (feature, feature) + (feature, sample) (sample, feature) => (feature, feature)
        try:
            self.w_pchol = linalg.cholesky(P)
            self.w_precision = P
            self.w_mean = g.cholesky_solve(self.w_pchol)
            self.w_chol = linalg.inv(self.w_pchol.t())  # well, this is not lower triangular
            # (feature, feature) (feature, output) => (feature, output)
        except RuntimeError:
            # P is a symmetric precision matrix; eigvalsh returns real eigenvalues ascending.
            # (torch.eig was removed in torch 2.0.)
            smallest_eig = torch.linalg.eigvalsh(P).min()
            scale = P.diagonal().abs().max().clamp_min(1.)
            jitter = 10 * torch.finfo(P.dtype).eps * scale
            shift = (-smallest_eig).clamp_min(0.) + jitter
            eye = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
            P = P + eye * shift
            self.w_pchol = linalg.cholesky(P)
            self.w_precision = P
            self.w_mean = g.cholesky_solve(self.w_pchol)
            self.w_chol = linalg.inv(self.w_pchol.t())  # well, this is not lower triangular
            warnings.warn('RLS precision matrix was not positive definite; added diagonal jitter.')

    @torch.no_grad()
    def kalman(self, x: Tensor, target: Tensor, v: Union[Tensor, float], diffusion: float = 0.):
        """Update weight using Kalman
        w[t] = w[t-1] + Q
        target[t] = f(x[t])'w[t] + v
        f(x) is the features, e.g. RBF
        Q is diffusion
        :param x: model prediction
        :param target: true x
        :param v: noise variance
        :param diffusion: Q = diffusion * I, default=0. (RLS)
        :return:
        """
        assert diffusion >= 0., 'diffusion needs to be non-negative'
        eye = torch.eye(self.w_mean.shape[0])  # identity matrix (feature, feature)

        # Kalman naming:
        # A: transition matrix
        # Q: state noise
        # H: loading matrix
        # R: observation noise
        Q = diffusion * eye
        A = eye  # diffusion
        H = self.feature(x)  # (sample, feature)
        R = torch.eye(H.shape[0]) * v  # (feature, feature)

        yhat, mhat, Vhat = kalman.predict(self.w_mean, self.w_chol, A, Q, H, R)
        # self.w_mean, self.w_chol = kalman.update(target, yhat, mhat, Vhat, H, R)
        self.w_mean, self.w_chol = kalman.joseph_update(target, yhat, mhat, Vhat, H, R)

    @torch.no_grad()
    def initialize(self, x: Tensor, target: Tensor, v):
        r = x.norm(dim=1).max().item()
        nn.init.uniform_(self.feature.centroid, a=-r, b=r)
        nn.init.constant_(self.feature.logwidth, math.log(r))
        if self.bayes:
            self.rls(x, target, v)
        else:
            # SGD flow: least-squares warm-start into the weight Parameter's data
            # (rls would rebind the Parameter to a plain tensor). lstsq returns the
            # min-norm solution, so unexcited RBF weights start at ~0.
            feat = self.feature(x)
            self.w_mean.data.copy_(torch.linalg.lstsq(feat, target).solution)
        # self.kalman(x, target, torch.tensor(.1))

    @torch.no_grad()
    def init_srls(self, x: Tensor, target: Tensor, p0: float = 1.0,
                  centers: Tensor = None, logwidths: Tensor = None):
        """Initialize the square-root RLS state: features, lstsq warm-start of
        w_mean, and the covariance square root w_chol = sqrt(p0)*I (P = w_chol w_chol').

        If ``centers`` (n_basis, n_dim) is given, the RBF centers/widths are set from
        it (data-driven placement); otherwise the original uniform-box init over the
        state radius is used."""
        if centers is not None:
            self.feature.centroid.data.copy_(torch.as_tensor(centers, dtype=self.feature.centroid.dtype))
            if logwidths is not None:
                self.feature.logwidth.data.copy_(torch.as_tensor(logwidths, dtype=self.feature.logwidth.dtype))
        else:
            r = x.norm(dim=1).max().item()
            nn.init.uniform_(self.feature.centroid, a=-r, b=r)
            nn.init.constant_(self.feature.logwidth, math.log(r))
        feat = self.feature(x)
        self.w_mean = torch.linalg.lstsq(feat, target).solution
        n = self.feature.n_feature
        self.w_chol = (p0 ** 0.5) * torch.eye(n, dtype=feat.dtype, device=feat.device)

    @torch.no_grad()
    def grow_basis(self, center: Tensor, logwidth: float, p0: float = 1.0):
        """Append ONE RBF basis function for the square-root-RLS path: a new center
        with a ZERO weight row (so the current velocity prediction is unchanged at
        every point) and a fresh independent prior block ``sqrt(p0)`` appended to the
        covariance square-root factor ``w_chol``. This is the standard way to add a
        parameter to a square-root RLS filter; the new weight is then learned online
        by subsequent ``srls`` updates. Bayes (srrls) path only; no intercept.
        """
        if not self.bayes:
            raise NotImplementedError("grow_basis is only for the bayesian (srrls) path")
        if self.feature.intercept:
            raise NotImplementedError("grow_basis assumes RBF without an intercept column")
        feat = self.feature
        c = torch.atleast_2d(torch.as_tensor(center, dtype=feat.centroid.dtype,
                                             device=feat.centroid.device))
        lw = torch.as_tensor([float(logwidth)], dtype=feat.logwidth.dtype,
                             device=feat.logwidth.device)
        feat.centroid = Parameter(torch.cat([feat.centroid.data, c], 0),
                                  requires_grad=feat.centroid.requires_grad)
        feat.logwidth = Parameter(torch.cat([feat.logwidth.data, lw], 0),
                                  requires_grad=feat.logwidth.requires_grad)
        feat.n_basis += 1
        n = self.w_chol.shape[0]
        z = torch.zeros(1, self.n_output, dtype=self.w_mean.dtype, device=self.w_mean.device)
        self.w_mean = torch.cat([self.w_mean, z], 0)             # zero weight -> prediction unchanged
        S = torch.zeros(n + 1, n + 1, dtype=self.w_chol.dtype, device=self.w_chol.device)
        S[:n, :n] = self.w_chol
        S[n, n] = float(p0) ** 0.5                               # fresh prior for the new weight
        self.w_chol = S
        # keep the precision-form arrays shape-consistent (unused by srls; padded as identity)
        for name in ("w_precision", "w_pchol"):
            old = getattr(self, name)
            M = torch.eye(n + 1, dtype=old.dtype, device=old.device)
            M[:n, :n] = old
            setattr(self, name, M)

    @torch.no_grad()
    def srls(self, x: Tensor, target: Tensor, v: Union[Tensor, float], shrink: float = 1.):
        """Square-root (Potter) recursive least squares update.

        Propagates the covariance Cholesky factor `w_chol` (P = w_chol w_chol')
        directly via rank-1 Potter updates -- guaranteed positive-definite, no
        precision matrix to accumulate/invert, so it stays numerically stable over
        very long streams where the plain `rls` precision form blows up. RLS-speed
        convergence; unexcited directions keep their prior (weights stay at init).
        `v` is the per-sample noise variance; `shrink`<1 is exponential forgetting.
        """
        feat = self.feature(x)  # (sample, feature)
        r = torch.as_tensor(v, dtype=feat.dtype, device=feat.device).reshape(())
        S = self.w_chol
        W = self.w_mean
        inv_sl = shrink ** -0.5
        for i in range(feat.shape[0]):
            phi = feat[i:i + 1]                 # (1, feature)
            if shrink < 1.:
                S = S * inv_sl                  # forgetting inflates the covariance
            f = S.t().mm(phi.t())               # (feature, 1)
            Sf = S.mm(f)                        # (feature, 1) = P phi'
            a = 1.0 / (f.t().mm(f) + r)         # (1, 1)
            K = a * Sf                          # (feature, 1) Kalman gain
            gamma = a / (1.0 + torch.sqrt(a * r))
            S = S - gamma * Sf.mm(f.t())        # rank-1 Potter downdate (stays a valid sqrt)
            e = target[i:i + 1] - phi.mm(W)     # (1, output) innovation
            W = W + K.mm(e)                      # (feature, output)
        self.w_chol = S
        self.w_mean = W


class RBFN(Module):
    """Radial basis function network
    Not Bayesian
    """
    def __init__(self, in_features: int, out_features: int, n_basis: int, bias: bool = True):
        """
        param in_features: dimensionality of input
        param out_features: dimensionality of output
        param n_basis: number of RBFs
        param bias: If set to False, the output layer will not learn an additive bias. Default: True
        """
        super().__init__()
        self.n_basis = n_basis
        self.bias = bias
        self.register_parameter('centroid', Parameter(torch.randn(n_basis, in_features)))
        self.register_parameter('logscale', Parameter(torch.zeros(1, n_basis)))  # singleton dim for broadcast over batches
        self.add_module('basis2output', nn.Linear(in_features=n_basis, out_features=out_features, bias=bias))

    def forward(self, x: Tensor) -> Tensor:
        h = rbf(x, self.centroid, self.logscale.exp())
        return self.basis2output(h)
