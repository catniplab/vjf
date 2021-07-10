"""
Sparse Gaussian Process
"""

import torch
from torch.nn import Module, Parameter

from .base import GP
from .operation import solve, kron


class Posterior(Module):
    def __init__(self, mean, cov):
        super().__init__()

        self.register_parameter("_mean", Parameter(mean, requires_grad=False))
        self.register_parameter("_cov", Parameter(cov, requires_grad=False))

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean.data = value

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, value):
        self._cov.data = value

    def forward(self, *inputs):
        raise NotImplementedError


class SGP(GP):
    """Sparse Gaussian Process
        y = f(x) + e
    """
    def __init__(
        self, in_features, out_features, mean_func, cov_func, noise_var, f_cov, inducing
    ):
        """
        :param in_features: dimension of y
        :param out_features: dimension of x
        :param mean_func: mean function
        :param cov_func: covariance function
        :param noise_var: variance of state noise
        :param f_cov: prior covariance between dimensions of f
        :param inducing: inducing variables, row vectors
        """
        super().__init__(in_features, out_features, mean_func, cov_func, noise_var, f_cov=None)
        self.register_parameter("inducing", Parameter(torch.as_tensor(inducing), requires_grad=False))
        self.initialize()

    def initialize(self):
        mu = self.mean_func(self.inducing).t().reshape(-1, 1) + torch.zeros(
            self.out_features * self.inducing.shape[0],
            1
        )
        Kzz = self.cov_func(self.inducing)
        Kzz = kron(Kzz, self.K)

        self.add_module("qz", Posterior(mu, Kzz))

    def precompute(self, x):
        Kff = self.cov_func(x)
        Kff = kron(Kff, self.K)
        Kfz = self.cov_func(x, self.inducing)
        Kfz = kron(Kfz, self.K)
        Kzz = self.cov_func(self.inducing)
        Kzz = kron(Kzz, self.K)

        A = solve(Kzz, Kfz.t())
        B = Kff - torch.mm(Kfz, A)
        A = A.t()
        C = B + kron(torch.eye(x.shape[0]), self.Q)

        return A, B, C

    def predict(self, x, sampling=False):
        x = torch.as_tensor(x)
        A, B, C = self.precompute(x)
        # Kfz Kzz^{-1} z, AtΓtAt' + Bt
        mu, G = self.qz.mean, self.qz.cov
        fcov = A @ G @ A.t() + B
        fmean = A @ mu

        if sampling:
            L = torch.linalg.cholesky(fcov)
            f = fmean + L @ torch.normal(torch.zeros_like(fmean))
        else:
            f = fmean

        f = torch.reshape(f, (-1, self.out_features))

        return f

    def forward(self, x):
        return self.predict(x)

    def fit(self, x, y):
        with torch.no_grad():
            x = torch.as_tensor(x)
            y = torch.as_tensor(y)  # (n, q)
            y = torch.reshape(y.t(), (-1, 1))  # (qn, 1)

            A, B, C = self.precompute(x)
            mu, G = self.qz.mean, self.qz.cov  # mu_{t-1}, Gamma_{t-1}

            AG = A @ G
            # Inversion by Woodbury identity
            D = C + AG @ A.t()
            G_t = G - AG.t() @ solve(D, AG)  # Gamma_t
            mu_t = G_t @ (solve(G, mu) + A.t() @ solve(C, y))  # mu_t

            self.qz.mean, self.qz.cov = mu_t, G_t

    def change_inducing(self, inducing):
        """Change inducing points
        :param inducing: new pseudo inputs, row vectors
        :return:
        """
        with torch.no_grad():
            inducing = torch.as_tensor(inducing)
            fmean, fcov = self.predict(inducing)

            self.qz.mean, self.qz.cov = fmean, fcov
            self.inducing.data = inducing
