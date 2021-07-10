"""
Dynamical system
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from .base import Noise
from .base import Component
from .gp.covfun import SquaredExponential
from .module import RBFN, IGRU
from .gp import SGP


class GaussianNoise(Noise):
    """Gaussian state noise"""

    def __init__(self, dim, Q, requires_grad=True):
        super().__init__()
        self.dim = dim
        self.register_parameter(
            "logvar", Parameter(torch.zeros(self.dim), requires_grad=requires_grad)
        )

        self.register_parameter(
            "logdecay", Parameter(torch.tensor(2.0), requires_grad=True)
        )
        if Q is not None:
            nn.init.constant_(self.logvar, np.log(Q))

    def loss(self, xhat, q, regularize=False):
        mu, logvar = q
        var = torch.exp(logvar)
        Q = torch.exp(self.logvar)
        loss = 0.5 * torch.mean(
            torch.sum(var / Q + (xhat - mu) ** 2 / Q + self.logvar, dim=-1)
        )
        if regularize:
            loss += torch.exp(self.logdecay) * torch.sum(self.logvar)
        return loss

    def forward(self):
        pass

    @property
    def var(self):
        return torch.exp(self.logvar)

    @property
    def std(self):
        return torch.exp(0.5 * self.logvar)


class System(nn.Module, metaclass=ABCMeta):
    def __init__(self, noise):
        """
        Parameters
        ----------
        noise: stochastic part
        """
        super().__init__()
        self.noise = noise
        self._optimizer = None

    def forward(self, x, u):
        return self.predict(x, u)

    def loss(self, q0, q1, u, sample=True, regularize=False):
        mu0, logvar0 = q0
        if sample:
            x0 = mu0 + torch.exp(0.5 * logvar0) * torch.randn_like(mu0)
        else:
            x0 = mu0
        xhat = self.predict(x0, u)
        return self.noise.loss(xhat, q1, regularize)

    @abstractmethod
    def velocity(self, x, u=None):
        pass

    @abstractmethod
    def predict(self, x, u):
        pass

    @abstractmethod
    def cov(self, q, u):
        pass

    @abstractmethod
    def expect(self, q, u):
        pass

    @abstractmethod
    def update(self, *args):
        """Update parameters without gradients"""
        pass

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters())
        return self._optimizer

    @staticmethod
    def get_system(config, noise):
        system = config["system"].lower()

        if system == "linear":
            return LDS(config["xdim"], config["udim"], config["A"], config["B"], noise)
        elif system == "rbf":
            return RBFS(
                config["xdim"], config["udim"], config["rdim"], config["B"], noise
            )
        elif system == "sgp":
            return SGPS(config["xdim"], config["udim"], noise, config["inducing"])
        elif system == "gru":
            return GRU(config["xdim"], config["udim"], config["gdim"], noise)
        else:
            raise ValueError(f"Unknown system: {system}")


class LDS(System):
    """Linear Dynamical System"""

    def __init__(self, xdim, udim, A, B, noise):
        super().__init__(noise)

        self.transition = nn.Linear(in_features=xdim, out_features=xdim, bias=False)
        self.transition.weight.requires_grad_(A[1])
        if A[0] is not None:
            self.transition.weight.data = torch.from_numpy(A[0]).float()

        self.control = nn.Linear(in_features=udim, out_features=xdim, bias=False)
        self.control.weight.requires_grad_(B[1])
        if B[0] is not None:
            self.control.weight.data = torch.from_numpy(B[0]).float()

        self.add_module("transition", self.transition)
        self.add_module("control", self.control)

    def velocity(self, x, u=None):
        if u is None:
            return self.transition(x)
        else:
            return self.transition(x) + self.control(u)

    def predict(self, x, u):
        return x + self.transition(x) + self.control(u)

    def cov(self, q, u):
        raise NotImplementedError()

    def expect(self, q, u):
        raise NotImplementedError()

    def update(self, *args):
        pass


class RBFS(System):
    """Radial Basis Function Network"""

    def __init__(self, xdim, udim, rdim, B, noise, center=None, logwidth=None):
        super().__init__(noise)

        self.rbfn = RBFN(xdim, rdim, center, logwidth)

        self.transition = nn.Linear(in_features=rdim, out_features=xdim, bias=False)

        self.control = nn.Linear(in_features=udim, out_features=xdim, bias=False)
        self.control.weight.requires_grad_(B[1])
        if B[0] is not None:
            self.control.weight.data = torch.from_numpy(B[0]).float()

        self.add_module("RBFN", self.rbfn)
        self.add_module("transition", self.transition)
        self.add_module("control", self.control)

        self._optimizer = torch.optim.Adam(self.parameters())

    def velocity(self, x, u=None):
        if u is None:
            return self.transition(self.rbfn(x))
        else:
            return self.transition(self.rbfn(x)) + self.control(u)

    def predict(self, x, u):
        return x + self.transition(self.rbfn(x)) + self.control(u)

    def cov(self, q, u):
        raise NotImplementedError()

    def expect(self, q, u):
        raise NotImplementedError()

    def update(self, *args):
        pass


class GRU(System):
    """Radial Basis Function Network"""

    def __init__(self, xdim, udim, hdim, noise):
        super().__init__(noise)

        self.udim = udim
        self.add_module("gru", IGRU(udim, xdim, hdim))

    def velocity(self, x, u=None):
        if u is None:
            u = torch.zeros(x.shape[0], self.udim)
        return self.predict(x, u) - x

    def predict(self, x, u):
        return self.gru(x, u)

    def cov(self, q, u):
        raise NotImplementedError()

    def expect(self, q, u):
        raise NotImplementedError()

    def update(self, *args):
        pass


class SGPS(System):
    """Sparse Gaussian Process"""

    def __init__(self, xdim, udim, noise, inducing):
        super().__init__(noise)

        q = torch.exp(0.5 * noise.logvar)
        self.sgp = SGP(
            in_features=xdim + udim,
            out_features=xdim,
            inducing=inducing,
            mean_func=None,
            cov_func=SquaredExponential(1.0, 1.0, jitter=1e-5),
            noise_var=q,
        )
        self.sgp.initialize()

    def velocity(self, x, u=None):
        return self.sgp.predict(x, sampling=False)

    def predict(self, x, u):
        return x + self.sgp.predict(x, sampling=False)

    def cov(self, q, u):
        raise NotImplementedError()

    def expect(self, q, u):
        raise NotImplementedError()

    def update(self, mu0, mu1):
        self.sgp.fit(mu0, mu1 - mu0)


class Dynamics(Component, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def velocity(self, x, u, *args, **kwargs):
        """dx = f(x, u) dt"""
        pass

    def forward(self, x, u, *args, **kwargs):
        return x + self.velocity(x, u, *args, **kwargs)


class SGPDynamics(Dynamics):
    def __init__(self):
        super().__init__()

    def velocity(self, x, u, *args, **kwargs):
        self.sgp.predict(x, sampling=False)

    @staticmethod
    def loss(q0, p, q1):
        pass
