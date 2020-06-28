from abc import abstractmethod

import numpy as np
import torch

from .operation import squared_scaled_dist


class CovarianceFunction:
    """Covariance function"""

    @abstractmethod
    def __call__(self, a, b=None):
        pass


class SquaredExponential(CovarianceFunction):
    """Squared exponential covariance function"""

    def __init__(self, var, scale, jitter=1e-6):
        self._logvar = torch.tensor(np.log(var))
        self._loggamma = torch.tensor(-np.log(scale))
        self.jitter = jitter

    @property
    def scale(self):
        return torch.exp(-self._loggamma)

    @property
    def var(self):
        return torch.exp(self._logvar)

    @property
    def gamma(self):
        return torch.exp(self._loggamma)

    def __call__(self, a, b=None):
        if b is None:
            sym = True
            b = a
        else:
            sym = False

        d2 = squared_scaled_dist(a, b, gamma=self.gamma)
        cov = self.var * torch.exp(-0.5 * d2)

        if sym:
            cov += torch.eye(a.shape[0]) * self.jitter

        return cov
