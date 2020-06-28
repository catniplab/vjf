from abc import ABCMeta, abstractmethod

import torch

from .metric import gaussian_entropy


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def entropy(self):
        pass

    def join(self, other):
        return Factorized(self, other)


class Factorized(Distribution):
    def __init__(self, *factors):
        self._factors = factors

    def sample(self):
        return [p.sample() for p in self._factors]

    def entropy(self):
        return [p.entropy() for p in self._factors]


class Gaussian(Distribution):
    def __init__(self, mean, logvar):
        self._mean = mean
        self._logvar = logvar

    @property
    def mean(self):
        return self._mean

    @property
    def covariance(self):
        return torch.exp(self._logvar)

    @property
    def logvar(self):
        return self._logvar

    @property
    def cholesky(self):
        return torch.exp(0.5 * self._logvar)

    def sample(self):
        return self.mean + self.cholesky * torch.randn_like(self.mean)

    def entropy(self):
        return gaussian_entropy(self._logvar)
