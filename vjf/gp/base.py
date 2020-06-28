from abc import ABCMeta

import torch
from torch.nn import Module

from .covfun import CovarianceFunction


class GP(Module, metaclass=ABCMeta):
    """Base class of Gaussian Process
    y = f(x) + e
    f ~ GP(m, k(x, x'))
    e ~ N(0, sigma^2)
    """
    def __init__(self, in_features, out_features, mean_func, cov_func: CovarianceFunction, noise_var, f_cov):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.mean_func = mean_func
        self.cov_func = cov_func
        self.noise_var = noise_var

        if self.mean_func is None or self.mean_func == 0:
            self.mean_func = lambda x: torch.zeros(x.shape[0], self.out_features)

        self.Q = noise_var * torch.eye(self.out_features)  # noise covariance

        if f_cov is None or f_cov == "I":
            self.K = torch.eye(self.out_features)
        else:
            self.K = f_cov
