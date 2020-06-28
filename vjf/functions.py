from abc import ABCMeta, abstractmethod
from copy import copy

import torch

from .distributions import Gaussian


class Integrable(metaclass=ABCMeta):
    """
    A concrete integrable function should implement computation and integration over certain variables
    """

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def compute(*args, **kwargs):
        pass

    def integrate(self, variable, distribution):
        """Integrate the loss function over a random variable

        :param variable: name of the integrand in the function
        :param distribution: distribution of the variable
        :return: the integration
        """

        def compute(*args, **kwargs):
            return self.compute(*args, **{variable: distribution.sample()}, **kwargs)

        integrated = copy(self)
        integrated.compute = compute

        return integrated


class GaussianLoss(Integrable):
    @staticmethod
    def compute(x, mean, logvar):
        return 0.5 * torch.exp(-logvar) * (x - mean) ** 2

    def integrate(self, variable, distribution):
        """Integrate the loss function over a random variable

        :param variable: name of the integrand in the function
        :param distribution: distribution of the variable
        :return: the integration
        """

        if (variable == "mean" or variable == "x") and isinstance(
            distribution, Gaussian
        ):
            # closed form
            def compute(*args, **kwargs):
                return self.compute(
                    *args, **{variable: distribution.mean}, **kwargs
                ) + 0.5 * torch.exp(distribution.logvar - kwargs["logvar"])

            integrated = copy(self)
            integrated.compute = compute
            return integrated
        else:
            # sampling
            return super().integrate(variable, distribution)
