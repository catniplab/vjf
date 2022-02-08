import torch
from torch import Tensor
from torch.nn import Module, Parameter, functional

from .functional import gaussian_loss
from .util import running_var


class GaussianLikelihood(Module):
    """
    Gaussian likelihood
    """

    def __init__(self):
        super().__init__()
        self.register_parameter('logvar', Parameter(torch.tensor(.1).log(), requires_grad=True))
        self.n_sample = 0

    def loss(self, eta: Tensor, target: Tensor) -> Tensor:
        """
        :param eta: pre inverse link
        :param target: observation
        :return:
            negative likelihood
        """
        return gaussian_loss(target, eta, self.logvar)

    @torch.no_grad()
    def update(self, eta: Tensor, target: Tensor):
        """
        Update noise
        :param eta: pre inverse link
        :param target: observation
        :return:
        """
        residual = target - eta
        mse = residual.pow(2).mean()
        var, n_sample = running_var(self.logvar.exp(), self.n_sample, mse, eta.shape[0])
        self.logvar.data = var.log()
        self.n_sample = n_sample


class PoissonLikelihood(Module):
    """
    Poisson likelihood
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def loss(eta: Tensor, target: Tensor) -> Tensor:
        """
        :param eta: pre inverse link
        :param target: observation
        :return:
        """
        if not isinstance(eta, Tensor):
            raise NotImplementedError
        nll = functional.poisson_nll_loss(eta.clamp(max=10.), target, log_input=True, reduction='none')
        # assert nll.ndim == 2
        return nll.sum(-1).mean()

    @torch.no_grad()
    def update(self, eta: Tensor, target: Tensor):
        pass
