import torch
from torch import Tensor
from torch.nn import Module, Parameter, functional

from .functional import gaussian_loss


class GaussianLikelihood(Module):
    """
    Gaussian likelihood
    """

    def __init__(self):
        super().__init__()
        self.register_parameter('logvar', Parameter(torch.tensor(.1).log(), requires_grad=False))

    def loss(self, eta: Tensor, target: Tensor) -> Tensor:
        """
        :param eta: pre inverse link
        :param target: observation
        :return:
        """
        return gaussian_loss(target, eta, self.logvar)


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
        nll = functional.poisson_nll_loss(eta, target, log_input=True, reduction='none')
        assert nll.ndim == 2
        return nll.sum(-1).mean()
