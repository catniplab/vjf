from typing import Tuple, Callable

import torch
from torch import Tensor
from torch.nn import Module, Linear, functional, Parameter

from .functional import gaussian_entropy


def reparametrize(q):
    mean, logvar = q
    return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)


def entropy(q):
    _, logvar = q
    return gaussian_entropy(logvar)


class GaussianLikelihood(Module):
    """
    Gaussian likelihood
    """
    def __init__(self):
        super().__init__()
        self.register_parameter('logvar', Parameter(torch.zeros(1)))

    def loss(self, eta, target):
        """
        :param eta: pre inverse link
        :param target: observation
        :return:
        """
        dim = target.size(-1)
        mse = functional.mse_loss(eta, target, reduction='sum')
        v = self.logvar.exp()
        return .5 * (mse / v + dim * self.logvar)


class PoissonLikelihood(Module):
    """
    Poisson likelihood
    """
    def __init__(self):
        super().__init__()

    def loss(self, eta, target):
        """
        :param eta: pre inverse link
        :param target: observation
        :return:
        """
        return functional.poisson_nll_loss(eta, target, log_input=True, reduction='sum')


class VJF(Module):
    def __init__(self, ydim, xdim, udim, likelihood: Module, transition: Module, recognition: Module):
        """
        :param likelihood: GLM likelihood, Gaussian or Poisson
        :param transition: f(x[t-1], u[t]) -> x[t]
        :param recognition: y[t], f(x[t-1], u[t]) -> x[t]
        """
        super().__init__()
        self.add_module('likelihood', likelihood)
        self.add_module('transition', transition)
        self.add_module('recognition', recognition)
        self.add_module('decoder', Linear(xdim, ydim))

    def forward(self, y: Tensor, qs: Tuple, u: Tensor = None):
        """
        :param y: new observation
        :param qs: posterior before new observation
        :param u: input, None if autonomous
        :return:
            pt: prediction before observation
            qt: posterior after observation
        """
        xs = reparametrize(qs)
        if u is not None:
            u = torch.atleast_2d(u)
            xs = torch.cat((xs, u), dim=-1)
        pt = self.transition(xs)

        y = torch.atleast_2d(y)
        qt = self.recognition(y, pt)

        return pt, qt

    def loss(self, y, pt, qt, components=False, full=False):
        if full:
            raise NotImplementedError
        # recon
        xt = reparametrize(qt)
        py = self.decoder(xt)
        l_recon = self.likelihood.loss(py, y)

        # dynamics
        l_dynamics = self.transition.loss(pt, xt)

        # entropy
        h = entropy(qt)

        loss = l_recon + l_dynamics - h

        if components:
            return loss, -l_recon, -l_dynamics, h, xt
        else:
            return loss

    def update(self, pt, qt):
        """
        :param pt:
        :param qt:
        :return:
        """
        # non gradient
        self.transition.update(pt, qt)


class LDS(Module):
    def __init__(self):
        super().__init__()
        self.