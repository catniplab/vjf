"""
Generalized Linear Models relevant stuff
"""
from abc import ABCMeta, abstractmethod

import torch


class Family(metaclass=ABCMeta):
    """
    Base class of the likelihoods used in GLM
    The link function
    """

    def __init__(self, link):
        self._link = link

    @abstractmethod
    def loss(self, q, decoder, target):
        pass


class Poisson(Family):
    def __init__(self, link=None):
        if link is None:
            link = torch.exp
        super().__init__(link)

    def loss(self, q, decoder, target):
        mu, logvar = q
        if self._link == torch.exp:
            eta = decoder.decode(mu)
            s = torch.exp(logvar)
            s = torch.unsqueeze(s, 2)
            csq = torch.unsqueeze(self.linear.weight ** 2, 0)
            csc = torch.squeeze(torch.matmul(csq, s), -1)
            loss = torch.mean(
                torch.sum(torch.exp(eta + 0.5 * csc) - eta * target, dim=-1)
            )
            return loss
        else:
            x = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            eta = decoder.decode(x)
            lam = self._link(eta)
            loss = torch.mean(torch.sum(lam - torch.log(lam) * target, dim=-1))
            return loss


class Gaussian(Family):
    def __init__(self, link=None):
        super().__init__(link)

    def loss(self, q, decoder, target):
        mu, logvar = q
        if self._link is None:
            x = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            eta = decoder.decode(x)
            mu = self._link(eta)
            var = torch.exp(self.logvar)
            loss = 0.5 * torch.mean(
                torch.sum((target - mu) ** 2 / var + logvar, dim=-1)
            )
            return loss
        else:
            x = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            eta = decoder.decode(x)
            mu = self._link(eta)
            var = torch.exp(self.logvar)
            loss = 0.5 * torch.mean(
                torch.sum((target - mu) ** 2 / var + logvar, dim=-1)
            )
            return loss
