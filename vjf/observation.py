from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter


class Likelihood(nn.Module, metaclass=ABCMeta):
    """Abstract class for observational likelihood"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, observation, prediction, q, decoder, sample=True):
        """

        Parameters
        ----------
        observation
        prediction
        q
        decoder
        sample

        Returns
        -------

        """
        pass

    @staticmethod
    def get_likelihood(config):
        dim = config["ydim"]

        if config["likelihood"].lower() == "gaussian":
            print("Gaussian likelihood")
            likelihood = Gaussian(dim, *config["R"])
        elif config["likelihood"].lower() == "poisson":
            print("Poisson likelihood")
            likelihood = Poisson(dim)
        elif config["likelihood"].lower() == "bernoulli":
            print("Bernoulli likelihood")
            likelihood = Bernoulli(dim)
        else:
            raise ValueError(f"Unknown likelihood: {config['likelihood']}")

        return likelihood


class Gaussian(Likelihood):
    def __init__(self, dim, R, requires_grad=True):
        super().__init__()
        self.dim = dim
        self.logvar = Parameter(torch.zeros(dim), requires_grad=requires_grad)
        if R is not None:
            nn.init.constant_(self.logvar, np.log(R))
        self.register_parameter("logvar", self.logvar)

    def loss(self, y, eta, q, decoder, sample=True):
        logvar = self.logvar
        var = torch.exp(logvar)
        loss = 0.5 * torch.mean(torch.sum((y - eta) ** 2 / var + logvar, dim=-1))
        return loss

    def forward(self, eta):
        return eta


class Poisson(Likelihood):
    def __init__(self, dim, activation=torch.exp):
        super().__init__()
        self.dim = dim
        self.activation = activation

    def loss(self, y, eta, q, decoder, sample=True):
        loss = torch.mean(torch.sum(self.activation(eta) - eta * y, dim=-1))
        if not sample:
            # analytical Poisson loss only supports canonical link
            loss = torch.mean(
                torch.sum(torch.exp(eta + 0.5 * decoder.cov(q)) - eta * y, dim=-1)
            )

        return loss

    def forward(self, eta):
        return self.activation(eta)


class Bernoulli(Likelihood):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def loss(self, y, eta, q, decoder, sample=True):
        p = torch.sigmoid(eta)
        loss = -torch.mean(torch.sum(eta * y + (1 - y) * torch.log(1 - p), dim=-1))
        return loss

    def forward(self, eta):
        return torch.sigmoid(eta)
