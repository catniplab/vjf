from abc import ABCMeta, abstractmethod

import torch
from torch import nn


class Decoder(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, x):
        pass


class GLMDecoder(Decoder):
    def __init__(self, family, link, config):
        super().__init__()
        self._family = family
        self._link = link

        xdim = config["xdim"]
        ydim = config["ydim"]

        self.add_module(
            "linear", nn.Linear(in_features=xdim, out_features=ydim, bias=True)
        )

        if config["C"][0] is not None:
            self.linear.weight.data = torch.from_numpy(config["C"][0]).float()
        if config["b"][0] is not None:
            self.linear.bias.data = torch.from_numpy(config["b"][0]).float()

        self.linear.weight.requires_grad = config["C"][1]
        self.linear.bias.requires_grad = config["b"][1]

        # if self.linear.weight.requires_grad and "gnorm" in config:
        #     self.normed_linear = nn.utils.weight_norm(self.linear, dim=config["gnorm"])
        # else:
        self.normed_linear = self.linear

        self.add_module("likelihood", self._family)

    def forward(self, x):
        return self.decode(x)

    def decode(self, x, norm=False):
        if norm:
            return self.normed_linear(x)
        else:
            return self.linear(x)

    def loss(self, q, target, sample=True, norm=True):
        mu, logvar = q
        if sample:
            # reparametrization trick
            x = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            eta = self.decode(x, norm)

            return self._family.loss(target, eta, q, self, sample)
        else:
            s = torch.exp(logvar)
            s = torch.unsqueeze(s, 2)
            csq = torch.unsqueeze(self.linear.weight ** 2, 0)
            csc = torch.squeeze(torch.matmul(csq, s), -1)
            eta = self.decode(mu, False)

            loss = torch.mean(
                torch.sum(torch.exp(eta + 0.5 * csc) - eta * target, dim=-1)
            )
            return loss
