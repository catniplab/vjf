"""
Recognition model
"""
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.nn.functional import batch_norm

from .base import Component
from .distributions import Gaussian, Distribution


class Recognizer(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.xdim = config["xdim"]
        self.ydim = config["ydim"]
        self.udim = config["udim"]
        self.norm = config["batch_norm"]

        # Maintain running statistics
        # See https://www.johndcook.com/blog/skewness_kurtosis/
        self.running_m = torch.zeros(self.xdim)
        # self.register_buffer('running_m', torch.zeros(self.xdim))
        # self.register_buffer('running_s', torch.ones(self.xdim))
        # self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    @abstractmethod
    def recognize(self, y, u, q):
        pass

    @staticmethod
    def get_recognizer(config, system=None):
        if config["recognizer"].lower() == "mlp":
            return MLPRecognizer(config)
        elif config["recognizer"].lower() == "res":
            return ResNetRecognizer(config)
        elif config["recognizer"].lower() == "gru":
            return GRURecognizer(config)
        elif config["recognizer"].lower() == "dyn":
            return DynRecognizer(config, system)
        else:
            raise ValueError("Unknown recognizer")

    def forward(self, y, u, q):
        mu, logvar = self.recognize(y, u, q)

        mu_detached = mu.detach()

        # self.num_batches_tracked += 1

        old_m = self.running_m
        self.running_m = 0.9 * old_m + torch.mean(mu_detached - old_m, dim=0) * 0.1
        # new_m = self.running_m
        # self.running_s = 0.9 * self.running_s + torch.mean((mu_detached - old_m) * (mu_detached - new_m), dim=0) * 0.1

        if self.norm:
            running_mean = self.running_m
            # running_var = self.running_s
            mu = (
                mu - running_mean
            )  # / torch.max(torch.sqrt(running_var), torch.tensor(1e-5))

        return mu, logvar


class MLPRecognizer(Recognizer):
    def __init__(self, config):
        super().__init__(config)
        self.hdim = config["hdim"]
        self.hidden = nn.Linear(
            in_features=self.ydim + self.udim + self.xdim + self.xdim,
            out_features=self.hdim,
        )
        self.mu = nn.Linear(in_features=self.hdim, out_features=self.xdim)
        self.logvar = nn.Linear(in_features=self.hdim, out_features=self.xdim)

    def recognize(self, y, u, q):
        mu0, logvar0 = q
        inputs = torch.cat((y, u, mu0, logvar0), dim=-1)
        h = torch.tanh(self.hidden(inputs))

        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar


class ResNetRecognizer(Recognizer):
    def __init__(self, config):
        super().__init__(config)
        ydim = config["ydim"]
        udim = config["udim"]
        xdim = config["xdim"]
        hdim = config["hdim"]

        self.add_module(
            "i2h", nn.Linear(ydim + udim + xdim + xdim, hdim)
        )  # input layer
        self.add_module("h2r", nn.Linear(hdim, xdim + xdim))  # hidden layer

    def recognize(self, y, u, q):
        mu0, logvar0 = q
        inputs = torch.cat((y, u, mu0, logvar0), dim=-1)
        res = self.h2r(torch.tanh(self.i2h(inputs)))
        res_mu, res_logvar = torch.split(res, 2, dim=-1)
        mu = mu0 + res_mu
        logvar = logvar0 + res_logvar
        return mu, logvar


class GRURecognizer(Recognizer):
    def __init__(self, config):
        super().__init__(config)
        self.hdim = config["hdim"]

        # reset gate
        self.reset_in = nn.Linear(
            in_features=self.ydim + self.udim,
            out_features=self.xdim + self.xdim,
            bias=True,
        )
        self.reset_h = nn.Linear(
            in_features=self.xdim + self.xdim,
            out_features=self.xdim + self.xdim,
            bias=False,
        )
        # update gate
        self.update_in = nn.Linear(
            in_features=self.ydim + self.udim,
            out_features=self.xdim + self.xdim,
            bias=True,
        )
        self.update_h = nn.Linear(
            in_features=self.xdim + self.xdim,
            out_features=self.xdim + self.xdim,
            bias=False,
        )
        # hidden state
        self.hidden_in = nn.Linear(
            in_features=self.ydim + self.udim, out_features=self.hdim, bias=True
        )
        self.hidden_h = nn.Linear(
            in_features=self.xdim + self.xdim, out_features=self.hdim, bias=False
        )
        self.hidden_out = nn.Linear(
            in_features=self.hdim, out_features=self.xdim + self.xdim, bias=True
        )

    def recognize(self, y, u, q):
        h0 = torch.cat(q, dim=-1)
        inputs = torch.cat((y, u), dim=-1)
        z = torch.sigmoid(self.update_in(inputs) + self.update_h(h0))
        r = torch.sigmoid(self.reset_in(inputs) + self.reset_h(h0))
        h = z * h0 + (1 - z) * self.hidden_out(
            torch.tanh(self.hidden_in(inputs) + self.hidden_h(r * h0))
        )
        mu, logvar = torch.split(h, [self.xdim, self.xdim], dim=-1)
        return mu, logvar


class DynRecognizer(Recognizer):
    def __init__(self, config, system):
        super().__init__(config)
        self.system = system
        self.hdim = config["hdim"]
        self.hidden = nn.Linear(
            in_features=self.ydim + self.udim,
            out_features=self.hdim,
        )
        self.mu = nn.Linear(in_features=self.hdim, out_features=self.xdim)
        self.logvar = nn.Linear(in_features=self.hdim, out_features=self.xdim)

    def recognize(self, y, u, q):
        mu0, logvar0 = q
        inputs = torch.cat((y, u), dim=-1)
        h = torch.tanh(self.hidden(inputs))

        mu = self.system.predict(mu0, u) + self.mu(h)
        logvar = logvar0 + self.logvar(h)

        return mu, logvar


class BootstrapRecognizer(Recognizer):
    def __init__(self, config, system):
        super().__init__(config)
        self.system = system

    def recognize(self, y, u, q):
        mu0, logvar0 = q
        inputs = torch.cat((y, u), dim=-1)
        h = torch.tanh(self.hidden(inputs))

        mu = self.system.predict(mu0, u)
        logvar = logvar0 + self.system.noise.logvar

        return mu, logvar


class Encoder(Component, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, *args, **kwargs) -> Distribution:
        pass

    def forward(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    @staticmethod
    def loss(q: Distribution):
        return -q.entropy()


class MLPEncoder(Encoder):
    def __init__(self, ydim, udim, xdim, hdim):
        super().__init__()

        self.ydim = ydim
        self.udim = udim
        self.xdim = xdim
        self.hdim = hdim

        self.hidden = nn.Linear(
            in_features=self.ydim + self.udim + self.xdim + self.xdim,
            out_features=self.hdim,
        )
        self.mu = nn.Linear(in_features=self.hdim, out_features=self.xdim)
        self.logvar = nn.Linear(in_features=self.hdim, out_features=self.xdim)

    def encode(self, y, u, q):
        mu0, logvar0 = q.mean, q.logvar
        h = torch.tanh(self.hidden(torch.cat((y, u, mu0, logvar0), dim=-1)))

        mu = self.mu(h)
        logvar = self.logvar(h)

        return Gaussian(mu, logvar)
