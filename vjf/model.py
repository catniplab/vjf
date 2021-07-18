import math
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import tqdm
import torch
from torch import Tensor, nn
from torch.nn import Module, functional, Linear, ReLU, Sequential, Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Likelihood(ABC):
    @abstractmethod
    def mean(self, *args, **kwargs):
        pass

    @abstractmethod
    def loss(self, *args, **kwargs):
        pass

    @classmethod
    def get_likelihood(cls, likelihood: str):
        if likelihood.lower() == 'poisson':
            return PoissonLikelihood()
        elif likelihood.lower() == 'gaussian':
            return PoissonLikelihood()
        else:
            raise ValueError(f'Unsupported likelihood {likelihood}')


class PoissonLikelihood(Module, Likelihood):
    def __init__(self):
        super().__init__()

    def mean(self, eta):
        return torch.exp(eta)

    def loss(self, prediction, target, *arg, **kwargs) -> Tensor:
        return functional.poisson_nll_loss(prediction, target, *arg, reduction='sum', **kwargs)


class GaussianLikelihood(Module, Likelihood):
    def __init__(self):
        super().__init__()

    def mean(self, eta):
        return eta

    def loss(self, prediction, target, *arg, **kwargs) -> Tensor:
        return functional.mse_loss(prediction, target, *arg, reduction='sum', **kwargs)


class GLMDecoder(Module):
    def __init__(self, xdim, ydim, likelihood):
        super().__init__()
        likelihood = Likelihood.get_likelihood(likelihood)

        self.add_module('likelihood', likelihood)
        self.add_module('linear', Linear(xdim, ydim))

    def forward(self, x):
        return self.linear(x)

    def generate(self, x):
        return self.lik.mean(self.forward(x))

    def loss(self, prediction, target, *arg, **kwargs):
        return self.lik.loss(prediction, target, *arg, **kwargs)


class MLPEncoder(Module):
    def __init__(self, n_inputs: int, n_outputs: int, hidden_sizes: List[int]):
        super().__init__()

        layers = [Linear(n_inputs, hidden_sizes[0]), ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(ReLU())
        layers.append(Linear(hidden_sizes[-1], n_outputs))

        self.add_module('layers', Sequential(*layers))

    def forward(self, x) -> Any:
        return self.layers(x)


class RBFN(Module):
    def __init__(self, n_inputs, n_outputs, n_bases, *, center=None, logwidth=None):
        super().__init__()

        # centers
        if center is None:
            center = torch.empty(n_bases, n_inputs)
        self.register_parameter(
            "c",
            Parameter(center),
        )
        nn.init.normal_(self.c)

        # kernel widths
        if logwidth is None:
            logwidth = torch.zeros(n_bases, n_inputs)
        self.register_parameter(
            "logwidth",
            Parameter(logwidth),
        )

        self.add_module('output', Linear(n_bases, n_outputs, bias=False))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (?, xdim) -> (?, 1, xdim)
        c = torch.unsqueeze(self.c, dim=0)  # (1, rdim, xdim)
        w = torch.unsqueeze(torch.exp(self.logwidth), dim=0)  # (1, rdim, xdim)
        # (?, 1, xdim) - (1, rdim, xdim) -> (?, rdim, xdim) -> (?, rdim)
        r = torch.exp(-0.5 * torch.sum((x / w - c / w) ** 2, dim=-1))
        return self.output(r)


class VJF(Module):
    def __init__(self, ydim, udim, xdim, *, likelihood='poisson', decoder='GLM', encoder='MLP', dynamics='RBFN',
                 **kwargs):
        super(VJF, self).__init__()

        self.ydim = ydim
        self.udim = udim
        self.xdim = xdim

        self.add_module('decoder', GLMDecoder(xdim, ydim, likelihood))
        self.add_module('encoder', MLPEncoder(ydim + udim + xdim, xdim * 2, kwargs['mlp_sizes']))
        self.add_module('dynamics', RBFN(xdim + udim, xdim, kwargs['n_rbfs']))
        self.register_parameter('state_lnv', Parameter(torch.zeros(xdim)))

    def forward(self, y, u, m, lnv) -> Tuple:
        y = torch.atleast_2d(y)
        u = torch.atleast_2d(u)
        m = torch.atleast_2d(m.detach())
        lnv = torch.atleast_2d(lnv.detach())

        xs = reparametrize(m, lnv)
        xt = self.dynamics(torch.cat((xs, u), dim=-1))

        q = self.encoder(torch.cat((y, u, xt), dim=-1))
        m, lnv = torch.chunk(q, chunks=2, dim=-1)
        recon_x = reparametrize(m, lnv)
        recon_y = self.decoder(recon_x)

        return recon_y, recon_x, xt, m, lnv

    def loss(self, y, recon_y, recon_x, xt, m, lnv) -> Tuple:
        s = torch.exp(.5 * self.state_lnv)
        ll = -self.decoder.likelihood.loss(recon_y, y)
        ld = -.5 * (functional.mse_loss(recon_x / s, xt / s, reduction='sum') + self.state_lnv.sum())
        le = entropy(m, lnv)

        return -(ll + ld + le), ll, ld, le


def reparametrize(m, lnv):
    return m + torch.randn_like(m) * torch.exp(.5 * lnv)


def entropy(m, lnv):
    return .5 * (lnv.sum() + math.prod(m.shape))


def fit(vjf, y, u, *, lr=1e-3, max_iter=1000, clip=1., gamma=.95) -> Tuple:
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    u = torch.as_tensor(u, dtype=torch.get_default_dtype())

    optimizer = Adam(vjf.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma)

    T = y.shape[0]
    batch = 1 if y.ndim == 2 else y.shape[1]
    ms = torch.empty(T, batch, vjf.xdim)
    lnvs = torch.empty(T, batch, vjf.xdim)

    m = torch.randn(batch, vjf.xdim)
    lnv = torch.randn(batch, vjf.xdim)

    with tqdm.trange(max_iter) as progress:
        loss = torch.tensor(np.nan)
        for i in progress:
            optimizer.zero_grad()
            new_loss = torch.tensor(0.)
            for yi, ui in zip(y, u):
                recon_y, recon_x, xt, m, lnv = vjf(yi, ui, m, lnv)
                iloss, *_ = vjf.loss(y, recon_y, recon_x, xt, m, lnv)
                new_loss += iloss
            new_loss /= T
            ms[i] = m.detach()
            lnvs[i] = lnv.detach()
            progress.set_postfix({'Loss': new_loss.item()})
            if torch.isclose(loss, new_loss):
                break
            loss = new_loss
            loss.backward()
            nn.utils.clip_grad_value_(vjf.parameters(), clip_value=clip)
            optimizer.step()
            scheduler.step()

    return ms, lnv


def test_vjf():
    N = 10
    ydim = 5
    xdim = 2
    udim = 1
    model = VJF(ydim, udim, xdim, mlp_sizes=[10], n_rbfs=10)

    y = torch.randn(N, ydim)
    u = torch.zeros(N, udim)

    fit(model, y, u)
