from typing import Tuple, Sequence, Union

import torch
from torch import Tensor
from torch.nn import Module, Linear, functional, Parameter, Sequential, ReLU
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from .module import bLinReg, RBF
from .functional import gaussian_entropy as entropy
from .util import complete_shape, reparametrize


class GaussianLikelihood(Module):
    """
    Gaussian likelihood
    """

    def __init__(self):
        super().__init__()
        self.register_parameter('logvar', Parameter(torch.zeros(1)))

    def loss(self, eta: Tensor, target: Tensor) -> Tensor:
        """
        :param eta: pre inverse link
        :param target: observation
        :return:
        """
        dim = target.shape[-1]
        mse = functional.mse_loss(eta, target, reduction='sum')
        v = self.logvar.exp()
        return .5 * (mse / v + dim * self.logvar)


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
        return functional.poisson_nll_loss(eta, target, log_input=True, reduction='sum')


def detach(qs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    mean, logvar = qs
    return mean.detach(), logvar.detach()


class VJF(Module):
    def __init__(self, ydim: int, xdim: int, likelihood: Module, transition: Module, recognition: Module):
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

        self.register_parameter('mean', Parameter(torch.zeros(xdim)))
        self.register_parameter('logvar', Parameter(torch.zeros(xdim)))
        self.q = (self.mean, self.logvar)

        self.optimizer = Adam(self.parameters())
        self.scheduler = ExponentialLR(self.optimizer, 0.95)  # TODO: argument gamma

    def check_q(self, q: Union[Tuple[Tensor, Tensor], None], y: Tensor) -> Tuple[Tensor, Tensor]:
        if q is None:
            q = self.q

        mean, logvar = q
        n_batch = y.shape[1]
        mean = torch.atleast_2d(mean)
        logvar = torch.atleast_2d(logvar)

        if mean.size(0) == 1:
            mean = mean.tile((n_batch, 1))
        if logvar.size(0) == 1:
            logvar = logvar.tile((n_batch, 1))
        assert mean.size(0) == n_batch and logvar.size(0) == n_batch

        return mean, logvar

    def forward(self, y: Tensor, qs: Tuple[Tensor, Tensor], u: Tensor = None) -> Tuple:
        """
        :param y: new observation
        :param qs: posterior before new observation
        :param u: input, None if autonomous
        :return:
            pt: prediction before observation
            qt: posterior after observation
        """
        # encode
        qs = detach(qs)
        xs = reparametrize(qs)
        if u is not None:
            u = torch.atleast_2d(u)
            xs = torch.cat((xs, u), dim=-1)  # TODO: xs is [xs, u] now, find an unambiguous name
        pt = self.transition(xs)

        y = torch.atleast_2d(y)
        qt = self.recognition(y, pt)

        # decode
        xt = reparametrize(qt)
        py = self.decoder(xt)

        return xs, pt, qt, xt, py

    def loss(self, y: Tensor, xs: Tensor, pt: Tensor, qt: Tuple[Tensor, Tensor], xt: Tensor, py: Tensor,
             components: bool = False, full: bool = False) -> Union[Tensor, Tuple]:
        if full:
            raise NotImplementedError
        # recon

        l_recon = self.likelihood.loss(py, y)

        # dynamics
        l_dynamics = self.transition.loss(pt, xt)

        # entropy
        h = entropy(qt)

        loss = l_recon + l_dynamics - h

        if components:
            return loss, -l_recon, -l_dynamics, h
        else:
            return loss

    @torch.no_grad()
    def update(self, y: Tensor, xs: Tensor, pt: Tensor, qt: Tuple[Tensor, Tensor], xt: Tensor, py: Tensor):
        # non gradient
        self.transition.update(y, xs, pt, qt, xt, py)

    def filter(self, y: Tensor, u: Tensor = None, q: Tuple[Tensor, Tensor] = None, update: bool = True) -> Sequence:
        """
        Filter a step or a sequence
        :param y: observation, assumed axis order (time, batch, dim). missing axis will be prepended.
        :param u: control
        :param q: previos posterior
        :param update: flag to learn the parameters
        :return:
            q: posterior
            loss: negative eblo
        """
        y = torch.as_tensor(y, dtype=torch.get_default_dtype())
        y = complete_shape(y)  # (time, batch, dim)
        if u is not None:
            u = torch.as_tensor(u, dtype=torch.get_default_dtype())
            u = complete_shape(u)
        else:
            u = [None] * y.shape[0]

        q = self.check_q(q, y)

        q_seq = []  # maybe deque is better?

        for yt, ut in zip(y, u):
            output = self.forward(yt, q, ut)
            loss = self.loss(yt, *output)
            print(loss.item())
            if update:
                self.optimizer.zero_grad()
            loss.backward()  # accumulate grad if not trained
            if update:
                self.optimizer.step()
                self.update(yt, *output)  # non-gradient step
            q = output[2]  # TODO: ugly
            q_seq.append(q)
        return q_seq

    def fit(self):
        """offline"""
        raise NotImplementedError

    @classmethod
    def make_model(cls, ydim: int, xdim: int, udim: int, n_rbf: int, hidden_sizes: Sequence[int],
                   likelihood: str = 'poisson'):
        if likelihood.lower() == 'poisson':
            likelihood = PoissonLikelihood()
        elif likelihood.lower() == 'gaussian':
            likelihood = GaussianLikelihood()

        model = VJF(ydim, xdim, likelihood, RBFLDS(n_rbf, xdim, udim), Recognition(ydim, xdim, hidden_sizes))
        return model


class RBFLDS(Module):
    def __init__(self, n_rbf: int, xdim: int, udim: int):
        super().__init__()
        self.add_module('linreg', bLinReg(RBF(xdim + udim, n_rbf), xdim))
        self.register_parameter('logvar', Parameter(torch.zeros(1)))

    def forward(self, x: Tensor) -> Tensor:
        return self.linreg(x)

    @torch.no_grad()
    def update(self, y: Tensor, xs: Tensor, pt: Tensor, qt: Tuple[Tensor, Tensor], xt: Tensor, py: Tensor):
        self.linreg.update(xt, xs, torch.exp(-self.logvar))

    def loss(self, pt: Tensor, xt: Tensor) -> Tensor:
        return 0.5 * torch.sum((pt - xt).pow(2) * torch.exp(-self.logvar) + self.logvar)


class Recognition(Module):
    def __init__(self, ydim: int, xdim: int, hidden_sizes: Sequence[int]):
        super().__init__()

        layers = [Linear(ydim + xdim, hidden_sizes[0]), ReLU()]  # input layer
        for k in range(len(hidden_sizes) - 1):
            layers.append(Linear(hidden_sizes[k], hidden_sizes[k + 1]))
            layers.append(ReLU())
        layers.append(Linear(hidden_sizes[-1], xdim * 2))

        self.add_module('mlp', Sequential(*layers))

    def forward(self, y: Tensor, pt: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.mlp(torch.cat((y, pt), dim=-1))
        return output.chunk(2, dim=-1)


# TODO: factory function for VJF
