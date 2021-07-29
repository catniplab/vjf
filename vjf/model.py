from itertools import zip_longest
from typing import Tuple, Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import Module, Linear, Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from .functional import gaussian_entropy as entropy, gaussian_loss
from .likelihood import GaussianLikelihood, PoissonLikelihood
from .module import LinearRegression, RBF
from .recognition import Gaussian, Recognition
from .util import reparametrize, symmetric
from .numerical import positivize


class LinearDecoder(Module):
    def __init__(self, xdim: int, ydim: int):
        super().__init__()
        self.add_module('decode', Linear(xdim, ydim))

    def forward(self, x):
        if isinstance(x, Tensor):
            return self.decode(x)
        elif isinstance(x, Gaussian):
            mean, logvar = x
            mean = self.decode(mean)
            C = self.decode.weight
            S = torch.diag_embed(torch.exp(.5 * logvar))  # sqrt of covariance
            CS = C.unsqueeze(0) @ S
            V = CS @ CS.transpose(-1, -2)
            assert symmetric(V)
            v = V.diagonal(dim1=-2, dim2=-1)
            return Gaussian(mean, v.log())
        else:
            raise NotImplementedError


def detach(q: Gaussian) -> Gaussian:
    mean, logvar = q
    return Gaussian(mean.detach(), logvar.detach())


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
        # self.add_module('decoder', Linear(xdim, ydim))
        self.add_module('decoder', LinearDecoder(xdim, ydim))

        self.register_parameter('mean', Parameter(torch.zeros(xdim)))
        self.register_parameter('logvar', Parameter(torch.zeros(xdim)))

        lr = 1e-4
        self.optimizer = SGD(
            [
                {'params': self.likelihood.parameters(), 'lr': lr},
                {'params': self.decoder.parameters(), 'lr': lr},
                {'params': self.transition.parameters(), 'lr': lr},
                {'params': self.recognition.parameters(), 'lr': lr},
            ],
            lr=lr,
        )
        self.scheduler = ExponentialLR(self.optimizer, 0.9)  # TODO: argument gamma

    def prior(self, y: Tensor) -> Gaussian:
        assert y.ndim == 2
        n_batch = y.shape[0]
        xdim = self.mean.shape[-1]

        mean = torch.atleast_2d(self.mean)
        logvar = torch.atleast_2d(self.logvar)

        one = torch.ones(n_batch, xdim)

        mean = one * mean
        logvar = one * logvar

        assert mean.size(0) == n_batch and logvar.size(0) == n_batch

        return Gaussian(mean, logvar)

    def forward(self, y: Tensor, qs: Gaussian, u: Tensor = None) -> Tuple:
        """
        :param y: new observation
        :param qs: posterior before new observation
        :param u: input, None if autonomous
        :return:
            pt: prediction before observation
            qt: posterior after observation
        """
        # encode
        if qs is None:
            qs = self.prior(y)
        else:
            qs = detach(qs)

        xs = reparametrize(qs)
        pt = self.transition(xs, u)
        # print(torch.linalg.norm(xs - pt).item())

        y = torch.atleast_2d(y)
        qt = self.recognition(y, qs)  # TODO: qs

        # decode
        xt = reparametrize(qt)
        py = self.decoder(xt)
        # py = self.decoder(qt.mean)

        return xs, pt, qt, xt, py

    def loss(self, y: Tensor, xs: Tensor, pt: Tensor, qt: Gaussian, xt: Tensor, py: Tensor,
             components: bool = False, full: bool = False) -> Union[Tensor, Tuple]:
        if full:
            raise NotImplementedError

        # recon
        l_recon = self.likelihood.loss(py, y)
        # dynamics
        l_dynamics = self.transition.loss(pt, xt)
        # entropy
        h = entropy(qt)

        assert torch.isfinite(l_recon), l_recon.item()
        assert torch.isfinite(l_dynamics), l_dynamics.item()
        assert torch.isfinite(h), h.item()

        loss = l_recon - h  # + l_dynamics

        # print(-l_recon, -l_dynamics, h)
        # print('loss', loss.item())
        # print('recon', l_recon.item(), py)
        # print('dyn', l_dynamics.item(), pt)
        # print('ent', h.item(), xt)

        if components:
            return loss, -l_recon, -l_dynamics, h
        else:
            return loss

    @torch.no_grad()
    def update(self, y: Tensor, xs: Tensor, pt: Tensor, qt: Gaussian, xt: Tensor, py: Tensor):
        # non gradient
        self.transition.update(xs, xt)

    def filter(self, y: Tensor, u: Tensor = None, qs: Gaussian = None, *,
               sgd: bool = True, update: bool = True, debug=False):
        """
        Filter a step or a sequence
        :param y: observation, assumed axis order (time, batch, dim). missing axis will be prepended.
        :param u: control
        :param qs: previos posterior. use prior if None, otherwise detached.
        :param sgd: flag to enable gradient step
        :param update: flag to update DS
        :param debug:
        :return:
            qt: posterior
            loss: negative eblo
        """
        y = torch.as_tensor(y, dtype=torch.get_default_dtype())
        y = torch.atleast_2d(y)  # (batch, dim)
        if u is not None:
            u = torch.as_tensor(u, dtype=torch.get_default_dtype())
            u = torch.atleast_2d(u)

        xs, pt, qt, xt, py = self.forward(y, qs, u)
        loss, *elbos = self.loss(y, xs, pt, qt, xt, py, components=debug)
        if sgd:
            self.optimizer.zero_grad()
            loss.backward()  # accumulate grad if not trained
            nn.utils.clip_grad_value_(self.parameters(), 1.)
            self.optimizer.step()
        if update:
            self.update(y, xs, pt, qt, xt, py)  # non-gradient step

        return qt, loss, *elbos

    def fit(self, y, u=None, *, max_iter=1, beta=0.1, offline=False, debug=True):
        y = torch.as_tensor(y)
        y = torch.atleast_2d(y)
        if u is None:
            u = [None]
        else:
            u = torch.as_tensor(u)

        with trange(max_iter) as progress:
            update_ds = False
            running_loss = torch.tensor(float('nan'))
            for i in progress:
                # collections
                q_seq = []  # maybe deque is better than list?
                losses = []

                q = None  # use prior
                for yt, ut in zip_longest(y, u):
                    q, loss, *elbos = self.filter(yt, ut, q,
                                                  sgd=True,
                                                  update=False, debug=debug)
                    losses.append(loss)
                    q_seq.append(q)
                    if debug:
                        progress.set_postfix({'Update': str(update_ds),
                                              'Loss': running_loss.item(),
                                              'Recon': elbos[0].item(),
                                              'Dynamics': elbos[1].item(),
                                              'Entropy': elbos[2].item(),
                                              'q norm': torch.norm(q[0]).item(),
                                              'obs noise': self.likelihood.logvar.exp().item(),
                                              })

                total_loss = sum(losses) / len(losses)
                print(f'{update_ds}, {total_loss.item():.4f}, {self.transition.logvar.exp():.4f}')

                if total_loss.isclose(running_loss, rtol=1e-4):
                    print('Converged.')
                    break
                #     if update_ds:
                #         break
                #     else:
                #         update_ds = True
                # else:
                #     if update_ds:
                #         update_ds = False
                #         # self.transition.reset()

                # if offline:
                #     self.optimizer.zero_grad()
                #     total_loss.backward()
                #     self.optimizer.step()

                # progress.set_postfix({'Loss': total_loss.item()})
                running_loss = beta * running_loss + (1 - beta) * total_loss if i > 0 else total_loss

                self.scheduler.step()

        return q_seq

    @classmethod
    def make_model(cls, ydim: int, xdim: int, udim: int, n_rbf: int, hidden_sizes: Sequence[int],
                   likelihood: str = 'poisson'):
        if likelihood.lower() == 'poisson':
            likelihood = PoissonLikelihood()
        elif likelihood.lower() == 'gaussian':
            likelihood = GaussianLikelihood()

        # model = VJF(ydim, xdim, likelihood, RBFDS(n_rbf, xdim, udim), Recognition(ydim, xdim, hidden_sizes))
        model = VJF(ydim, xdim, likelihood, RBFDS(n_rbf, xdim, udim), Recognition(ydim, xdim, hidden_sizes))
        return model


class RBFDS(Module):
    def __init__(self, n_rbf: int, xdim: int, udim: int):
        super().__init__()
        self.add_module('linreg', LinearRegression(RBF(xdim + udim, n_rbf), xdim))
        self.register_parameter('logvar',
                                Parameter(torch.tensor(-5.), requires_grad=False))  # state noise

    def forward(self, x: Tensor, u: Tensor = None, sampling: bool = True, leak: float = 1e-2) -> Tensor:
        if u is None:
            xu = x
        else:
            u = torch.atleast_2d(u)
            xu = torch.cat((x, u), dim=-1)

        return (1 - leak) * x + self.linreg(xu, sampling=sampling)  # model dx
        # return self.linreg(xu, sampling=sampling)  # model f(x)

    def simulate(self, x0: Tensor, step=1, *, noise=False) -> Tensor:
        x0 = torch.as_tensor(x0, dtype=torch.get_default_dtype())
        x0 = torch.atleast_2d(x0)
        x = torch.empty(step + 1, *x0.shape)
        x[0] = x0
        s = torch.exp(.5 * self.logvar)

        for t in range(step):
            x[t + 1] = self.forward(x[t], sampling=True)
            if noise:
                x[t + 1] = x[t + 1] + torch.randn_like(x[t + 1]) * s

        return x

    @torch.no_grad()
    def update(self, xs: Tensor, xt: Tensor):
        self.linreg.update(xs, xt - xs, torch.exp(self.logvar))  # model dx

    @torch.no_grad()
    def reset(self):
        self.linreg.reset()

    def loss(self, pt: Tensor, xt: Tensor) -> Tensor:
        return gaussian_loss(xt, pt, self.logvar)
