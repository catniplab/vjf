from itertools import zip_longest
from typing import Tuple, Sequence, Union

import torch
from torch import Tensor, nn, linalg
from torch.nn import Module, Linear, Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from .distribution import Gaussian
from .functional import gaussian_entropy as entropy, gaussian_loss
from .likelihood import GaussianLikelihood, PoissonLikelihood
from .module import LinearRegression, RBF
from .recognition import Recognition
from .util import reparametrize, symmetric, running_var, nonecat
from .numerical import positivize


class LinearDecoder(Module):
    def __init__(self, xdim: int, ydim: int):
        super().__init__()
        self.add_module('decode', Linear(xdim, ydim))
        self.XX = torch.zeros(xdim + 1, xdim + 1)
        self.n_sample = 0

    def forward(self, x: Union[Tensor, Gaussian]) -> Union[Tensor, Gaussian]:
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

    # @torch.no_grad()
    # def update(self, x: Tensor, y: Tensor, *, decay=0.5):
    #     w = torch.column_stack((self.decode.bias.data, self.decode.weight.data)).t()
    #     n, xdim = x.shape
    #     x = torch.column_stack((torch.ones(n), x))
    #     xx = x.t().mm(x)
    #     g = self.XX.mm(w * (1 - decay)) + x.t().mm(y)
    #     self.XX += xx
    #     if self.n_sample > xdim:
    #         self.requires_grad_(False)
    #         w = g.cholesky_solve(linalg.cholesky(self.XX))
    #         b, C = w.split([1, xdim])
    #         self.decode.bias.data = b.squeeze()
    #         self.decode.weight.data = C.t()
    #     self.n_sample += n


def detach(q: Gaussian) -> Gaussian:
    mean, logvar = q
    return Gaussian(mean.detach(), logvar.detach())


class VJF(Module):
    def __init__(self, ydim: int, xdim: int, likelihood: Module, transition: Module, recognition: Module):
        """
        Use VJF.make_model
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

        # if isinstance(self.likelihood, GaussianLikelihood):
        #     self.decoder.requires_grad_(False)
        #     # print(self.decoder.decode.weight.shape,
        #     #       self.decoder.decode.bias.shape,
        #     #       )
        #     nn.init.zeros_(self.decoder.decode.weight)
        #     nn.init.zeros_(self.decoder.decode.bias)

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
        pt = self.transition(xs, u, sampling=False)
        # print(torch.linalg.norm(xs - pt).item())

        y = torch.atleast_2d(y)
        qt = self.recognition(y, qs, u)

        # decode
        xt = reparametrize(qt)
        py = self.decoder(xt)  # NOTE: closed-form did not work well

        return xs, pt, qt, xt, py

    def loss(self, y: Tensor, xs: Tensor, pt: Tensor, qt: Gaussian, xt: Tensor, py: Tensor,
             components: bool = False, warm_up: bool = False) -> Union[Tensor, Tuple]:

        # recon
        l_recon = self.likelihood.loss(py, y)
        # dynamics
        l_dynamics = self.transition.loss(pt, qt)
        # entropy
        h = entropy(qt)

        assert torch.isfinite(l_recon), l_recon.item()
        assert torch.isfinite(l_dynamics), l_dynamics.item()
        assert torch.isfinite(h), h.item()

        loss = l_recon - h
        if not warm_up:
            loss = loss + l_dynamics

        if components:
            return loss, -l_recon, -l_dynamics, h
        else:
            return loss

    @torch.no_grad()
    def update(self, y: Tensor, xs: Tensor, u: Tensor, pt: Tensor, qt: Gaussian, xt: Tensor, py: Tensor, *,
               likelhood=True, decoder=True, transition=True, recognition=True, warm_up=False):
        """Learning without gradient
        :param y:
        :param xs:
        :param u:
        :param pt:
        :param qt:
        :param xt:
        :param py:
        :param likelhood:
        :param decoder:
        :param transition:
        :param recognition:
        :param warm_up:
        :return:
        """
        if likelhood:
            self.likelihood.update(py, y)
        if transition:
            self.transition.update(xt, xs, u, warm_up=warm_up)

    def filter(self, y: Tensor, u: Tensor = None, qs: Gaussian = None, *,
               sgd: bool = True, update: bool = True, debug: bool = False, warm_up: bool = False):
        """
        Filter a step or a sequence
        :param y: observation, assumed axis order (time, batch, dim). missing axis will be prepended.
        :param u: control
        :param qs: previos posterior. use prior if None, otherwise detached.
        :param sgd: flag to enable gradient step
        :param update: flag to update DS
        :param debug: verbose output
        :param warm_up: do not learn dynamics if True, default=False
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
        loss, *elbos = self.loss(y, xs, pt, qt, xt, py, components=debug, warm_up=warm_up)
        if sgd:
            self.optimizer.zero_grad()
            loss.backward()  # accumulate grad if not trained
            nn.utils.clip_grad_value_(self.parameters(), 1.)
            self.optimizer.step()
        if update:
            self.update(y, xs, u, pt, qt, xt, py, warm_up=warm_up)  # non-gradient step

        return qt, loss, *elbos

    def fit(self, y: Tensor, u: Union[None, Tensor] = None, *,
            max_iter: int = 1, beta: float = 0.1, debug: bool = True):
        """
        :param y: observation, (time, ..., dim)
        :param u: control input, None if
        :param max_iter: maximum number of epochs
        :param beta: discounting factor for running loss, large weight on current epoch loss for small value
        :param debug: verbose output
        :return:
            q_seq: list of posterior each step
        """
        y = torch.as_tensor(y, dtype=torch.get_default_dtype())
        y = torch.atleast_2d(y)
        if u is None:
            u_ = [None]
        else:
            u_ = torch.as_tensor(u, dtype=torch.get_default_dtype())
            u_ = torch.atleast_2d(u_)

        warm_up = True
        with trange(max_iter) as progress:
            running_loss = torch.tensor(float('nan'))
            for i in progress:
                # collections
                q_seq = []  # maybe deque is better than list?
                losses = []

                q = None  # use prior
                for yt, ut in zip_longest(y, u_):
                    q, loss, *elbos = self.filter(yt, ut, q,
                                                  sgd=True,
                                                  update=True,
                                                  debug=debug,
                                                  warm_up=warm_up,
                                                  )
                    losses.append(loss)
                    q_seq.append(q)
                    if debug:
                        progress.set_postfix({'Warm up': str(warm_up),
                                              'Loss': running_loss.item(),
                                              'Recon': elbos[0].item(),
                                              'Dynamics': elbos[1].item(),
                                              'Entropy': elbos[2].item(),
                                              'q norm': torch.norm(q[0]).item(),
                                              'obs noise': self.likelihood.logvar.exp().item(),
                                              'state noise': self.transition.logvar.exp().item(),
                                              })

                epoch_loss = sum(losses) / len(losses)
                print(f'epoch loss: {epoch_loss.item():.3f}')

                if warm_up:
                    if epoch_loss.isclose(running_loss, rtol=1e-4):
                        warm_up = False
                        running_loss = epoch_loss
                        print('Warm up stopped.')
                        self.decoder.requires_grad_(False)  # freeze decoder after warm up
                        m = torch.stack([q.mean for q in q_seq]).squeeze()
                        self.transition.initialize(m[1:], m[:-1], u)
                        progress.reset()
                else:
                    if epoch_loss.isclose(running_loss, rtol=1e-4):
                        print('Converged.')
                        break

                running_loss = beta * running_loss + (1 - beta) * epoch_loss if i > 0 else epoch_loss

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
        model = VJF(ydim, xdim, likelihood, RBFDS(n_rbf, xdim, udim), Recognition(ydim, xdim, udim, hidden_sizes))
        return model


class RBFDS(Module):
    def __init__(self, n_rbf: int, xdim: int, udim: int):
        super().__init__()
        self.add_module('velocity', LinearRegression(RBF(xdim + udim, n_rbf), xdim))
        self.register_parameter('logvar', Parameter(torch.tensor(0.), requires_grad=False))  # state noise
        self.n_sample = 0  # sample counter

    def forward(self, x: Tensor, u: Tensor = None, sampling: bool = True, leak: float = 0.) -> Union[Tensor, Gaussian]:
        xu = nonecat(x, u)
        dx = self.velocity(xu, sampling=sampling)
        if isinstance(dx, Gaussian):
            return Gaussian((1 - leak) * x + dx.mean, dx.logvar)
        else:
            return (1 - leak) * x + dx

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
    def update(self, xt: Tensor, xs: Tensor, ut: Tensor = None, *, warm_up=False):
        """Train regression"""
        xs = torch.atleast_2d(xs)
        xu = nonecat(xs, ut)
        xt = torch.atleast_2d(xt)  # TODO: use qt, add qt.logvar to state noise
        dx = xt - xs
        if not warm_up:
            self.velocity.rls(xu, dx, self.logvar.exp(), shrink=1.)  # model dx
            # self.velocity.kalman(xs, dx, self.logvar.exp(), diffusion=.01)  # model dx
        residual = dx - self.velocity(xu, sampling=False).mean
        mse = residual.pow(2).mean()
        var, n_sample = running_var(self.logvar.exp(), self.n_sample, mse, xs.shape[0], size_cap=500)
        self.logvar.data = var.log()
        self.n_sample = n_sample

    @torch.no_grad()
    def initialize(self, xt: Tensor, xs: Tensor, ut: Tensor = None):
        xs = torch.atleast_2d(xs)
        xt = torch.atleast_2d(xt)
        xu = nonecat(xs, ut)
        mse = (xt - xs).pow(2).mean()
        self.velocity.initialize(xu, xt - xs, mse)
        d, V = self.velocity(xu, sampling=False)
        mse = (xt - xs - d).pow(2).mean()
        self.logvar.data = mse.log()

    def loss(self, pt: Tensor, qt: Tensor) -> Tensor:
        return gaussian_loss(pt, qt, self.logvar)
