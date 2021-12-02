from itertools import zip_longest
from typing import Tuple, Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import Module, Linear, Parameter, GRU
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from .distribution import Gaussian
from .functional import gaussian_entropy as entropy, gaussian_loss
from .likelihood import GaussianLikelihood, PoissonLikelihood
from .module import LinearRegression, RBF
from .util import reparametrize, symmetric, running_var, nonecat


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


class GRURecognition(Module):
    def __init__(self, ydim: int, xdim: int, udim: int, hidden_size: int):
        super().__init__()

        self.add_module('gru', GRU(input_size=ydim + udim, hidden_size=hidden_size, batch_first=False, bidirectional=True))
        self.register_parameter('h0', Parameter(torch.zeros(2, 1, hidden_size)))

        self.add_module('hidden_q', Linear(hidden_size * 2, xdim * 2, bias=True))

    def forward(self, y: Tensor, u: Tensor = None) -> Gaussian:
        yu = nonecat(y, u)
        if yu.ndim == 2:
            yu = yu.unsqueeze(1)
        L, N, _ = yu.shape
        h0 = self.h0.expand(-1, N, -1)  # expand batch axis
        output, h_n = self.gru(yu, h0)  # (L, N, D*H), (D, N, H)
        output = self.hidden_q(output)
        mean, logvar = output.chunk(2, -1)
        h_n = h0.transpose(0, 1).view(N, -1)
        mean0, logvar0 = self.hidden_q(h_n).chunk(2, -1)
        return mean, logvar, mean0, logvar0


class VJF(Module):
    def __init__(self, ydim: int, xdim: int, likelihood: Module, transition: Module, recognition: Module):
        """
        Use VJF.make_model
        :param likelihood: GLM likelihood, Gaussian or Poisson
        :param transition: f(x[t-1], u[t]) -> x[t]
        :param recognition: y[t], f(x[t-1], u[t]) -> x[t]
        :param lr_decay: multiplicative factor of learning rate decay
        """
        super().__init__()
        self.add_module('likelihood', likelihood)
        self.add_module('transition', transition)
        self.add_module('recognition', recognition)
        self.add_module('decoder', LinearDecoder(xdim, ydim))

        self.register_parameter('mean', Parameter(torch.zeros(xdim)))
        self.register_parameter('logvar', Parameter(torch.zeros(xdim)))

        # lr = 1e-4
        # self.optimizer = SGD(
        #     [
        #         {'params': self.likelihood.parameters(), 'lr': lr},
        #         {'params': self.decoder.parameters(), 'lr': lr},
        #         {'params': self.transition.parameters(), 'lr': lr},
        #         {'params': self.recognition.parameters(), 'lr': lr},
        #     ],
        #     lr=lr,
        # )
        # self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)

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

    def forward(self, y: Tensor, u: Tensor = None) -> Tuple:
        """
        :param y: new observation
        :param u: input, None if autonomous
        :return:
            pt: prediction before observation
            qt: posterior after observation
        """
        # encode

        y = torch.atleast_2d(y)
        mean, logvar, mean0, logvar0 = self.recognition(y, u)
        m = torch.stack((mean, mean0))
        lv = torch.stack((logvar, logvar0))
        x = reparametrize(m, lv)

        m1, lv1 = self.transition(x, u)

        x1 = x[:-1, ...]
        # decode
        yhat = self.decoder(x1)  # NOTE: closed-form did not work well

        return yhat, x, m, lv, m1, lv1

    def loss(self, y: Tensor, yhat, x, m, lv, m1, lv1, components: bool = True, warm_up: bool = False):
        # recon
        l_recon = self.likelihood.loss(yhat, y)
        # dynamics
        l_dynamics = gaussian_loss(m1, x[:-1, ...], self.logvar)  # TODO: use posterior variance
        # entropy
        h = entropy(Gaussian(m, lv))

        assert torch.isfinite(l_recon), l_recon.item()
        assert torch.isfinite(l_dynamics), l_dynamics.item()
        assert torch.isfinite(h), h.item()

        loss = l_recon - h
        if not warm_up:
            loss = loss + l_dynamics

        if components:
            return loss, l_recon, l_dynamics, h
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

    @classmethod
    def make_model(cls, ydim: int, xdim: int, udim: int, n_rbf: int, hidden_sizes: Sequence[int],
                   likelihood: str = 'poisson', *args, **kwargs):
        if likelihood.lower() == 'poisson':
            likelihood = PoissonLikelihood()
        elif likelihood.lower() == 'gaussian':
            likelihood = GaussianLikelihood()

        model = VJF(ydim, xdim, likelihood, RBFDS(n_rbf, xdim, udim), GRURecognition(ydim, xdim, udim, hidden_sizes),
                    *args, **kwargs)
        return model

    def forecast(self, x0: Tensor, u: Tensor = None, n_step: int = 1, *, noise: bool = False) -> Tuple[Tensor, Tensor]:
        x = self.transition.forecast(x0, u, n_step, noise=noise)
        y = self.decoder(x)
        return x, y


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

    def forecast(self, x0: Tensor, u: Tensor = None, n_step: int = 1, *, noise: bool = False) -> Tensor:
        x0 = torch.as_tensor(x0, dtype=torch.get_default_dtype())
        x0 = torch.atleast_2d(x0)
        x = torch.empty(n_step + 1, *x0.shape)
        x[0] = x0
        s = torch.exp(.5 * self.logvar)

        if u is None:
            u = [None] * n_step
        else:
            u = torch.as_tensor(u, dtype=torch.get_default_dtype())
            u = torch.atleast_2d(u)
            assert u.shape[0] == n_step, 'u must have length of n_step if present'

        for t in range(n_step):
            x[t + 1] = self.forward(x[t], u[t], sampling=True)
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


def train(model: VJF, y: Tensor, u: Tensor = None, *, max_iter: int = 200, beta: float = .1, verbose: bool = False, rtol: float = 1e-4, lr: float = 1e-4, lr_decay: float = .9):
    optimizer = Adam(
        [
            {'params': model.likelihood.parameters(), 'lr': lr},
            {'params': model.decoder.parameters(), 'lr': lr},
            {'params': model.transition.parameters(), 'lr': lr},
            {'params': model.recognition.parameters(), 'lr': lr},
        ],
        lr=lr,
    )
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    y = torch.atleast_2d(y)
    if u is None:
        u_ = [None]
    else:
        u_ = torch.as_tensor(u, dtype=torch.get_default_dtype())
        u_ = torch.atleast_2d(u_)

    warm_up = True
    epoch_loss = torch.tensor(float('nan'))
    with trange(max_iter) as progress:
        running_loss = torch.tensor(float('nan'))
        for i in progress:
            # collections
            q_seq = []  # maybe deque is better than list?
            losses = []

            q = None  # use prior
            for yt, ut in zip_longest(y, u_):
                q, loss, *elbos = model.filter(yt, ut, q,
                                                sgd=True,
                                                update=True,
                                                verbose=verbose,
                                                warm_up=warm_up,
                                                )
                losses.append(loss)
                q_seq.append(q)
                if verbose:
                    progress.set_postfix({
                        # 'Warm up': str(warm_up),
                        'Loss': running_loss.item(),
                        'Recon': elbos[0].item(),
                        'Dynamics': elbos[1].item(),
                        'Entropy': elbos[2].item(),
                        # 'q norm': torch.norm(q[0]).item(),
                        # 'obs noise': self.likelihood.logvar.exp().item(),
                        # 'state noise': self.transition.logvar.exp().item(),
                        # 'centroid': self.transition.velocity.feature.centroid.mean().item(),
                        # 'width': self.transition.velocity.feature.logwidth.exp().mean().item(),
                    })

            epoch_loss = sum(losses) / len(losses)

            if warm_up:
                if epoch_loss.isclose(running_loss, rtol=rtol):
                    warm_up = False
                    running_loss = epoch_loss
                    print('\nWarm up stopped.\n')
                    model.decoder.requires_grad_(False)  # freeze decoder after warm up
                    m = torch.stack([q.mean for q in q_seq])
                    if isinstance(u_, Tensor) and u_.shape[-1] > 0:
                        u_init = u_[1:, :].reshape(-1, u_.shape[-1])
                    else:
                        u_init = None
                    model.transition.initialize(m[1:].reshape(-1, m.shape[-1]),
                                                m[:-1].reshape(-1, m.shape[-1]),
                                                u_init)
            else:
                if epoch_loss.isclose(running_loss, rtol=rtol):
                    print('\nConverged.\n')
                    break

            running_loss = beta * running_loss + (1 - beta) * epoch_loss if i > 0 else epoch_loss

            progress.set_postfix({
                'Loss': running_loss.item(),
            })

            scheduler.step()

    mu = torch.stack([q.mean for q in q_seq])
    logvar = torch.stack([q.logvar for q in q_seq])
    return mu, logvar, epoch_loss
