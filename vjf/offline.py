from abc import ABCMeta, abstractmethod
from typing import Tuple, Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import Module, Linear, Parameter, GRU
from torch.nn.modules.rnn import RNNCell
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from .distribution import Gaussian
from .functional import gaussian_entropy as entropy, gaussian_loss, normed_linear
from .likelihood import GaussianLikelihood, PoissonLikelihood
from .module import LinearRegression, RBF
from .util import reparametrize, symmetric, running_var, nonecat, flat2d


class LinearDecoder(Module):
    def __init__(self, xdim: int, ydim: int):
        super().__init__()
        self.add_module('decode', Linear(xdim, ydim))
        self.n_sample = 0

    def forward(self, x: Union[Tensor, Gaussian]) -> Union[Tensor, Gaussian]:
        if isinstance(x, Tensor):
            return self.decode(x)
            # return normed_linear(x, self.decode.weight.T, self.decode.bias)
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

        self.add_module(
            'gru',
            GRU(input_size=ydim + udim,
                hidden_size=hidden_size,
                batch_first=False,
                bidirectional=True))
        D = 2  # bidirectional
        self.register_parameter('h0', Parameter(torch.zeros(D, hidden_size)))  # (D, H), batch?

        self.add_module('hidden_q', Linear(hidden_size * 2,
                                           xdim * 2,
                                           bias=True))

    def forward(self, y: Tensor, u: Tensor = None) -> Gaussian:
        yu = nonecat(y, u)
        if yu.ndim == 2:
            yu = yu.unsqueeze(1)
        L, N, _ = yu.shape
        h0 = self.h0.unsqueeze(1).expand(-1, N, -1)  # expand batch axis, (D, N, H)
        h, _ = self.gru(yu, h0)  # (L, N, D*H) <- (L, N, Y), (D, N, H)
        h0 = h0 = h0.transpose(0, 1).reshape(N, -1).unsqueeze(0)  # (D, N, H) -> (N, D, H) -> (N, D*H) -> (1, N, D*H)
        h_all = torch.concat((h0, h), axis=0)
        output = self.hidden_q(h_all)  # (L + 1, N, 2*X)
        mean, logvar = output.chunk(2, -1)
        return mean, logvar


class VJF(Module):
    def __init__(self, ydim: int, xdim: int, likelihood: Module,
                 transition: Module, recognition: Module):
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

    def forward(self, y: Tensor, u: Tensor) -> Tuple:
        """
        :param y: new observation
        :param u: input
        :return:
        """
        # encode

        y = torch.atleast_2d(y)
        m, lv = self.recognition(y, u)
        x = reparametrize((m, lv))  # (L + 1, N, X)
        x0 = x[:-1, ...]  # (L, N, X), 0...T-1
        x1 = x[1:, ...]  # (L, N, X), 1...T
        m1 = self.transition(flat2d(x0), flat2d(u))
        m1 = m1.reshape(*x1.shape)
        # lv1 = torch.ones_like(m1) * self.transition.logvar
        # decode
        yhat = self.decoder(x1)  # NOTE: closed-form did not work well

        return yhat, m, lv, x0, x1, m1

    def loss(self,
             y: Tensor,
             yhat,
             m,
             lv,
             x1,
             m1,
             components: bool = True,
             warm_up: bool = False):
        # recon
        l_recon = self.likelihood.loss(yhat, y)
        # dynamics
        l_dynamics = self.transition.loss(m1, x1)  # TODO: use posterior variance
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
    def update(self,
               y: Tensor,
               xs: Tensor,
               u: Tensor,
               pt: Tensor,
               qt: Gaussian,
               xt: Tensor,
               py: Tensor,
               *,
               likelhood=False,
               decoder=True,
               transition=True,
               recognition=True,
               warm_up=False):
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
    def make_model(cls,
                   ydim: int,
                   xdim: int,
                   udim: int,
                   n_rbf: int,
                   hidden_sizes: Sequence[int],
                   likelihood: str = 'poisson',
                   ds: str = 'rbf',
                   *args,
                   **kwargs):
        if likelihood.lower() == 'poisson':
            likelihood = PoissonLikelihood()
        elif likelihood.lower() == 'gaussian':
            likelihood = GaussianLikelihood()
        
        if ds == 'rbf':
            model = VJF(ydim, xdim, likelihood, RBFDS(n_rbf, xdim, udim),
                        GRURecognition(ydim, xdim, udim, hidden_sizes), *args,
                        **kwargs)
        else:
            model = VJF(ydim, xdim, likelihood, RNNDS(xdim, udim),
                        GRURecognition(ydim, xdim, udim, hidden_sizes), *args,
                        **kwargs)
        return model

    def forecast(self, x0: Tensor, u: Tensor = None, n_step: int = 1, *, noise: bool = False) -> Tuple[Tensor, Tensor]:
        x = self.transition.forecast(x0, u, n_step, noise=noise)
        y = self.decoder(x)
        return x, y


class Transition(Module, metaclass=ABCMeta):
    @abstractmethod
    def velocity(x, u):
        pass

    def forecast(self,
                x0: Tensor,
                u: Tensor = None,
                n_step: int = 1,
                *,
                noise: bool = False) -> Tensor:
        x0 = torch.as_tensor(x0, dtype=torch.get_default_dtype())
        x0 = torch.atleast_2d(x0)
        x = torch.empty(n_step + 1, *x0.shape)
        x[0] = x0
        s = torch.exp(.5 * self.logvar)

        if u is None:
            u = torch.zeros(n_step, x0.shape[0], 0)  # (L, N, 0)
        else:
            u = torch.as_tensor(u, dtype=torch.get_default_dtype())
            u = torch.atleast_2d(u)
            assert u.shape[
                0] == n_step, 'u must have length of n_step if present'

        for t in range(n_step):
            x[t + 1] = self.forward(x[t], u[t], sampling=True)
            if noise:
                x[t + 1] = x[t + 1] + torch.randn_like(x[t + 1]) * s

        return x

    def loss(self, pt: Tensor, qt: Tensor) -> Tensor:
        return gaussian_loss(pt, qt, self.logvar)


class RBFDS(Transition):
    def __init__(self, n_rbf: int, xdim: int, udim: int):
        super().__init__()
        self.add_module('linreg', LinearRegression(RBF(xdim + udim, n_rbf, requires_grad=False), xdim, bayes=True))
        self.register_parameter('logvar',
                                Parameter(torch.tensor(0.),
                                          requires_grad=False))  # state noise
        self.n_sample = 0  # sample counter
    
    def velocity(self, x, sampling=True):
        return self.linreg(x, sampling)

    def forward(self,
                x: Tensor,
                u: Tensor = None,
                sampling: bool = True,
                leak: float = 0.) -> Union[Tensor, Gaussian]:
        xu = nonecat(x, u)
        dx = self.velocity(xu, sampling=sampling)
        # return self.transition(xu, sampling=sampling)
        if isinstance(dx, Gaussian):
            return Gaussian((1 - leak) * x + dx.mean, dx.logvar)
        else:
            return (1 - leak) * x + dx

    @torch.no_grad()
    def update(self,
               xt: Tensor,
               xs: Tensor,
               ut: Tensor = None,
               *,
               warm_up=False):
        """Train regression"""
        xs = torch.atleast_2d(xs)
        xu = nonecat(xs, ut)
        xt = torch.atleast_2d(xt)  # TODO: use qt, add qt.logvar to state noise
        dx = xt - xs
        if not warm_up:
            self.linreg.rls(xu, dx, self.logvar.exp(), shrink=1.)  # model dx
            # self.velocity.kalman(xs, dx, self.logvar.exp(), diffusion=.01)  # model dx
        residual = dx - self.linreg(xu, sampling=False).mean
        mse = residual.pow(2).mean()
        var, n_sample = running_var(self.logvar.exp(),
                                    self.n_sample,
                                    mse,
                                    xs.shape[0],
                                    size_cap=500)
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


class RNNDS(Transition):
    """DS with GRU transition"""
    def __init__(self, xdim: int, udim: int):
        super().__init__()
        self.xdim = xdim
        self.udim = udim
        self.add_module('rnn', RNNCell(input_size=udim, hidden_size=xdim))  # TODO: use GRU, better for offline
        self.register_parameter('logvar',
                                Parameter(torch.tensor(0.),
                                          requires_grad=True))  # state noise
    
    def velocity(self, x, u):
        return self.forward(x, u) - x

    def forward(self,
                x: Tensor,
                u: Tensor,
                sampling: bool = True,
                leak: float = 0.) -> Union[Tensor, Gaussian]:
        return self.rnn(u, x)


def train(model: VJF,
          y: Tensor,
          u: Tensor = None,
          *,
          max_iter: int = 200,
          beta: float = 0.,
          verbose: bool = False,
          rtol: float = 1e-5,
          lr: float = 1e-3,
          lr_decay: float = .99):
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    y = torch.atleast_2d(y)

    L, N, _ = y.shape  # 3D, time first

    if u is None:
        udim = 0
        u = torch.zeros(L, N, udim)  # 0D input
    udim = u.shape[-1]

    with trange(max_iter) as progress:
        running_loss = torch.tensor(float('nan'))
        for i in progress:
            # collections

            yhat, m, lv, x0, x1, m1 = model.forward(y, u)
            total_loss, loss_recon, loss_dynamics, h = model.loss(y, yhat, m, lv, x1, m1, components=True, warm_up=False)
            
            kl_scale = torch.sigmoid(torch.tensor(i, dtype=torch.get_default_dtype()) - 10)
            total_loss = loss_recon + kl_scale * (loss_dynamics - h)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            if total_loss.isclose(running_loss, rtol=rtol):
                print('\nConverged.\n')
                break

            running_loss = beta * running_loss + (1 - beta) * total_loss if i > 0 else total_loss
            # print(total_loss)

            progress.set_postfix({
                'Loss': running_loss.item(),
                'KL scale': kl_scale.item(),
            })

            scheduler.step()

            if torch.isclose(kl_scale, torch.tensor(1.)):
                model.update(flat2d(y), flat2d(x0),  flat2d(u), None, None, flat2d(x1), flat2d(yhat))
            #    y: Tensor,
            #    xs: Tensor,
            #    u: Tensor,
            #    pt: Tensor,
            #    qt: Gaussian,
            #    xt: Tensor,
            #    py: Tensor,

    return m, lv
