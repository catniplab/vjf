"""
Offline mode
For autonomous systems, use 0D input
"""
from abc import ABCMeta, abstractmethod
import math
from typing import Tuple, Sequence, Union, List

import torch
from torch import Tensor, nn, linalg
from torch.nn import functional
from torch.optim import AdamW, lr_scheduler
from tqdm import trange

from .distribution import Gaussian
from .module import RBFN
from .util import ensure_2d, ensure_3d, reparametrize, flat2d, at_least2d
from .functional import gaussian_kl
from .regression import *


###
def gaussian_entropy(q) -> Tensor:
    """Elementwise Gaussian entropy"""
    _, logvar = q
    assert logvar.ndim >= 2
    return .5 * logvar.sum()


def gaussian_loss(a, b, logvar: Tensor) -> Tensor:
    """
    Negative Gaussian log-likelihood
    0.5 * [1/sigma^2 (a - b)^2 + log(sigma^2) + log(2*pi)]
    :param a: Tensor
    :param b: Tensor
    :param logvar: log(sigma^2)
    :return:
        (expected) Gaussian loss
    """
    a = at_least2d(a)
    b = at_least2d(b)

    p = torch.exp(-.5 * logvar)  # 1/sigma

    mse = functional.mse_loss(a * p, b * p, reduction='none')
    assert mse.ndim >= 2
    assert torch.all(torch.isfinite(mse)), mse

    nll = .5 * (mse + logvar + math.log(2 * math.pi))

    return nll.sum()


###


class GaussianLikelihood(nn.Module):
    """
    Gaussian likelihood
    """
    def __init__(self):
        super().__init__()
        self.register_parameter('logvar', nn.Parameter(torch.tensor(.0)))

    def loss(self, eta: Tensor, target: Tensor) -> Tensor:
        """
        :param eta: pre inverse link
        :param target: observation
        :return:
            negative likelihood
        """
        return gaussian_loss(target, eta, self.logvar)


class PoissonLikelihood(nn.Module):
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
        if not isinstance(eta, Tensor):
            raise NotImplementedError
        nll = functional.poisson_nll_loss(eta.clamp(max=10.),
                                          target,
                                          log_input=True,
                                          reduction='none')
        # assert nll.ndim == 2
        return nll.sum()


class LinearDecoder(nn.Module):
    def __init__(self, xdim: int, ydim: int, norm='none'):
        super().__init__()
        self.norm = norm
        self.register_parameter('weight',
                                nn.Parameter(torch.empty(ydim,
                                                         xdim)))  # (out, in)
        self.register_parameter('bias', nn.Parameter(torch.empty(ydim)))
        self.reset_parameters()

    def reset_parameters(self):
        # copy from torch Linear layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self.norm == 'none':
            return functional.linear(x, w, self.bias)

        normalizer = 1.
        if self.norm == 'svd':
            w = linalg.svd(w, False).U
        elif self.norm == 'fro':
            normalizer = torch.linalg.norm(w, keepdim=True)
        elif self.norm == 'row':
            normalizer = torch.linalg.norm(w, dim=0, keepdim=True)
        elif self.norm == 'col':
            normalizer = torch.linalg.norm(w, dim=1, keepdim=True)

        w = w / normalizer
        return functional.linear(x, w, self.bias)


class GRUEncoder(nn.Module):
    def __init__(self, ydim: int, xdim: int, udim: int, hidden_size: int):
        super().__init__()

        self.add_module(
            'gru',
            nn.GRU(input_size=ydim + udim,
                   hidden_size=hidden_size,
                   batch_first=True,
                   bidirectional=True))

        D = 2  # bidirectional
        self.register_parameter('e_0', nn.Parameter(torch.zeros(
            D, hidden_size)))  # initial encoder hidden state (D, H)

        self.register_parameter('w_x',
                                nn.Parameter(torch.randn(
                                    xdim, hidden_size *
                                    D)))  # output layer weight for state mean
        self.register_parameter(
            'w_x_0', nn.Parameter(
                torch.randn(xdim, hidden_size *
                            D)))  # output layer weight for initial state mean
        self.add_module('hidden2logvar',
                        nn.Linear(hidden_size * D, xdim, bias=True))  # logvar
        self.add_module('hidden2logvar_0',
                        nn.Linear(hidden_size * D, xdim,
                                  bias=True))  # logvar_0

    def forward(self, y: Tensor, u: Tensor) -> Gaussian:
        y = torch.cat([y, u], dim=-1)  # append input
        y = ensure_3d(y)
        N, _, _ = y.shape

        e_0 = self.e_0.unsqueeze(1).expand(-1, N,
                                           -1)  # expand batch axis, (D, N, H)
        e, e_n = self.gru(
            y, e_0)  # (N, L, Y), (D, N, H) -> (N, L, D*H), (D, N, H)

        e_n = torch.swapaxes(e_n, 0, 1)  # (D, N, H) -> (N, D, H)
        e_n = e_n.reshape(N, -1)  # (D, N, H) -> (N, D*H)

        logvar_0 = self.hidden2logvar_0(e_n)  # (N, D*H) -> (N, X)
        logvar = self.hidden2logvar(e)  # (N, L, D*H) -> (N, L, X)

        w_x = linalg.svd(self.w_x, full_matrices=False).Vh / math.sqrt(
            self.w_x.shape[0])  # normalize F-norm of V is sqrt(min(m, n))
        w_x_0 = linalg.svd(self.w_x_0, full_matrices=False).Vh / math.sqrt(
            self.w_x_0.shape[0])

        mu_0 = functional.linear(e_n, w_x_0)  # (N, D*H) -> (N, X)
        mu = functional.linear(e, w_x)  # (N, L, D*H) -> (N, L, X)

        return (mu_0, logvar_0), (mu, logvar)


class VJF(nn.Module):
    def __init__(self, ydim: int, xdim: int, udim: int, likelihood: nn.Module,
                 transition: nn.Module, encoder: nn.Module):
        """
        Use VJF.make_model
        :param likelihood: GLM likelihood, Gaussian or Poisson
        :param transition: f(x[t-1]) -> x[t]
        :param encoder: y -> x
        """
        super().__init__()
        self.add_module('likelihood', likelihood)
        self.add_module('transition', transition)
        self.add_module('encode', encoder)
        self.add_module('decode', LinearDecoder(xdim, ydim, norm='none'))

    def forward(self, y: Tensor, u: Tensor) -> Tuple:
        """
        :param y: new observation
        :param u: input
        :return:
        """
        # encode
        y = ensure_3d(y)
        u = ensure_3d(u)
        q_0, q = self.encode(y, u)
        mu_0, logvar_0 = q_0
        p_0 = (torch.zeros_like(mu_0), torch.zeros_like(logvar_0))
        return p_0, q_0, q

    def loss(
        self,
        y: Tensor,
        u: Tensor,
        p_0: Tuple[Tensor, Tensor],
        q_0: Tuple[Tensor, Tensor],
        q: Tuple[Tensor, Tensor],
        kl_scale: float=0.
    ):
        y = ensure_3d(y)
        u = ensure_3d(u)

        x_0 = reparametrize(q_0)  # (N, X)
        x = reparametrize(q)  # (N, L, X)

        # x_pred = self.generate(x_0, u)
        x_cond = torch.cat([x_0.unsqueeze(1), x[:, :-1, :]],
                           dim=1)  # x_{0, ..., L-1}
        x_pred = self.transition(flat2d(x_cond), flat2d(u))
        x_pred = x_pred.reshape(*x_cond.shape)
        assert x.shape == x_pred.shape

        yhat = self.decode(x)  # NOTE: closed-form did not work well

        # recon
        l_recon = self.likelihood.loss(yhat, y)

        p = (x_pred, torch.ones_like(x_pred) * self.transition.logvar)
        kl_x = gaussian_kl(q, p)
        kl_x_0 = gaussian_kl(q_0, p_0)
        kl_params = self.kl()

        assert torch.isfinite(l_recon)
        assert torch.isfinite(kl_x)
        assert torch.isfinite(kl_x_0)

        loss = l_recon + kl_x + kl_x_0 + kl_params * kl_scale

        return loss
    
    def kl(self):
        if hasattr(self.transition, 'kl'):
            return self.transition.kl()
        else:
            return 0.

    def generate(self, x_0, u):
        x_0 = ensure_2d(x_0)  # (N, X)
        u = ensure_3d(u)  # (N, L, U)
        L = u.shape[1]

        x = torch.empty(L + 1, *x_0.shape)
        x[0] = x_0

        for t in range(L):
            x[t + 1, ...] = self.transition(x[t], u[:, t, :])

        x = x[1:, ...]
        return x.transpose(0, 1)
    
    def sample(self):
        if hasattr(self.transition, 'sample'):
            self.transition.sample()

    @torch.no_grad()
    def update(self, y, u, p_0, q_0, q):
        if hasattr(self.transition, 'update'):
            u = torch.cat([ensure_3d(u_k) for u_k in u], 1)  # k-th batch
            mu0 = torch.cat([
                torch.cat((ensure_3d(q_0_k[0]), q_k[0][:, :-1, :]), 1)
                for q_0_k, q_k in zip(q_0, q)
            ], 1)
            mu1 = torch.cat([q_k[0] for q_k in q], 1)
            self.transition.update(mu0, u, mu1)

    @classmethod
    def make_model(cls,
                   ydim: int,
                   xdim: int,
                   udim: int,
                   n_rbf: int,
                   n_layer: int,
                   ds_bias: bool,
                   edim: int,
                   likelihood: str = 'poisson',
                   ds: str = 'rbf',
                   normalized_rbfn: bool = False,
                   state_logvar: float = 0.,
                   *args,
                   **kwargs):
        if likelihood.lower() == 'poisson':
            likelihood = PoissonLikelihood()
        elif likelihood.lower() == 'gaussian':
            likelihood = GaussianLikelihood()

        encoder = GRUEncoder(ydim, xdim, udim, edim)
        if ds == 'rbf':
            state = RBFDS(xdim, udim, n_rbf, normalized_rbfn,
                          state_logvar)
        if ds == 'bayesrbf':
            state = BayesRBFDS(xdim, udim, n_rbf, normalized_rbfn,
                               state_logvar)
        elif ds == 'bayesrff':
            state = BayesRFFDS(xdim, udim, n_rbf, normalized_rbfn,
                               state_logvar)
        elif ds == 'mlp':
            state = MLPDS(xdim, udim, n_rbf, n_layer, state_logvar)
        elif ds == 'vi':
            state = VIRBFDS(xdim, udim, n_rbf, normalized_rbfn, state_logvar)
        model = VJF(ydim, xdim, udim, likelihood, state, encoder, *args,
                    **kwargs)

        return model

    def forecast(self,
                 x0: Tensor,
                 u: Tensor,
                 n_step: int = 1,
                 *,
                 noise: bool = False) -> Tuple[Tensor, Tensor]:
        x = self.transition.forecast(x0, u, n_step, noise=noise)
        y = self.decode(x)
        return x, y


class Transition(nn.Module, metaclass=ABCMeta):
    def __init__(self, logvar=0.) -> None:
        super().__init__()
        self.register_parameter(
            'logvar', nn.Parameter(torch.tensor(logvar),
                                   requires_grad=False))  # state noise

    @abstractmethod
    def velocity(x, u):
        pass

    def forecast(self,
                 x0: Tensor,
                 u: Tensor,
                 n_step: int = 1,
                 *,
                 noise: bool = False) -> Tensor:
        x0 = torch.as_tensor(x0, dtype=torch.get_default_dtype())
        x0 = torch.atleast_2d(x0)
        u = torch.atleast_2d(u)
        assert u.shape[1] == n_step

        x = torch.empty(n_step + 1, *x0.shape)
        x[0] = x0
        s = torch.exp(.5 * self.logvar)

        for t in range(n_step):
            x[t + 1, ...] = self.forward(x[t], u[:, t, :])
            if noise:
                x[t + 1, ...] = x[t + 1] + torch.randn_like(x[t + 1]) * s

        return x.transpose(0, 1)

    def loss(self, pt: Tensor, qt: Tensor) -> Tensor:
        return gaussian_loss(pt, qt, self.logvar)

    def kl(self):
        return torch.tensor(0.)


class MLPDS(Transition):
    def __init__(self,
                 xdim: int,
                 udim: int,
                 hidden_size: int,
                 n_layer: int,
                 logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__(logvar)
        layers = [nn.Linear(xdim + udim, hidden_size), nn.Tanh()]
        for k in range(n_layer):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, xdim))

        self.add_module('predict', nn.Sequential(*layers))

    def velocity(self, x, u):
        x = torch.cat([x, u], dim=-1)
        return self.predict(x)

    def forward(self, x: Tensor, u: Tensor, leak: float = 0.) -> Tensor:
        dx = self.velocity(x, u)
        return x + dx


class RBFDS(Transition):
    def __init__(self,
                 xdim: int,
                 udim: int,
                 n_basis: int,
                 normalized=False,
                 logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__(logvar)
        self.add_module(
            'predict',
            RBFN(in_features=xdim + udim,
                 out_features=xdim,
                 n_basis=n_basis,
                 normalized=normalized))

    def velocity(self, x, u):
        x = torch.cat([x, u], dim=-1)
        return self.predict(x)

    def forward(self, x: Tensor, u: Tensor, leak: float = 0.) -> Tensor:
        dx = self.velocity(x, u)
        return (1 - leak) * x + dx


class BayesRBFDS(Transition):
    def __init__(self,
                 xdim: int,
                 udim: int,
                 n_basis: int,
                 normalized=False,
                 logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__(logvar)
        self.n_basis = n_basis
        self.normalized = normalized

        center = (torch.rand(n_basis, xdim + udim) - .5) * 2  # (-1, 1)
        center = center + torch.randn_like(center)  # jitter        
        pdist = functional.pdist(center)
        logscale = torch.log(pdist.max() / math.sqrt(2 * n_basis))
        feature = RBFFeature(center, logscale, normalized=normalized)
        self.add_module('lm', LinearModel(feature, xdim, beta=math.exp(-logvar)))

    @torch.no_grad()
    def update(self, x0, u, x1):
        x_0_2d = flat2d(x0)
        u_2d = flat2d(u)
        x_1_2d = flat2d(x1)
        dx_2d = x_1_2d - x_0_2d
        xu_2d = torch.cat((x_0_2d, u_2d), -1)
        self.lm.update(xu_2d, dx_2d)

    def velocity(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        return self.lm(xu)

    def forward(self, x: Tensor, u: Tensor, leak: float = 0.) -> Tensor:
        dx = self.velocity(x, u)
        return (1 - leak) * x + dx

    def loss(self, pt: Tensor, qt: Tensor) -> Tensor:
        # logvar = reparametrize((self.noise_mean, self.noise_logvar))
        logvar = self.logvar
        return gaussian_loss(pt, qt, logvar)


class BayesRFFDS(Transition):
    def __init__(self,
                 xdim: int,
                 udim: int,
                 n_basis: int,
                 normalized=False,
                 logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__(logvar)
        self.n_basis = n_basis
        self.normalized = normalized

        feature = RFFFeature(xdim + udim, xdim, ndim=n_basis, logscale=2.5)
        self.add_module('lm',
                        LinearModel(feature, xdim, beta=math.exp(-logvar)))

    @torch.no_grad()
    def update(self, x0, u, x1):
        x_0_2d = flat2d(x0)
        u_2d = flat2d(u)
        x_1_2d = flat2d(x1)
        dx_2d = x_1_2d - x_0_2d
        xu_2d = torch.cat((x_0_2d, u_2d), -1)
        self.lm.update(xu_2d, dx_2d)

    def velocity(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        return self.lm(xu)

    def forward(self, x: Tensor, u: Tensor, leak: float = 0.) -> Tensor:
        dx = self.velocity(x, u)
        return x + dx

    def loss(self, pt: Tensor, qt: Tensor) -> Tensor:
        # logvar = reparametrize((self.noise_mean, self.noise_logvar))
        logvar = self.logvar
        return gaussian_loss(pt, qt, logvar)

    def kl(self):
        # kl_b = gaussian_kl((self.b_mean, self.b_logvar), (self.b_prior_mean, self.b_prior_logvar)) if self.bias else 0.
        # kl_w = gaussian_kl((self.w_mean, self.w_logvar), (self.w_prior_mean, self.w_prior_logvar))
        # kl_noise = gaussian_kl((self.noise_mean, self.noise_logvar), (self.noise_prior_mean, self.noise_prior_logvar))
        kl_b = 0.
        kl_w = 0.
        kl_noise = 0.
        return kl_b + kl_w + kl_noise


class VIRBFDS(Transition):
    def __init__(self,
                 xdim: int,
                 udim: int,
                 n_basis: int,
                 normalized=False,
                 logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__(logvar)
        self.n_basis = n_basis
        self.normalized = normalized

        center = (torch.rand(n_basis, xdim + udim) - .5) * 2  # (-1, 1)
        # center = center + torch.randn_like(center)  # jitter        
        pdist = functional.pdist(center)
        logscale = torch.log(pdist.max() / math.sqrt(2 * n_basis))
        feature = RBFFeature(center, logscale, normalized=normalized)
        self.add_module('lm', LinearModelVI(feature, xdim))

    @torch.no_grad()
    def update(self, x0, u, x1):
        x_0_2d = flat2d(x0)
        u_2d = flat2d(u)
        x_1_2d = flat2d(x1)
        dx_2d = x_1_2d - x_0_2d
        xu_2d = torch.cat((x_0_2d, u_2d), -1)
        self.lm.update(xu_2d, dx_2d)
        pass

    def velocity(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        return self.lm(xu)

    def forward(self, x: Tensor, u: Tensor, leak: float = 0.) -> Tensor:
        dx = self.velocity(x, u)
        return (1 - leak) * x + dx

    def loss(self, pt: Tensor, qt: Tensor) -> Tensor:
        # logvar = reparametrize((self.noise_mean, self.noise_logvar))
        # logvar = self.lm.logvar
        logvar = self.logvar
        return gaussian_loss(pt, qt, logvar)

    def kl(self):
        return self.lm.kl()


def train(
    model: VJF,
    y: List,
    u: List,
    *,
    max_iter: int = 200,
    beta: float = 0.,
    verbose: bool = False,
    rtol: float = 1e-05,
    lr: float = 1e-3,
):
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    y_seq = y
    u_seq = u

    assert len(y_seq) == len(u_seq)

    losses = []
    with trange(max_iter) as pbar:
        running_loss = torch.tensor(float('nan'))
        for i in pbar:
            epoch_loss = 0.
            seqs = []
            Ls = 0
            for k in torch.randperm(len(y_seq)):
                y = y_seq[k]
                u = u_seq[k]

                y = torch.as_tensor(y, dtype=torch.get_default_dtype())
                y = ensure_3d(y)

                u = torch.as_tensor(u, dtype=torch.get_default_dtype())
                u = ensure_3d(u)

                N, L, _ = y.shape  # batch first for offline

                model.sample()
                p_0, q_0, q = model.forward(y, u)
                trial_loss = model.loss(y, u, p_0, q_0, q, kl_scale=1/len(y_seq))
                
                optimizer.zero_grad()
                trial_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                optimizer.step()
                epoch_loss += trial_loss.detach()

                Ls += L

                seqs.append((y, u, p_0, q_0, q))

            model.update(*zip(*seqs))

            # epoch_loss += model.transition.kl()
            epoch_loss /= Ls

            if torch.isclose(epoch_loss, running_loss, rtol=rtol):
                print('\nConverged.\n')
                break

            running_loss = beta * running_loss + (
                1 - beta) * epoch_loss.detach() if i > 0 else epoch_loss.detach()
            # print(total_loss)

            pbar.set_postfix({
                'Loss': f'{running_loss:.5f}',
                # 'KL scale': kl_scale.item(),
            })
            losses.append(epoch_loss.detach())

            # scheduler.step()

    mu_seq = []
    logvar_seq = []

    for y, u in zip(y_seq, u_seq):
        y = torch.as_tensor(y, dtype=torch.get_default_dtype())
        y = ensure_3d(y)

        u = torch.as_tensor(u, dtype=torch.get_default_dtype())
        u = ensure_3d(u)

        p_0, q_0, q = model.forward(y, u)

        mu_seq.append(q[0].detach())
        logvar_seq.append(q[1].detach())

    return losses, mu_seq, logvar_seq
