"""
Offline mode
For autonomous systems, use 0D input
"""
from abc import ABCMeta, abstractmethod
import math
from typing import Tuple, Sequence, Union, List

import torch
from torch import Tensor, nn, linalg
from torch.nn import Module, Linear, Parameter, GRU, functional, GRUCell, RNNCell
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from .distribution import Gaussian
from .module import RBFN
from .util import ensure_2d, ensure_3d, reparametrize, flat2d, at_least2d
from .functional import rbf


from .regression import *

###
def gaussian_entropy(q) -> Tensor:
    """Elementwise Gaussian entropy"""
    _, logvar = q
    assert logvar.ndim >= 2
    return .5 * logvar.sum()


def gaussian_kl(q, p):
    """
    Elementwise KL(q|p) = 0.5 * [tr(v_q/v_p) - 1 + (m_q - m_p)^2 / v_p + log(var_p/var_q)]
    """
    m_q, logvar_q = q
    m_p, logvar_p = p

    var_p = torch.exp(logvar_p)

    trace = torch.exp(logvar_p - logvar_q)
    logdet = logvar_q - logvar_p

    kl = .5 * ((m_q - m_p) ** 2 / var_p + logdet + trace - 1)
    return kl.sum()


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

class GaussianLikelihood(Module):
    """
    Gaussian likelihood
    """

    def __init__(self):
        super().__init__()
        self.register_parameter('logvar', Parameter(torch.tensor(.0)))

    def loss(self, eta: Tensor, target: Tensor) -> Tensor:
        """
        :param eta: pre inverse link
        :param target: observation
        :return:
            negative likelihood
        """
        return gaussian_loss(target, eta, self.logvar)


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
        if not isinstance(eta, Tensor):
            raise NotImplementedError
        nll = functional.poisson_nll_loss(eta.clamp(max=10.), target, log_input=True, reduction='none')
        # assert nll.ndim == 2
        return nll.sum()


class LinearDecoder(Module):
    def __init__(self, xdim: int, ydim: int, norm='none'):
        super().__init__()
        self.norm = norm
        self.register_parameter('weight', Parameter(torch.empty(ydim, xdim)))  # (out, in)
        self.register_parameter('bias', Parameter(torch.empty(ydim)))
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


class GRUEncoder(Module):
    def __init__(self, ydim: int, xdim: int, udim: int, hidden_size: int):
        super().__init__()

        self.add_module(
            'gru',
            GRU(input_size=ydim + udim,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True))

        D = 2  # bidirectional
        self.register_parameter('e_0', Parameter(torch.zeros(D, hidden_size)))  # initial encoder hidden state (D, H)
        
        self.register_parameter('w_x_t', Parameter(torch.randn(xdim, hidden_size * D)))  # output layer weight for state mean
        self.register_parameter('w_x_0', Parameter(torch.randn(xdim, hidden_size * D)))  # output layer weight for initial state mean
        self.add_module('hidden2logvar_t', Linear(hidden_size * D, xdim, bias=True))  # logvar_t
        self.add_module('hidden2logvar_0', Linear(hidden_size * D, xdim, bias=True))  # logvar_0

    def forward(self, y: Tensor, u: Tensor) -> Gaussian:
        y = torch.cat([y, u], dim=-1)  # append input
        y = ensure_3d(y)
        N, _, _ = y.shape

        e_0 = self.e_0.unsqueeze(1).expand(-1, N, -1)  # expand batch axis, (D, N, H)
        e_t, e_n = self.gru(y, e_0)  # (N, L, Y), (D, N, H) -> (N, L, D*H), (D, N, H)
        
        e_n = torch.swapaxes(e_n, 0, 1)  # (D, N, H) -> (N, D, H)
        e_n = e_n.reshape(N, -1)  # (D, N, H) -> (N, D*H)
        
        logvar_0 = self.hidden2logvar_0(e_n)  # (N, D*H) -> (N, X)
        logvar_t = self.hidden2logvar_t(e_t)  # (N, L, D*H) -> (N, L, X)

        w_x_t = linalg.svd(self.w_x_t, full_matrices=False).Vh  # normalize
        w_x_0 = linalg.svd(self.w_x_0, full_matrices=False).Vh

        mu_0 = functional.linear(e_n, w_x_0)  # (N, D*H) -> (N, X)
        mu_t = functional.linear(e_t, w_x_t)  # (N, L, D*H) -> (N, L, X)

        return (mu_0, logvar_0), (mu_t, logvar_t)


class VJF(Module):
    def __init__(self, ydim: int, xdim: int, udim: int, likelihood: Module,
                 transition: Module, encoder: Module):
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

        y = torch.atleast_2d(y)
        u = torch.atleast_2d(u)
        m, lv = self.encode(y, u)
        x = reparametrize((m, lv))  # (N, L + 1, X)
        x0 = x[:, :-1, :]  # (N, L, X), 0...T-1
        x1 = x[:, 1:, :]  # (N, L, X), 1...T
        m1 = self.transition(flat2d(x0), flat2d(u))
        m1 = m1.reshape(*x1.shape)
        # lv1 = torch.ones_like(m1) * self.transition.logvar
        # decode
        yhat = self.decode(x1)  # NOTE: closed-form did not work well

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
        h = gaussian_entropy(Gaussian(m, lv))
        kl = self.transition.kl()  # KL

        assert torch.isfinite(l_recon), l_recon.item()
        assert torch.isfinite(l_dynamics), l_dynamics.item()
        assert torch.isfinite(h), h.item()

        loss = l_recon - h + kl
        if not warm_up:
            loss = loss + l_dynamics

        batch = y.shape[0]

        if components:
            return loss / batch, l_recon / batch, l_dynamics / batch, h / batch
        else:
            return loss / batch

    @classmethod
    def make_model(cls,
                   ydim: int,
                   xdim: int,
                   udim: int,
                   n_rbf: int,
                   ds_bias: bool,
                   hidden_sizes: Sequence[int],
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
        
        if ds == 'rbf':
            model = VJF(ydim, xdim, udim, likelihood, RBFDS(xdim, udim, n_rbf, ds_bias, normalized_rbfn, state_logvar),
                            GRUEncoder(ydim, xdim, udim, hidden_sizes), *args,
                            **kwargs)
        if ds == 'bayesrbf':
            model = VJF(ydim, xdim, udim, likelihood, BayesRBFDS(xdim, udim, n_rbf, ds_bias, normalized_rbfn, state_logvar),
                            GRUEncoder(ydim, xdim, udim, hidden_sizes), *args,
                            **kwargs)
        elif ds == 'bayesrff':
            model = VJF(ydim, xdim, udim, likelihood, BayesRFFDS(xdim, udim, n_rbf, ds_bias, normalized_rbfn, state_logvar),
                            GRUEncoder(ydim, xdim, udim, hidden_sizes), *args,
                            **kwargs)

        return model

    def forecast(self, x0: Tensor, u: Tensor, n_step: int = 1, *, noise: bool = False) -> Tuple[Tensor, Tensor]:
        x = self.transition.forecast(x0, u, n_step, noise=noise)
        y = self.decode(x)
        return x, y


class Transition(Module, metaclass=ABCMeta):
    def __init__(self, logvar=0.) -> None:
        super().__init__()
        self.register_parameter('logvar', Parameter(torch.tensor(logvar), requires_grad=False))  # state noise

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
        

class RBFDS(Transition):
    def __init__(self, xdim: int, udim: int, n_basis: int, bias=True, normalized=False, logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__(logvar)
        self.add_module('predict', RBFN(in_features=xdim + udim, out_features=xdim, n_basis=n_basis, bias=bias, normalized=normalized))
    
    def velocity(self, x, u):
        x = torch.cat([x, u], dim=-1)
        return self.predict(x)

    def forward(self,
                x: Tensor,
                u: Tensor,
                leak: float = 0.) -> Tensor:
        dx = self.velocity(x, u)
        return (1 - leak) * x + dx


# class BayesRBFDS(Transition):
#     def __init__(self, xdim: int, udim: int, n_basis: int, bias=True, normalized=False, logvar=0.):
#         """
#         param xdim: state dimensionality
#         param udim: input dimensionality
#         param n_basis: number of radial basis functions
#         """
#         super().__init__(logvar)
#         self.n_basis = n_basis
#         self.bias = bias
#         self.normalized = normalized
#         self.centroid = torch.randn(n_basis, xdim + udim)
#         self.logscale = torch.zeros(1, n_basis)
#         # self.register_parameter('centroid', Parameter(torch.randn(n_basis, xdim + udim), requires_grad=False))
#         # self.register_parameter('logscale', Parameter(torch.zeros(1, n_basis), requires_grad=False))  # singleton dim for broadcast over batches

#         self.register_parameter('b_mean', Parameter(torch.zeros(xdim)))
#         self.register_parameter('b_logvar', Parameter(torch.zeros(xdim)))
#         self.register_parameter('b_prior_mean', Parameter(torch.zeros(xdim), requires_grad=False))
#         self.register_parameter('b_prior_logvar', Parameter(torch.zeros(xdim), requires_grad=False))
        
#         self.register_parameter('w_mean', Parameter(torch.zeros(xdim, n_basis)))
#         self.register_parameter('w_logvar', Parameter(torch.zeros(xdim, n_basis)))
#         self.register_parameter('w_prior_mean', Parameter(torch.zeros(xdim, n_basis), requires_grad=False))
#         self.register_parameter('w_prior_logvar', Parameter(torch.zeros(xdim, n_basis), requires_grad=False))

#         self.register_parameter('noise_mean', Parameter(torch.tensor(logvar)))
#         self.register_parameter('noise_logvar', Parameter(torch.tensor(0.)))
#         self.register_parameter('noise_prior_mean', Parameter(torch.tensor(logvar), requires_grad=False))
#         self.register_parameter('noise_prior_logvar', Parameter(torch.tensor(0.), requires_grad=False))
    
#     @torch.no_grad()
#     def update(self, x, u):
#         x2d = torch.concat([m[0, 1:, :] for m in x], dim=0)
#         u2d = torch.concat([u for u in u], dim=0)

#         xu = torch.cat([x2d, u2d], dim=-1)
#         cidx = torch.multinomial(torch.ones(self.n_basis), self.n_basis)
#         center = xu[cidx]
#         pdist = functional.pdist(center)
#         self.logscale.fill_(torch.log(pdist.max() / math.sqrt(self.n_basis)))
#         self.centroid = center

    
#     def velocity(self, x, u):
#         eps = 1e-8
#         x = torch.cat([x, u], dim=-1)
#         h = rbf(x, self.centroid, self.logscale.exp())
#         if self.normalized:
#             h = h / (h.sum(-1, keepdim=True) + eps)
        
#         w = reparametrize((self.w_mean, self.w_logvar))
#         b = reparametrize((self.b_mean, self.b_logvar)) if self.bias else None
#         return functional.linear(h, w, b)

#     def forward(self,
#                 x: Tensor,
#                 u: Tensor,
#                 leak: float = 0.) -> Tensor:
#         dx = self.velocity(x, u)
#         return (1 - leak) * x + dx

#     def loss(self, pt: Tensor, qt: Tensor) -> Tensor:
#         # logvar = reparametrize((self.noise_mean, self.noise_logvar))
#         logvar = self.logvar
#         return gaussian_loss(pt, qt, logvar)

#     def kl(self):
#         kl_b = gaussian_kl((self.b_mean, self.b_logvar), (self.b_prior_mean, self.b_prior_logvar)) if self.bias else 0.
#         kl_w = gaussian_kl((self.w_mean, self.w_logvar), (self.w_prior_mean, self.w_prior_logvar)) 
#         # kl_noise = gaussian_kl((self.noise_mean, self.noise_logvar), (self.noise_prior_mean, self.noise_prior_logvar))
#         kl_noise = 0.
#         return kl_b + kl_w + kl_noise


class BayesRBFDS(Transition):
    def __init__(self, xdim: int, udim: int, n_basis: int, bias=True, normalized=False, logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__(logvar)
        self.n_basis = n_basis
        self.bias = bias
        self.normalized = normalized

        center = torch.randn(n_basis, xdim + udim)
        logscale = torch.zeros(1, n_basis)
        feature = RBFFeature(center, logscale, normalized=normalized)
        self.lm = LinearModel(feature, xdim, beta=math.exp(-logvar))

    @torch.no_grad()
    def train(self, x, u):
        x2d0 = torch.concat([m.squeeze()[:-1, :] for m in x], dim=0)  # x0...
        x2d1 = torch.concat([m.squeeze()[1:, :] for m in x], dim=0)  # x1...
        dx2d = x2d1 - x2d0
        u2d = torch.concat([u for u in u], dim=0)
        xu = torch.cat([x2d0, u2d], dim=-1)
        self.lm.train(xu, dx2d)
    
    def velocity(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        return self.lm.sample(xu)

    def forward(self,
                x: Tensor,
                u: Tensor,
                leak: float = 0.) -> Tensor:
        dx = self.velocity(x, u)
        return (1 - leak) * x + dx

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


class BayesRFFDS(Transition):
    def __init__(self, xdim: int, udim: int, n_basis: int, bias=True, normalized=False, logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__(logvar)
        self.n_basis = n_basis
        self.bias = bias
        self.normalized = normalized

        feature = RFFFeature(xdim + udim, xdim, ndim=n_basis, logscale=logvar)
        self.lm = LinearModel(feature, xdim, beta=math.exp(-logvar))

    @torch.no_grad()
    def train(self, x, u):
        x2d0 = torch.concat([m.squeeze()[:-1, :] for m in x], dim=0)  # x0...
        x2d1 = torch.concat([m.squeeze()[1:, :] for m in x], dim=0)  # x1...
        dx2d = x2d1 - x2d0
        u2d = torch.concat([u for u in u], dim=0)
        xu = torch.cat([x2d0, u2d], dim=-1)
        self.lm.train(xu, dx2d)
    
    def velocity(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        return self.lm.sample(xu)
        # dx = self.velocity(x, u)
        # return self.forward(x, u) - x

    def forward(self,
                x: Tensor,
                u: Tensor,
                leak: float = 0.) -> Tensor:
        # xu = torch.cat([x, u], dim=-1)
        # return self.lm.sample(xu)
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


def train(model: VJF,
          y: Tensor,
          u: Tensor,
          *,
          max_iter: int = 200,
          beta: float = 0.,
          verbose: bool = False,
          rtol: float = 1e-5,
          lr: float = 1e-3,
          lr_decay: float = .99):
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)

    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    y = torch.atleast_2d(y)

    u = torch.as_tensor(u, dtype=torch.get_default_dtype())
    u = torch.atleast_2d(u)

    N, L, _ = y.shape  # 3D, time first
    losses = []
    with trange(max_iter) as progress:
        running_loss = torch.tensor(float('nan'))
        for i in progress:
            # collections

            yhat, m, lv, x0, x1, m1 = model.forward(y, u)
            total_loss, loss_recon, loss_dynamics, h = model.loss(y, yhat, m, lv, x1, m1, components=True, warm_up=False)
            
            # kl_scale = torch.sigmoid(torch.tensor(i, dtype=torch.get_default_dtype()) - 10)
            kl_scale = 1.
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
                # 'KL scale': kl_scale.item(),
            })
            losses.append(total_loss.item())
            # scheduler.step()

    return losses, m, lv


def train_seq(model: VJF,
          y: List,
          u: List,
          *,
          max_iter: int = 200,
          beta: float = 0.,
          verbose: bool = False,
          rtol: float = 1e-5,
          lr: float = 1e-3,
          lr_decay: float = .99):
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    ylist = y
    ulist = u
    
    losses = []
    with trange(max_iter) as progress:
        running_loss = torch.tensor(float('nan'))
        for i in progress:
            # collections
            epoch_loss = 0.
            m_list = []
            logvar_list = []
            for y, u in zip(ylist, ulist):
                y = torch.as_tensor(y, dtype=torch.get_default_dtype())
                y = torch.atleast_2d(y)
                y = y.unsqueeze(0)

                u = torch.as_tensor(u, dtype=torch.get_default_dtype())
                u = torch.atleast_2d(u)
                u = u.unsqueeze(0)

                N, L, _ = y.shape  # 3D, time first

                yhat, m, lv, x0, x1, m1 = model.forward(y, u)
                total_loss, loss_recon, loss_dynamics, h = model.loss(y, yhat, m, lv, x1, m1, components=True, warm_up=False)
            
                # kl_scale = torch.sigmoid(torch.tensor(i, dtype=torch.get_default_dtype()) - 10)
                kl_scale = 1.
                total_loss = loss_recon + kl_scale * (loss_dynamics - h)

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                optimizer.step()
            
                epoch_loss += total_loss
                
                m_list.append(m)
                logvar_list.append(lv)

            if hasattr(model.transition, 'train'):
                # print(m_list[0].shape)
                model.transition.train(m_list, ulist)
                # model.transition.update(x2d, u2d)
            
            if epoch_loss.isclose(running_loss, rtol=rtol):
                print('\nConverged.\n')
                break

            running_loss = beta * running_loss + (1 - beta) * epoch_loss if i > 0 else epoch_loss
            # print(total_loss)

            progress.set_postfix({
                'Loss': running_loss.item(),
                # 'KL scale': kl_scale.item(),
            })
            losses.append(epoch_loss.item())

            # scheduler.step()

    return losses, m_list, logvar_list
