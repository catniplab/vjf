"""
input-driven system
batch training
non-Bayesian state model
"""
from abc import ABCMeta, abstractmethod
from typing import Tuple, Sequence, Union, List

import torch
from torch import Tensor, nn
from torch.nn import Module, Linear, Parameter, GRU
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from .distribution import Gaussian
from .functional import gaussian_entropy as entropy, gaussian_loss, normed_linear
from .likelihood import GaussianLikelihood, PoissonLikelihood
from .module import RBFN
from .util import reparametrize, symmetric, running_var, nonecat, flat2d


class LinearDecoder(Module):
    def __init__(self, xdim: int, ydim: int, norm='fro'):
        super().__init__()
        # self.add_module('decode', Linear(xdim, ydim))
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
        if self.norm != 'none':
            if self.norm == 'fro':
                normalizer = torch.linalg.norm(w, keepdim=True)
            elif self.norm == 'row':
                normalizer = torch.linalg.norm(w, dim=0, keepdim=True)
            elif self.norm == 'col':
                normalizer = torch.linalg.norm(w, dim=1, keepdim=True)
            
            w = w / normalizer
        return functional.linear(x, w, self.bias)


class GRUEncoder(Module):
    def __init__(self, ydim: int, xdim: int, udim: int, hidden_size: int, batch_first: bool = True):
        super().__init__()

        self.add_module(
            'gru',
            GRU(input_size=ydim + udim,
                hidden_size=hidden_size,
                batch_first=batch_first,
                bidirectional=True))
        D = 2  # bidirectional
        self.register_parameter('h0', Parameter(torch.zeros(D, hidden_size)))  # (D, H)

        self.add_module('hidden2posterior', Linear(hidden_size * D, xdim * 2, bias=True))  # q_t
        self.add_module('hidden2initial', Linear(hidden_size * D, xdim * 2, bias=True))  # q_0

    def forward(self, y: Tensor, u: Tensor) -> Gaussian:
        y = torch.cat([y, u], dim=-1)  # append input
        if y.ndim == 2:
            y = y.unsqueeze(1)
        N, L, ydim = y.shape
        h0 = self.h0.unsqueeze(1).expand(-1, N, -1)  # expand batch axis, (D, N, H)
        h_t, h_n = self.gru(y, h0)  # (N, L, Y), (D, N, H) -> (N, L, D*H), (D, N, H)
        
        h_n = torch.swapaxes(h_n, 0, 1)  # (D, N, H) -> (N, D, H)
        h_n = h_n.reshape(N, -1).unsqueeze(1)  # (D, N, H) -> (N, D*H) -> (N, 1, D*H)
        
        o_t = self.hidden2posterior(h_t)  # (N, L, D*H) -> (N, L, 2X)
        o_0 = self.hidden2initial(h_n)  # (N, 1, D*H) -> (N, 1, 2X)

        o = torch.concat([o_0, o_t], dim=1)  # (N, L + 1, 2X)

        mean, logvar = o.chunk(2, -1)  # (N, L + 1, X), (N, L + 1, X)
        return mean, logvar


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
        self.add_module('decode', LinearDecoder(xdim, ydim))

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

    @classmethod
    def make_model(cls,
                   ydim: int,
                   xdim: int,
                   udim: int,
                   n_rbf: int,
                   rbfn_bias: bool,
                   hidden_sizes: Sequence[int],
                   likelihood: str = 'poisson',
                   ds: str = 'rbf',
                   state_logvar: float = 0.,
                   *args,
                   **kwargs):
        if likelihood.lower() == 'poisson':
            likelihood = PoissonLikelihood()
        elif likelihood.lower() == 'gaussian':
            likelihood = GaussianLikelihood()
        
        model = VJF(ydim, xdim, udim, likelihood, RBFDS(xdim, udim, n_rbf, rbfn_bias, state_logvar),
                        GRUEncoder(ydim, xdim, udim, hidden_sizes), *args,
                        **kwargs)
        return model

    def forecast(self, x0: Tensor, u: Tensor, n_step: int = 1, *, noise: bool = False) -> Tuple[Tensor, Tensor]:
        x = self.transition.forecast(x0, u, n_step, noise=noise)
        y = self.decode(x)
        return x, y


class Transition(Module, metaclass=ABCMeta):
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


class RBFDS(Transition):
    def __init__(self, xdim: int, udim: int, n_basis: int, bias=True, logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__()
        self.add_module('predict', RBFN(in_features=xdim + udim, out_features=xdim, n_basis=n_basis, bias=bias))
        self.register_parameter('logvar',
                                Parameter(torch.tensor(logvar),
                                          requires_grad=False))  # state noise
    
    def velocity(self, x, u):
        x = torch.cat([x, u], dim=-1)
        return self.predict(x)

    def forward(self,
                x: Tensor,
                u: Tensor,
                leak: float = 0.) -> Tensor:
        dx = self.velocity(x, u)
        return (1 - leak) * x + dx


class RFFDS(Transition):
    def __init__(self, xdim: int, udim: int, n_basis: int, bias=True, logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        param n_basis: number of radial basis functions
        """
        super().__init__()
        self.add_module('predict', RFF(in_features=xdim + udim, out_features=xdim, n_basis=n_basis))
        self.register_parameter('logvar',
                                Parameter(torch.tensor(logvar),
                                          requires_grad=False))  # state noise
    
    def velocity(self, x, u):
        x = torch.cat([x, u], dim=-1)
        return self.predict(x) - x

    def forward(self,
                x: Tensor,
                u: Tensor,
                leak: float = 0.) -> Tensor:
        x = torch.cat([x, u], dim=-1)
        return self.predict(x)
        # dx = self.velocity(x, u)
        # return (1 - leak) * x + dx
class GRUDS(Transition):
    def __init__(self, xdim: int, udim: int, bias=True, logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        """
        super().__init__()
        self.add_module('predict', GRUCell(input_size=udim, hidden_size=xdim, bias=bias))
        self.add_module('linear', nn.Linear(xdim, xdim))
        # self.add_module('input_dropout', nn.Dropout(0.1))
        # self.add_module('output_dropout', nn.Dropout(0.1))
        self.register_parameter('logvar',
                                Parameter(torch.tensor(logvar),
                                          requires_grad=False))  # state noise
    
    def velocity(self, x, u):
        return self.linear(self.predict(u, x)) - x

    def forward(self,
                x: Tensor,
                u: Tensor,
                leak: float = 0.) -> Tensor:
        # u = self.input_dropout(u)
        return self.linear(self.predict(u, x))
class RNNDS(Transition):
    def __init__(self, xdim: int, udim: int, bias=True, logvar=0.):
        """
        param xdim: state dimensionality
        param udim: input dimensionality
        """
        super().__init__()
        self.add_module('predict', RNNCell(input_size=udim, hidden_size=xdim, bias=bias))
        self.add_module('linear', nn.Linear(xdim, xdim))
        # self.add_module('input_dropout', nn.Dropout(0.1))
        # self.add_module('output_dropout', nn.Dropout(0.1))
        self.register_parameter('logvar',
                                Parameter(torch.tensor(logvar),
                                          requires_grad=False))  # state noise
    
    def velocity(self, x, u):
        return self.linear(self.predict(u, x)) - x

    def forward(self,
                x: Tensor,
                u: Tensor,
                leak: float = 0.) -> Tensor:
        # u = self.input_dropout(u)
        return self.linear(self.predict(u, x))

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

            # scheduler.step()

    return m, lv


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
            
            if epoch_loss.isclose(running_loss, rtol=rtol):
                print('\nConverged.\n')
                break

            running_loss = beta * running_loss + (1 - beta) * epoch_loss if i > 0 else epoch_loss
            # print(total_loss)

            progress.set_postfix({
                'Loss': running_loss.item(),
                # 'KL scale': kl_scale.item(),
            })

            # scheduler.step()

    return m_list, logvar_list
