import logging
import math
from itertools import zip_longest
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import Linear, Module, Parameter
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from .distribution import Gaussian
from .functional import gaussian_entropy as entropy
from .functional import gaussian_loss
from .likelihood import GaussianLikelihood, PoissonLikelihood
from .module import RBF, LinearRegression
from .recognition import Recognition
from .util import nonecat, reparametrize, running_var, symmetric


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


def detach(q: Gaussian) -> Gaussian:
    mean, logvar = q
    return Gaussian(mean.detach(), logvar.detach())


class VJF(Module):
    def __init__(self, ydim: int, xdim: int, likelihood: Module, transition: Module, recognition: Module,
                 *, lr: float = 1e-4, lr_decay: float = .9, optimizer: str = 'sgd'):
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
        self.encoder = 'spikes'  # 'projection' (set by make_model) requires a y_enc feature

        self.register_parameter('mean', Parameter(torch.zeros(xdim)))
        self.register_parameter('logvar', Parameter(torch.zeros(xdim)))

        # gradient optimizer for the ELBO step. 'sgd' (default) is the current behavior;
        # 'adam' restores the optimizer used by the original VJF (Zhao & Park 2020).
        opt_cls = {'sgd': SGD, 'adam': Adam}.get(optimizer.lower())
        if opt_cls is None:
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer!r}")
        self.optimizer = opt_cls(
            [
                {'params': self.likelihood.parameters(), 'lr': lr},
                {'params': self.decoder.parameters(), 'lr': lr},
                {'params': self.transition.parameters(), 'lr': lr},
                {'params': self.recognition.parameters(), 'lr': lr},
            ],
            lr=lr,
        )
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)
        self._opt_n_grown = 0        # tracks transition basis growth to refresh the optimizer (sgd)
        # Denoising-stabilization noise added to the flow input for the DYNAMICS ELBO term
        # only (pt; it is used nowhere else). Trains the flow to contract off-cycle states
        # back to the trajectory -> a stable attractor. 0 = original ELBO (off by default).
        # Schedule: a DECAYING RAISED SINUSOID over filter steps t --
        #   sigma(t) = dyn_noise * dyn_noise_decay**(t/P) * 0.5*(1 - cos(2*pi*t/P)),  P = period
        # i.e. a train of noise bumps (one per period) whose peak fades each period.
        self.dyn_noise = 0.0          # peak std sigma_0 (0 = off)
        self.dyn_noise_period = 1000  # steps per bump
        self.dyn_noise_decay = 1.0    # per-period envelope decay (1 = undecayed sinusoid)
        # Fit-gate: scale the injected noise by exp(-resid_var / fit_ref**2), where
        # resid_var = exp(transition.logvar) is the running one-step dynamics residual.
        # While the flow underfits (large residual) the gate -> 0, so we do not perturb a
        # flow that cannot yet predict the clean transition. 0 = no gating (off).
        self.dyn_noise_fit_ref = 0.0
        self._dyn_step = 0            # schedule step counter
        # R2 curvature regularizer on the SGD-flow one-step map (0 = original ELBO, off).
        # Adds lambda_smooth * mean_t ||d^2 v/dx dx^T||_F^2 to the SGD-flow loss; see
        # RBFDS.curvature_penalty + IMPL.md (2026-06-15). sgd flow only.
        self.smooth_lambda = 0.0

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

    def forward(self, y: Tensor, qs: Gaussian, u: Tensor = None, y_enc: Tensor = None) -> Tuple:
        """
        :param y: new observation (used by the decoder/likelihood)
        :param qs: posterior before new observation
        :param u: input, None if autonomous
        :param y_enc: optional separate input for the recognition network (e.g. a
            subspace projection of y). Defaults to y (the original behavior).
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
        x_dyn = xs                                       # flow input for the dynamics term
        if self.dyn_noise > 0:                           # denoising bump (decaying raised sinusoid)
            t, P = self._dyn_step, self.dyn_noise_period
            sigma = (self.dyn_noise * self.dyn_noise_decay ** (t / P)
                     * 0.5 * (1.0 - math.cos(2.0 * math.pi * t / P)))
            if self.dyn_noise_fit_ref > 0:               # suppress noise while the flow underfits
                resid_var = float(torch.exp(self.transition.logvar))
                sigma *= math.exp(-resid_var / (self.dyn_noise_fit_ref ** 2))
            x_dyn = xs + sigma * torch.randn_like(xs)    # perturb only pt's input, not xs (update uses xs)
            self._dyn_step += 1
        pt = self.transition(x_dyn, u, sampling=False)

        y = torch.atleast_2d(y)
        if y_enc is None:
            if getattr(self, 'encoder', 'spikes') == 'projection':
                raise ValueError(
                    "encoder='projection' needs a recognition feature: call "
                    "filter(..., y_enc=...) with a subspace projection (e.g. from "
                    "vjf.readout.OnlineReadout). The raw-y path (VJF.fit / y_enc=None) "
                    "is unsupported for this encoder."
                )
            y_enc = y
        else:
            y_enc = torch.atleast_2d(y_enc)
        qt = self.recognition(y_enc, qs, u)

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

        # assert torch.isfinite(l_recon), l_recon.item()
        # assert torch.isfinite(l_dynamics), l_dynamics.item()
        # assert torch.isfinite(h), h.item()

        if not torch.isfinite(l_recon):
            l_recon = torch.tensor(0.)
        
        if not torch.isfinite(l_dynamics):
            l_dynamics = torch.tensor(0.)

        if not torch.isfinite(h):
            h = torch.tensor(0.)

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

    def _repoint_transition_params(self):
        """Point the optimizer's transition group (index 2: likelihood, decoder,
        transition, recognition) at the current parameters after grow_basis replaced the
        sgd flow-weight Parameter with a larger one. Per-parameter optimizer state (Adam
        exp_avg / exp_avg_sq) is migrated onto the enlarged tensor, zero-padding the grown
        rows, so Adam does not restart its moments for the whole flow on every addition.
        SGD keeps no such state, so this is then just a re-point."""
        grp = self.optimizer.param_groups[2]
        old_params = grp['params']
        new_params = list(self.transition.parameters())
        added = [p for p in new_params if id(p) not in {id(q) for q in old_params}]
        removed = [p for p in old_params if id(p) not in {id(q) for q in new_params}]
        for newp in added:                             # match the regrown weight to its predecessor
            match = next((op for op in removed if op.dim() == newp.dim()
                          and op.shape[1:] == newp.shape[1:] and op.shape[0] <= newp.shape[0]), None)
            if match is not None and match in self.optimizer.state:
                st = self.optimizer.state.pop(match)
                mig = {}
                for k, v in st.items():
                    if torch.is_tensor(v) and v.shape == match.shape:
                        pad = torch.zeros_like(newp)
                        pad[:v.shape[0]] = v
                        mig[k] = pad
                    else:                              # e.g. the scalar step count -> keep as is
                        mig[k] = v
                self.optimizer.state[newp] = mig
        grp['params'] = new_params

    def filter(self, y: Tensor, u: Tensor = None, qs: Gaussian = None, *,
               sgd: bool = True, update: bool = True, verbose: bool = False, warm_up: bool = False,
               y_enc: Tensor = None):
        """
        Filter a step or a sequence
        :param y: observation, assumed axis order (time, batch, dim). missing axis will be prepended.
        :param u: control
        :param qs: previos posterior. use prior if None, otherwise detached.
        :param sgd: flag to enable gradient step
        :param update: flag to update DS
        :param verbose: verbose output
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

        if y_enc is not None:
            y_enc = torch.as_tensor(y_enc, dtype=torch.get_default_dtype())
        xs, pt, qt, xt, py = self.forward(y, qs, u, y_enc=y_enc)
        output = self.loss(y, xs, pt, qt, xt, py, components=verbose, warm_up=warm_up)
        if verbose:
            loss, *elbos = output
        else:
            loss = output
        if sgd:
            # R2 curvature penalty on the SGD-flow field, added only on the learning step
            # (not during frozen eval, to avoid its per-bin cost) and only when active.
            if (not warm_up and self.smooth_lambda > 0
                    and self.transition.flow_learner == 'sgd'):
                loss = loss + self.smooth_lambda * self.transition.curvature_penalty(xs)
            try:
                self.optimizer.zero_grad()
                loss.backward()  # accumulate grad if not trained
                nn.utils.clip_grad_value_(self.parameters(), 1.)
                self.optimizer.step()
            except RuntimeError as err:
                logging.warning(err)
                self.optimizer.zero_grad()
        if update:
            self.update(y, xs, u, pt, qt, xt, py, warm_up=warm_up)  # non-gradient step
            # grow_basis (sgd path) swaps the flow-weight Parameter for a larger one;
            # re-point the optimizer at it (migrating any Adam moments) so the new weight
            # is trained without restarting the optimizer state for the whole flow.
            if (self.transition.flow_learner == 'sgd'
                    and getattr(self.transition, '_n_grown', 0) != self._opt_n_grown):
                self._repoint_transition_params()
                self._opt_n_grown = self.transition._n_grown

        if verbose:
            return qt, loss, *elbos
        else:
            return qt, loss

    def fit(self, y: Tensor, u: Tensor = None, *,
            max_iter: int = 200, beta: float = 0.1, verbose: bool = False, rtol: float = 1e-4):
        """
        :param y: observation, (time, ..., dim)
        :param u: control input, None if
        :param max_iter: maximum number of epochs
        :param beta: discounting factor for running loss, large weight on current epoch loss for small value
        :param verbose: verbose output
        :param rtol: relative tolerance for convergence detection
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
        epoch_loss = torch.tensor(float('nan'))
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
                        self.decoder.requires_grad_(False)  # freeze decoder after warm up
                        m = torch.stack([q.mean for q in q_seq])
                        if isinstance(u_, Tensor) and u_.shape[-1] > 0:
                            u_init = u_[1:, :].reshape(-1, u_.shape[-1])
                        else:
                            u_init = None
                        self.transition.initialize(m[1:].reshape(-1, m.shape[-1]),
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

                self.scheduler.step()

        mu = torch.stack([q.mean for q in q_seq])
        logvar = torch.stack([q.logvar for q in q_seq])
        return mu, logvar, epoch_loss

    @classmethod
    def make_model(cls, ydim: int, xdim: int, udim: int, n_rbf: int, hidden_sizes: Sequence[int],
                   likelihood: str = 'poisson', *args, transition_flow: str = 'rls',
                   encoder: str = 'spikes', **kwargs):
        if likelihood.lower() == 'poisson':
            likelihood = PoissonLikelihood()
        elif likelihood.lower() == 'gaussian':
            likelihood = GaussianLikelihood()

        # encoder='spikes' (default): recognition reads the ydim-dim observation.
        # encoder='projection': recognition reads an xdim-dim feature (e.g. a subspace
        # projection of the observation) passed via filter(..., y_enc=...).
        if encoder not in ('spikes', 'projection'):
            raise ValueError(f"encoder must be 'spikes' or 'projection', got {encoder!r}")
        rec_in = xdim if encoder == 'projection' else ydim

        model = VJF(ydim, xdim, likelihood, RBFDS(n_rbf, xdim, udim, flow_learner=transition_flow),
                    Recognition(rec_in, xdim, udim, hidden_sizes), *args, **kwargs)
        model.encoder = encoder  # gates the y_enc requirement in forward()
        return model

    def forecast(self, x0: Tensor, u: Tensor = None, n_step: int = 1, *, noise: bool = False) -> Tuple[Tensor, Tensor]:
        x = self.transition.forecast(x0, u, n_step, noise=noise)
        y = self.decoder(x)
        return x, y


class RBFDS(Module):
    def __init__(self, n_rbf: int, xdim: int, udim: int, flow_learner: str = 'rls'):
        super().__init__()
        # How the velocity (flow) weights are learned:
        #   'sgd' - weights are an nn.Parameter trained by SGD/Adam via the dynamics
        #           ELBO term (gradient-clipped in VJF.filter). This is the ORIGINAL
        #           VJF (Zhao & Park 2020): every parameter is learned by stochastic
        #           gradient. Stable but slow to converge online.
        #   'rls' - online recursive least squares (a real-time deviation, not in the
        #           original VJF; fast convergence because W enters linearly, but the
        #           precision matrix grows/ill-conditions over very long streams and
        #           the weights can explode).
        #   'srrls' - square-root (Potter) RLS: same fast convergence as 'rls' but
        #           propagates the covariance Cholesky factor directly (PD by
        #           construction, no precision accumulation/inversion), so it is
        #           numerically stable over very long streams. Preferred for sVJF.
        if flow_learner not in ('rls', 'srrls', 'sgd'):
            raise ValueError(f"flow_learner must be 'rls', 'srrls' or 'sgd', got {flow_learner!r}")
        self.flow_learner = flow_learner
        self.add_module('velocity', LinearRegression(RBF(xdim + udim, n_rbf), xdim,
                                                     bayes=(flow_learner != 'sgd')))
        self.register_parameter('logvar', Parameter(torch.tensor(0.), requires_grad=False))  # state noise
        self.n_sample = 0  # sample counter
        # RLS forgetting + ridge ('rls' flow only). Defaults reproduce the original
        # (shrink=1, ridge=0). Set shrink<1 with ridge>0 to bound the precision.
        self.rls_shrink = 1.0
        self.rls_ridge = 0.0
        # Growing RBF basis ('srrls' flow only; off by default = original behavior).
        # When grow_rbf, after each online update the current predictor's RBF coverage
        # is checked: if the max basis activation < grow_thresh (all existing centers
        # are far), a new center is appended there (LinearRegression.grow_basis). This
        # lets the flow basis track a latent that drifts/inflates during learning.
        self.grow_rbf = False
        self.max_rbf = None          # cap on the number of centers (None = uncapped)
        self.grow_thresh = 0.5       # add a center if max RBF activation falls below this
        self.grow_min_gap = 1        # min update steps between additions
        self.grow_logwidth = None    # new-center logwidth (None = median of existing)
        self.grow_p0 = 1.0           # prior covariance for a new weight
        # new-weight init: 'zero' (srls fills it fast) or 'residual' (RAN-style: weight =
        # current flow error at the new center, so the slow sgd gradient need not fill it).
        self.grow_weight_init = 'zero'
        self._since_grow = 0
        self._n_grown = 0

    def _velocity_mean(self, xu: Tensor) -> Tensor:
        out = self.velocity(xu, sampling=False)
        return out.mean if isinstance(out, Gaussian) else out

    @torch.no_grad()
    def _maybe_grow(self, xu: Tensor, dx: Tensor = None) -> None:
        """Append an RBF center at the least-covered predictor row if it is not yet
        covered by the basis (its max activation < grow_thresh = all existing centers
        far). Throttled by grow_min_gap and bounded by max_rbf. The novelty test
        follows Memming's criterion: grow when the RBF-projected state's activation is
        small. Coverage is per ROW so a batched update (xu has batch>1) grows for its
        most-novel row rather than being blocked by a single covered row.

        ``dx`` (target velocity) enables the 'residual' weight init: the new center,
        whose activation is ~1 at its own location, takes the current flow error there
        so it corrects the local velocity immediately (needed for the slow sgd flow)."""
        self._since_grow += 1
        cap = self.max_rbf if self.max_rbf is not None else float('inf')
        if self.velocity.feature.n_basis >= cap or self._since_grow < self.grow_min_gap:
            return
        phi = self.velocity.feature(xu)                  # (batch, n_basis)
        cover = phi.max(dim=1).values                    # per-row max activation (batch,)
        j = int(cover.argmin())                          # least-covered (most novel) row
        if float(cover[j]) < self.grow_thresh:           # uncovered -> add a center at that row
            lw = (self.grow_logwidth if self.grow_logwidth is not None
                  else float(self.velocity.feature.logwidth.median()))
            w_new = None
            if self.grow_weight_init == 'residual' and dx is not None:
                w_new = dx[j:j + 1] - self._velocity_mean(xu[j:j + 1])   # local flow error
            self.velocity.grow_basis(xu[j:j + 1], lw, p0=self.grow_p0, weight=w_new)
            self._since_grow = 0
            self._n_grown += 1

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
            if self.flow_learner == 'rls':
                self.velocity.rls(xu, dx, self.logvar.exp(), shrink=self.rls_shrink, ridge=self.rls_ridge)  # model dx
            elif self.flow_learner == 'srrls':
                self.velocity.srls(xu, dx, self.logvar.exp(), shrink=self.rls_shrink)
            # 'sgd' flow weights are trained by the main optimizer via the dynamics ELBO.
            # The growing basis applies to srrls AND sgd ('rls' precision would degrade).
            if self.grow_rbf and self.flow_learner in ('srrls', 'sgd'):
                self._maybe_grow(xu, dx)
        residual = dx - self._velocity_mean(xu)
        mse = residual.pow(2).mean()
        var, n_sample = running_var(self.logvar.exp(), self.n_sample, mse, xs.shape[0], size_cap=500)
        self.logvar.data = var.log()
        self.n_sample = n_sample

    @torch.no_grad()
    def initialize(self, xt: Tensor, xs: Tensor, ut: Tensor = None, *,
                   rbf_centers: Tensor = None, rbf_logwidths: Tensor = None):
        xs = torch.atleast_2d(xs)
        xt = torch.atleast_2d(xt)
        xu = nonecat(xs, ut)
        mse = (xt - xs).pow(2).mean()
        if self.flow_learner == 'srrls':
            self.velocity.init_srls(xu, xt - xs, centers=rbf_centers, logwidths=rbf_logwidths)
        else:
            # rls/sgd: LinearRegression.initialize accepts the data-driven centers too
            self.velocity.initialize(xu, xt - xs, mse, centers=rbf_centers, logwidths=rbf_logwidths)
        d = self._velocity_mean(xu)
        mse = (xt - xs - d).pow(2).mean()
        self.logvar.data = mse.log()

    def loss(self, pt: Tensor, qt: Tensor) -> Tensor:
        return gaussian_loss(pt, qt, self.logvar)

    def curvature_penalty(self, x: Tensor) -> Tensor:
        """R2 smoothness regularizer: mean squared curvature ||d^2 v / dx dx^T||_F^2 of the
        SGD-flow velocity field at the states x (B, d), in closed form for the Gaussian RBF.
        For phi_j(x) = exp(-||x-c_j||^2 / (2 w_j^2)):
            d^2 phi_j/dx dx^T = phi_j [ (x-c_j)(x-c_j)^T / w_j^4 - I / w_j^2 ]
            H_a(x) = d^2 v_a/dx dx^T = sum_j W_{j,a} d^2 phi_j/dx dx^T
            R2(x) = sum_a ||H_a(x)||_F^2
        Penalizes only nonlinear curvature, so an affine/rotational field (a limit cycle) is
        unpenalized while high-frequency wiggle is suppressed. Added to the SGD-flow ELBO loss
        as lambda_smooth * mean_t R2(x_t) (see VJF.filter, IMPL.md 2026-06-15). x is detached:
        the penalty shapes the field (gradient -> flow weights), not the state. sgd flow only
        (udim=0: the RBF input is x). The ELBO terms themselves are unchanged."""
        feat = self.velocity.feature
        c = feat.centroid                                       # (M, d)
        x = torch.atleast_2d(x).detach()                        # (B, d)
        if c.shape[1] != x.shape[-1]:                           # RBF built over [x,u]; penalty is state-only
            raise NotImplementedError("curvature_penalty requires udim=0 (RBF input == state x)")
        w2 = feat.logwidth.exp().pow(2)                         # (M,)
        W = self.velocity.w_mean                                # (M, d_out)
        phi = feat(x)                                           # (B, M)
        diff = x[:, None, :] - c[None, :, :]                    # (B, M, d)
        eye = torch.eye(c.shape[1], dtype=x.dtype, device=x.device)
        d2phi = (diff[..., :, None] * diff[..., None, :] / w2[None, :, None, None].pow(2)
                 - eye[None, None] / w2[None, :, None, None]) * phi[..., None, None]  # (B,M,d,d)
        H = torch.einsum('ja,bjpq->bapq', W, d2phi)             # (B, d_out, d, d)
        return H.pow(2).sum(dim=(1, 2, 3)).mean()
