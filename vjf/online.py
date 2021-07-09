import logging

import numpy as np
import torch
from tqdm import trange

from . import observation, decoder, dynamics, recognition, metric
from .base import Model
from .metric import gaussian_entropy

__all__ = ["VJF"]
logger = logging.getLogger(__name__)


class VJF(Model):
    def __init__(self, config):
        super().__init__(config)

        self.ydim = self.config["ydim"]
        self.xdim = self.config["xdim"]
        self.udim = self.config["udim"]

        self.add_module(
            "likelihood", observation.Likelihood.get_likelihood(self.config)
        )

        self.add_module(
            "decoder", decoder.GLMDecoder(self.likelihood, None, self.config)
        )
        self.add_module(
            "state_noise", dynamics.GaussianNoise(self.xdim, *self.config["Q"])
        )
        self.add_module(
            "system", dynamics.System.get_system(self.config, self.state_noise)
        )
        self.add_module(
            "recognizer", recognition.Recognizer.get_recognizer(self.config, system=self.system)
        )
        # self.add_module("smoother", recognition.Recognizer.get_recognizer(self.config))

        self.trainable_variables = list(
            filter(lambda p: p.requires_grad, self.parameters())
        )

        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=self.config["lr"], amsgrad=False
        )

        self.noise_optimizer = torch.optim.Adam(
            self.state_noise.parameters(), lr=self.config["lr"], amsgrad=False
        )

        self.dynamics_optimizer = self.system.optimizer
        for group in self.dynamics_optimizer.param_groups:
            group["lr"] = self.config["lr"]

        self.encoder_optimizer = torch.optim.Adam(
            self.recognizer.parameters(), lr=self.config["lr"], amsgrad=False
        )

        # self.smoother_optimizer = torch.optim.Adam(
        #     self.smoother.parameters(), lr=self.config["lr"], amsgrad=False
        # )

    def forward(self):
        pass

    def elbo(self, q0, q, obs, sample, regularize):
        return metric.elbo(
            q0,
            q,
            obs,
            decoder=self.decoder,
            system=self.system,
            sample=sample,
            regularize=regularize,
        )

    def filter(
            self,
            y,
            u,
            q0=None,
            time_major=False,
            decoder=True,
            encoder=True,
            dynamics=True,
            noise=True,
            sample=True,
            regularize=False,
    ):
        ys, us = (
            torch.as_tensor(y, dtype=torch.float),
            torch.as_tensor(u, dtype=torch.float),
        )
        if not time_major:
            ys, us = torch.transpose(ys, 1, 0), torch.transpose(us, 1, 0)

        losses = []
        mu = torch.zeros(ys.shape[0], ys.shape[1], self.xdim)
        logvar = torch.zeros(ys.shape[0], ys.shape[1], self.xdim)

        for i, obs in enumerate(zip(ys, us)):
            # dual = True if i >= delay else False
            q, loss = self.feed(
                obs,
                q0,
                decoder=decoder,
                encoder=encoder,
                dynamics=dynamics,
                noise=noise,
                sample=sample,
                regularize=regularize,
            )
            mu[i, :, :], logvar[i, :, :] = q
            q0 = q
            losses.append(loss)

        if not time_major:
            mu = torch.transpose(mu, 1, 0)
            logvar = torch.transpose(logvar, 1, 0)
        return mu, logvar, losses

    def feed(
            self,
            obs,
            q0=None,
            decoder=True,
            encoder=True,
            dynamics=True,
            noise=True,
            sample=True,
            regularize=False,
    ):
        y, u = obs
        batch = y.shape[0]

        if q0 is None:
            mu0 = torch.zeros(batch, self.xdim)
            logvar0 = torch.zeros(batch, self.xdim)
        else:
            mu0, logvar0 = q0

        # mu0, logvar0 = self.smoother(y, u, (mu0, logvar0))

        mu1, logvar1 = self.recognizer(y, u, (mu0, logvar0))

        ll_recon, ll_dyn, entropy = self.elbo(
            (mu0, logvar0), (mu1, logvar1), (y, u), sample, regularize
        )
        cost = torch.neg(ll_recon + ll_dyn + entropy)  # + torch.sum(
        # torch.exp(self.system.noise.logvar)
        # ) * torch.exp(self.logdecay)

        self.decoder_optimizer.zero_grad()
        self.dynamics_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.noise_optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_value_(
            self.parameters(), self.config["clip_gradients"]
        )
        if decoder:
            self.decoder_optimizer.step()
        if dynamics:
            self.dynamics_optimizer.step()
        if encoder:
            self.encoder_optimizer.step()
        if noise:
            self.noise_optimizer.step()

        mu1, logvar1 = self.recognizer(y, u, (mu0.detach(), logvar0.detach()))
        # self.system.fit(mu0.detach(), mu1.detach())

        return (mu1, logvar1), (ll_recon, ll_dyn, entropy)

    def set_props(self):
        self.config.setdefault("R", 1.0)
        self.config.setdefault("Q", 1.0)
        self.config.setdefault("likelihood", "poisson")
        self.config.setdefault("recognizer", "mlp")
        self.config.setdefault("system", "linear")
        self.config.setdefault("clip_gradients", None)
        self.config.setdefault("activation", "tanh")
        self.config.setdefault("batch_norm", False)
        self.config.setdefault("optimizer", "adam")

    def preprocess(self, y, u, mu, logvar, time_major=False, center=True):
        ys, us, mus, logvars = (
            torch.as_tensor(y, dtype=torch.float),
            torch.as_tensor(u, dtype=torch.float),
            torch.as_tensor(mu, dtype=torch.float),
            torch.as_tensor(logvar, dtype=torch.float),
        )

        if not time_major:
            ys, us, mus, logvars = (
                torch.transpose(ys, 1, 0),
                torch.transpose(us, 1, 0),
                torch.transpose(mus, 1, 0),
                torch.transpose(logvars, 1, 0),
            )

        if center:
            z = mus.reshape(-1, self.xdim)
            m = torch.mean(z, dim=0, keepdim=True)
            mus.sub_(m)

        T, M, _ = us.shape
        if mus.shape[0] == us.shape[0]:
            mus = torch.cat([torch.zeros(1, M, self.xdim), mus], dim=0)
            logvars = torch.cat([torch.zeros(1, M, self.xdim), logvars], dim=0)

        return ys, us, mus, logvars

    # def train_dynamics(
    #     self, y, u, mu, logvar, time_major=False, n_iter=1000, sample=False, rtol=1e-4
    # ):
    #     y, u, mu, logvar = self.preprocess(y, u, mu, logvar, time_major)
    #
    #     y = torch.reshape(y, (-1, self.ydim))
    #     u = torch.reshape(u, (-1, self.udim))
    #     q0 = (
    #         torch.reshape(mu[:-1, :, :], (-1, self.xdim)),
    #         torch.reshape(logvar[:-1, :, :], (-1, self.xdim)),
    #     )
    #     q1 = (
    #         torch.reshape(mu[1:, :, :], (-1, self.xdim)),
    #         torch.reshape(logvar[1:, :, :], (-1, self.xdim)),
    #     )
    #
    #     old_loss = 0.0
    #     for i in range(n_iter):
    #         loss = self.system.loss(q0, q1, u, sample=sample)
    #
    #         if abs((loss - old_loss) / loss) < rtol:
    #             break
    #         old_loss = loss
    #
    #         self.dynamics_optimizer.zero_grad()
    #         loss.backward()
    #         self.dynamics_optimizer.step()
    #
    # def mstep(
    #     self,
    #     y,
    #     u,
    #     mu,
    #     logvar,
    #     time_major=False,
    #     n_iter=1000,
    #     sample=False,
    #     rtol=1e-4,
    #     decoder=True,
    #     dynamics=True,
    #     regularize=True,
    # ):
    #     y, u, mu, logvar = self.preprocess(y, u, mu, logvar, time_major)
    #
    #     y = torch.reshape(y, (-1, self.ydim))
    #     u = torch.reshape(u, (-1, self.udim))
    #     q0 = (
    #         torch.reshape(mu[:-1, :, :], (-1, self.xdim)),
    #         torch.reshape(logvar[:-1, :, :], (-1, self.xdim)),
    #     )
    #     q1 = (
    #         torch.reshape(mu[1:, :, :], (-1, self.xdim)),
    #         torch.reshape(logvar[1:, :, :], (-1, self.xdim)),
    #     )
    #
    #     old_loss = 0.0
    #     for i in range(n_iter):
    #         loss = 0.0
    #         if decoder:
    #             loss += self.decoder.loss(q1, y, norm=False)
    #
    #         loss += self.system.loss(q0, q1, u, sample=sample, regularize=regularize)
    #
    #         if abs((loss - old_loss) / loss) < rtol:
    #             break
    #         old_loss = loss
    #
    #         self.decoder_optimizer.zero_grad()
    #         self.dynamics_optimizer.zero_grad()
    #         self.noise_optimizer.zero_grad()
    #         loss.backward()
    #         self.decoder_optimizer.step()
    #         if dynamics:
    #             self.dynamics_optimizer.step()
    #         else:
    #             self.noise_optimizer.step()
    #
    #     return loss
    #
    # def estep(self, y, u, mu, logvar, time_major=False, n_iter=1000, rtol=1e-4):
    #     if time_major:
    #         T, B, _ = y.shape
    #     else:
    #         B, T, _ = y.shape
    #
    #     y, u, mu, logvar = self.preprocess(y, u, mu, logvar, time_major)
    #
    #     y = torch.reshape(y, (-1, self.ydim))
    #     u = torch.reshape(u, (-1, self.udim))
    #
    #     mu.requires_grad_()
    #     logvar.requires_grad_()
    #
    #     state_optimizer = torch.optim.Adam([mu, logvar], lr=1e-3, amsgrad=False)
    #
    #     old_loss = 0.0
    #     for i in range(n_iter):
    #         q0 = (
    #             torch.reshape(mu[:-1, :, :], (-1, self.xdim)),
    #             torch.reshape(logvar[:-1, :, :], (-1, self.xdim)),
    #         )
    #         q1 = (
    #             torch.reshape(mu[1:, :, :], (-1, self.xdim)),
    #             torch.reshape(logvar[1:, :, :], (-1, self.xdim)),
    #         )
    #         # print(y.shape, u.shape, q1[0].shape, q1[1].shape, q0[0].shape, q0[0].shape)
    #         loss = (
    #             self.decoder.loss(q1, y, norm=False)
    #             + self.system.loss(q0, q1, u)
    #             + gaussian_entropy(q1[1])
    #         )
    #
    #         if abs((loss - old_loss) / loss) < rtol:
    #             break
    #         old_loss = loss
    #
    #         state_optimizer.zero_grad()
    #         loss.backward()
    #         # print(mu.grad)
    #         state_optimizer.step()
    #
    #     mu = torch.reshape(mu, (T + 1, B, self.xdim))
    #     logvar = torch.reshape(logvar, (T + 1, B, self.xdim))
    #     mu = mu[1:, :, :]
    #     logvar = logvar[1:, :, :]
    #
    #     if not time_major:
    #         mu = torch.transpose(mu, 1, 0)
    #         logvar = torch.transpose(logvar, 1, 0)
    #     mu = mu.detach().numpy()
    #     logvar = logvar.detach().numpy()
    #
    #     return mu, logvar
    #
    # def em(
    #     self,
    #     y,
    #     u,
    #     mu,
    #     logvar,
    #     time_major=False,
    #     n_iter=20,
    #     m_iter=100,
    #     e_iter=100,
    #     rtol=1e-4,
    #     decoder=True,
    #     dynamics=True,
    # ):
    #     old_loss = 0.0
    #     for k in range(n_iter):
    #         # M step
    #         if m_iter > 0:
    #             loss = self.mstep(
    #                 y,
    #                 u,
    #                 mu,
    #                 logvar,
    #                 time_major,
    #                 m_iter,
    #                 decoder=decoder,
    #                 dynamics=dynamics,
    #             )
    #
    #             if abs((loss - old_loss) / loss) < rtol:
    #                 break
    #             old_loss = loss
    #
    #         # E step
    #         if e_iter > 0:
    #             mu, logvar = self.estep(y, u, mu, logvar, time_major, e_iter)
    #
    #     return mu, logvar

    # def fit(
    #     self,
    #     y,
    #     u,
    #     mu,
    #     logvar,
    #     time_major=False,
    #     n_iter=50,
    #     n_iter1=1000,
    #     decoder=True,
    #     encoder=True,
    #     dynamics=True,
    #     noise=True,
    #     regularize=True,
    # ):
    #     y, u, mu, logvar = self.preprocess(y, u, mu, logvar, time_major)
    #
    #     for i in range(n_iter1):
    #         mu1, logvar1 = self.recognizer(
    #             torch.reshape(y, (-1, self.ydim)),
    #             torch.reshape(u, (-1, self.udim)),
    #             (
    #                 torch.reshape(mu[:-1, :, :], (-1, self.xdim)),
    #                 torch.reshape(logvar[:-1, :, :], (-1, self.xdim)),
    #             ),
    #         )
    #         loss = torch.mean((torch.reshape(mu[1:, :, :], (-1, self.xdim)) - mu1) ** 2)
    #         loss += torch.mean(
    #             (torch.reshape(logvar[1:, :, :], (-1, self.xdim)) - logvar1) ** 2
    #         )
    #         self.encoder_optimizer.zero_grad()
    #         loss.backward()
    #         self.encoder_optimizer.step()
    #
    #     for i in range(n_iter):
    #         mu, logvar, _ = self.filter(
    #             y,
    #             u,
    #             True,
    #             decoder=decoder,
    #             encoder=encoder,
    #             dynamics=dynamics,
    #             noise=noise,
    #             regularize=regularize,
    #         )
    #
    #     if not time_major:
    #         mu = torch.transpose(mu, 1, 0)
    #         logvar = torch.transpose(logvar, 1, 0)
    #
    #     return mu.detach().numpy(), logvar.detach().numpy()

    def fit(self,
            y,
            u,
            q0=None,
            *,
            max_iter=1000,
            time_major=False,
            decoder=True,
            encoder=True,
            dynamics=True,
            noise=True,
            sample=True,
            regularize=False,
            ):

        loss = torch.tensor(np.nan)
        with trange(max_iter) as progress:
            for i in progress:
                mu, logvar, losses = self.filter(y,
                                                 u,
                                                 q0=q0,
                                                 time_major=time_major,
                                                 decoder=decoder,
                                                 encoder=encoder,
                                                 dynamics=dynamics,
                                                 noise=noise,
                                                 sample=sample,
                                                 regularize=regularize,
                                                 )
                new_loss = -torch.tensor(losses).sum()
                progress.set_postfix({'Loss': new_loss.item()})
                if torch.isclose(loss, new_loss):
                    break
                loss = new_loss

        return mu, logvar, loss
