"""Reusable per-sample online-filtering loop for real-time VJF inference.

``VJF.filter(y_t, u, q, ..., y_enc=)`` is already the per-sample online primitive.
What a real-time deployment additionally needs -- threading the posterior across
calls, the warm-up buffer + one-time dynamics initialization, divergence recovery,
the state-noise (logvar) floor, wiring an :class:`~vjf.readout.OnlineReadout`, and
the periodic Procrustes refresh -- is packaged here as a single generator,
:func:`online_filter`, so callers do not re-implement (and subtly mis-implement)
that scaffolding. No new math: every step calls the existing ``VJF``/``OnlineReadout``
methods.

Push-driven (real-time) use: ``online_filter`` consumes any iterable of ``y_t`` and
is itself a generator, so it can be driven one sample at a time -- e.g. wrap a
blocking queue as the iterator::

    def source():
        while True:
            yield q.get()          # blocks until the next sample arrives
    for res in online_filter(model, source(), readout=ro, warmup_steps=W):
        act_on(res.mean)           # one result per arriving sample

Callers who cannot provide an iterator can inline the loop body below.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np
import torch


def _mean(out):
    """Mean of a transition output: Gaussian (RLS/sRLS) -> .mean; Tensor (SGD) -> itself."""
    return out.mean if isinstance(out, tuple) else out


def _as_u(u):
    """Normalize a control sample to a (1, udim) tensor, or None."""
    if u is None:
        return None
    return torch.atleast_2d(torch.as_tensor(u, dtype=torch.get_default_dtype()))


@dataclass
class OnlineResult:
    """Per-sample output of :func:`online_filter` (all arrays are 1-D, length xdim)."""
    step: int
    mean: np.ndarray            # posterior mean of x_t
    logvar: np.ndarray          # posterior log-variance of x_t
    pred_mean: Optional[np.ndarray]  # one-step prediction E[x_t | x_{t-1}] (None at t=0)
    loss: float                 # negative ELBO this step (nan if diverged)
    recon: float
    dynamics: float
    entropy: float
    warming_up: bool            # dynamics frozen (t < warmup_steps)
    diverged: bool              # guard tripped; mean repeats the previous good mean
    refreshed: bool             # projection (C,b) Procrustes refresh fired this step
    elapsed_s: float            # wall time of the online inference region (excl. diagnostics)
    trial: int = -1            # trial index (real-time multi-trial use)
    t_in_trial: int = -1       # step within the trial


def online_filter(model, stream: Iterable, *, readout=None, u_stream: Optional[Iterable] = None,
                  adapt_readout: bool = True, warmup_steps: int = 1000, rbf_width_scale: float = 1.0,
                  logvar_floor: float = math.log(1e-6), max_abs_state: float = 1e3
                  ) -> Iterator[OnlineResult]:
    """Drive ``model`` over a per-sample observation ``stream``, yielding one
    :class:`OnlineResult` per sample.

    :param model: a :class:`vjf.model.VJF` (any ``transition_flow``/``encoder``).
    :param stream: iterable of observations ``y_t`` (each shape ``(ydim,)`` or ``(1, ydim)``).
    :param readout: optional :class:`vjf.readout.OnlineReadout`. Required iff the model was
        built with ``encoder='projection'``; it supplies the recognition input
        ``pinv(C)(g~(y)-b)`` per sample. When ``adapt_readout`` it is updated (CCIPCA) each
        step and Procrustes-refreshed into the decoder every ``readout.K`` steps.
        Warm-start / ``set_fixed`` it (and write to the decoder) before calling.
    :param u_stream: optional iterable of controls ``u_t`` aligned with ``stream``.
    :param adapt_readout: if False, hold the readout fixed -- feature without advancing its
        running mean, no CCIPCA update, no refresh (use with ``OnlineReadout.set_fixed``).
    :param warmup_steps: dynamics are frozen for the first ``warmup_steps`` samples while the
        recognition settles; at the boundary the RBF flow is initialized once from the
        buffered latent means (and controls) and its bump widths are scaled by ``rbf_width_scale``.
    :param logvar_floor: lower clamp on ``transition.logvar`` each step (anti-collapse).
    :param max_abs_state: a step whose posterior mean is non-finite or exceeds this in
        magnitude is treated as diverged (mean repeats previous, posterior reset).
    """
    xdim = model.mean.shape[-1]
    has_u = u_stream is not None
    us = iter(u_stream) if has_u else None
    q = None
    prev_mean = np.zeros(xdim, dtype=np.float32)   # repeat-last fallback on divergence
    warm_means: list[np.ndarray] = []              # latent means buffered during warm-up
    warm_us: list = []                             # matching controls (tensors) if has_u
    initialized = False

    def diverged_result(t, t0):
        return OnlineResult(step=t, mean=prev_mean.copy(), logvar=np.zeros(xdim, dtype=np.float32),
                            pred_mean=None, loss=float("nan"), recon=float("nan"),
                            dynamics=float("nan"), entropy=float("nan"),
                            warming_up=(t < warmup_steps), diverged=True, refreshed=False,
                            elapsed_s=time.perf_counter() - t0)

    for t, y_t in enumerate(stream):
        u_t = _as_u(next(us)) if has_u else None
        warm = t < warmup_steps

        # One-time dynamics initialization at the warm-up boundary (the only buffered step).
        # Controls align with x_next (u applied in the transition x_prev -> x_next), so pass u[1:].
        if t == warmup_steps and not initialized and len(warm_means) > 1:
            m = torch.as_tensor(np.asarray(warm_means), dtype=torch.get_default_dtype())
            ut = torch.cat(warm_us[1:], 0) if has_u else None
            model.transition.initialize(m[1:], m[:-1], ut)
            model.transition.velocity.feature.logwidth.data += math.log(rbf_width_scale)
            initialized = True
            warm_means, warm_us = [], []

        # One-step prediction from the previous posterior (diagnostic; outside the timed region).
        pred = None
        if q is not None:
            with torch.no_grad():
                pred = _mean(model.transition(q.mean, u_t, sampling=False)).detach().cpu().numpy()[0]

        t0 = time.perf_counter()  # online inference cost: projection + filter + refresh + guards

        y_enc = None
        if readout is not None:
            g = readout.feature(np.asarray(y_t), update_mean=adapt_readout)
            if adapt_readout:
                readout.update(g)
            y_enc = torch.as_tensor(readout.project(g))

        def _buffer_divergence():
            if warm:
                warm_means.append(prev_mean.copy())
                if has_u:
                    warm_us.append(u_t)

        try:
            qt, loss, recon, dyn, ent = model.filter(
                y_t, u_t, q, sgd=True, update=True, verbose=True, warm_up=warm, y_enc=y_enc)
        except AssertionError:        # gaussian_loss non-finite guard tripped
            model.transition.logvar.data.clamp_(min=logvar_floor)
            q = None
            _buffer_divergence()
            yield diverged_result(t, t0)
            continue

        model.transition.logvar.data.clamp_(min=logvar_floor)
        mu = qt.mean.detach()
        if not torch.isfinite(mu).all() or mu.abs().max() > max_abs_state:
            q = None
            _buffer_divergence()
            yield diverged_result(t, t0)
            continue

        # Accepted step: only now (re)write the decoder via Procrustes refresh (skip t==0).
        refreshed = False
        if readout is not None and adapt_readout and t > 0:
            refreshed = bool(readout.maybe_refresh(model.decoder, t))
        elapsed = time.perf_counter() - t0

        q = qt
        mean_np = mu.cpu().numpy()[0].astype(np.float32)
        prev_mean = mean_np
        if warm:
            warm_means.append(mean_np)
            if has_u:
                warm_us.append(u_t)

        yield OnlineResult(
            step=t, mean=mean_np, logvar=qt.logvar.detach().cpu().numpy()[0].astype(np.float32),
            pred_mean=pred, loss=float(loss.detach()), recon=float(recon.detach()),
            dynamics=float(dyn.detach()), entropy=float(ent.detach()),
            warming_up=warm, diverged=False, refreshed=refreshed, elapsed_s=elapsed)


def online_filter_trials(model, trials, *, readout=None, adapt_readout=True,
                         warmup_trials=1, rbf_width_scale=1.0,
                         logvar_floor=math.log(1e-6), max_abs_state=1e3,
                         seed_centers=None):
    """Drive ``model`` over a sequence of independent ``trials`` (each an iterable of
    y_t), resetting the posterior to the prior at every trial start. The model
    (readout, flow) persists across trials. Trials 0..warmup_trials-1 are the
    coverage warm-up (dynamics off, latent means buffered across ALL of them); at the
    boundary the flow is initialized once from the buffered means, and (if
    ``seed_centers``) RBF centers are placed data-drivenly (Task 4). Yields one
    OnlineResult per sample with .trial/.t_in_trial set.

    No new math: every step calls model.filter / OnlineReadout, exactly as
    online_filter, only the reset bookkeeping differs.

    Note: seed_centers requires the data-driven RBFDS.initialize kwargs from Task 4.
    """
    xdim = model.mean.shape[-1]
    warm_means = []
    warm_trial_starts = []                         # index into warm_means at each warm-trial start
    initialized = False
    global_t = 0
    for ti, trial in enumerate(trials):
        warm = ti < warmup_trials
        q = None                                   # reset to prior at trial start
        prev_mean = np.zeros(xdim, dtype=np.float32)
        for k, y_t in enumerate(trial):
            if warm and k == 0:
                warm_trial_starts.append(len(warm_means))
            # one-time init at the coverage boundary (first sample after warm-up trials)
            if ti == warmup_trials and not initialized and len(warm_means) > 1:
                m = torch.as_tensor(np.asarray(warm_means), dtype=torch.get_default_dtype())
                # Consecutive transition pairs (i -> i+1), EXCLUDING the spurious ones
                # that straddle a warm-trial boundary (the posterior reset to the prior
                # at each trial start, so last-bin(trial)->first-bin(next-trial) is not a
                # real transition). The center seeding still uses ALL warm states.
                starts = set(warm_trial_starts)
                idx = [i for i in range(len(warm_means) - 1) if (i + 1) not in starts]
                xs = m[idx]
                xt = m[[i + 1 for i in idx]]
                if seed_centers is not None:
                    centers, logw = seed_centers(np.asarray(warm_means))
                    model.transition.initialize(xt, xs, None,
                                                rbf_centers=centers, rbf_logwidths=logw)
                else:
                    model.transition.initialize(xt, xs, None)
                    model.transition.velocity.feature.logwidth.data += math.log(rbf_width_scale)
                initialized = True
            pred = None
            if q is not None:
                with torch.no_grad():
                    pred = _mean(model.transition(q.mean, None, sampling=False)).detach().cpu().numpy()[0]
            t0 = time.perf_counter()
            y_enc = None
            if readout is not None:
                g = readout.feature(np.asarray(y_t), update_mean=adapt_readout)
                if adapt_readout:
                    readout.update(g)
                y_enc = torch.as_tensor(readout.project(g))
            try:
                qt, loss, recon, dyn, ent = model.filter(
                    y_t, None, q, sgd=True, update=True, verbose=True, warm_up=warm, y_enc=y_enc)
            except AssertionError:
                model.transition.logvar.data.clamp_(min=logvar_floor)
                q = None
                if warm:
                    warm_means.append(prev_mean.copy())
                yield OnlineResult(step=global_t, mean=prev_mean.copy(),
                                   logvar=np.zeros(xdim, np.float32), pred_mean=None,
                                   loss=float("nan"), recon=float("nan"), dynamics=float("nan"),
                                   entropy=float("nan"), warming_up=warm, diverged=True,
                                   refreshed=False, elapsed_s=time.perf_counter()-t0,
                                   trial=ti, t_in_trial=k)
                global_t += 1
                continue
            model.transition.logvar.data.clamp_(min=logvar_floor)
            mu = qt.mean.detach()
            if (not torch.isfinite(mu).all()) or (mu.abs().max() > max_abs_state):
                # Same nan sentinel as the AssertionError path / online_filter: reject the
                # posterior, emit nan metrics + zero logvar + repeat-last mean (not the
                # rejected step's finite values).
                q = None
                if warm:
                    warm_means.append(prev_mean.copy())
                yield OnlineResult(step=global_t, mean=prev_mean.copy(),
                                   logvar=np.zeros(xdim, np.float32), pred_mean=None,
                                   loss=float("nan"), recon=float("nan"), dynamics=float("nan"),
                                   entropy=float("nan"), warming_up=warm, diverged=True,
                                   refreshed=False, elapsed_s=time.perf_counter()-t0,
                                   trial=ti, t_in_trial=k)
                global_t += 1
                continue
            refreshed = False
            if readout is not None and adapt_readout and not warm:
                refreshed = bool(readout.maybe_refresh(model.decoder, global_t))
            elapsed = time.perf_counter() - t0
            q = qt
            mean_np = mu.cpu().numpy()[0].astype(np.float32)
            prev_mean = mean_np
            if warm:
                warm_means.append(mean_np)
            yield OnlineResult(step=global_t, mean=mean_np,
                               logvar=qt.logvar.detach().cpu().numpy()[0].astype(np.float32),
                               pred_mean=pred, loss=float(loss.detach()), recon=float(recon.detach()),
                               dynamics=float(dyn.detach()), entropy=float(ent.detach()),
                               warming_up=warm, diverged=False, refreshed=refreshed,
                               elapsed_s=elapsed, trial=ti, t_in_trial=k)
            global_t += 1
