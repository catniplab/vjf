"""Multi-step (rollout) fine-tuning of the VJF flow to capture the autonomous limit cycle.

One-step ELBO training underfits the free-run: a contracting focus minimizes the
observation-corrected one-step residual while its autonomous free-run decays. This phase trains
the FREE-RUN directly. The decoded-grating response, after the onset transient, is an
autonomous attracting limit cycle observed through the spikes; we recover it by fine-tuning only
the flow weights with a non-causal rollout loss on the buffered filtered trajectory (training is
non-causal; deployment/eval stays causal). From each steady-state start t (bin >= t0), free-run
the flow k steps and match the filtered means:

    xhat^0 = mu_t,   xhat^j = xhat^{j-1} + f(xhat^{j-1}),   j = 1..k
    L_roll = (1/k) sum_t sum_{j=1..k} || xhat^j - mu_{t+j} ||^2          (latent-MSE form)

The mu are detached targets (the frozen filtered means); gradients flow through the k-step
recurrence to the flow weights only (gradient-clipped). See report eq:roll.
"""
from __future__ import annotations
import numpy as np
import torch

from experiments.v1_graf.run_m1 import _infer_latent_paths

PERIOD = 16                                                     # grating period (10 ms bins/cycle)
N_TILE = 20                                                     # cycles in the tiled periodic target


def rollout_finetune(model, ro, train_trials, *, k: int = 12, epochs: int = 25,
                     lr: float = 5e-4, t0: int = 32, seed: int = 20260615,
                     k_ladder=None, target_mode: str = "filtered", clip: float = 1.0,
                     max_starts: int = 64, verbose: bool = False):
    """Fine-tune ONLY the flow weights so the autonomous free-run reproduces the filtered
    trajectory over k steps on the steady-state window (bins >= t0). sgd flow only. Returns model.

    Fixed-horizon (``k_ladder=None``): every update free-runs the same ``k`` steps.

    Curriculum + mixture (``k_ladder`` a list of horizons, e.g. ``[8, 16, 32, 48]``): the
    epochs are split into ``len(k_ladder)`` blocks; in block ``b`` the unlocked horizons are
    ``k_ladder[:b+1]`` and each per-trial update samples its horizon uniformly from them. A
    short horizon (<= half a cycle) only constrains the local flow; a horizon spanning >= one
    full period is what can close the orbit, but back-propagating through it is unstable -- so
    the curriculum unlocks long horizons only after short ones have warmed up the flow, and the
    mixture keeps short (stable-gradient) horizons in every later block. ``k`` is ignored when
    ``k_ladder`` is given.

    ``target_mode`` chooses what the free-run must reproduce:
      - ``"filtered"`` (default): each trial's own single-trial filtered means. Across trials the
        cycle is phase/frequency/amplitude jittered, so no single autonomous flow can follow all
        of them over >1 cycle -- the MSE-optimal compromise is a contracting (decaying) flow.
      - ``"trialavg"``: the stimulus-locked trial-AVERAGE filtered orbit -- one coherent,
        phase-aligned closed orbit (the limit cycle the single trials noisily sample). A single
        flow can follow it, so this is what can yield a sustained/attracting orbit. With one
        target trajectory each epoch is a single batched step over its phases, so pass more
        ``epochs``.
      - ``"cycle"``: the trial-average steady state FOLDED into one clean period and TILED to
        ``N_TILE`` cycles -- a long exactly-periodic target. MSE over a few cycles cannot feel a
        slow per-cycle amplitude decay (~1%/cycle is <3% over 3 cycles); rolling out over many
        cycles against this target turns that decay into a large penalty, which is what can pin
        the amplitude mode to marginal stability (a limit cycle) rather than a decaying focus.

    ``clip`` is the grad-norm cap (raise it to let long-horizon gradients through); ``max_starts``
    subsamples the free-run start phases each update to bound the autograd-graph memory at large k."""
    if model.transition.flow_learner != "sgd":
        return model                                            # rollout trains the sgd flow weights
    W = model.transition.velocity.w_mean
    opt = torch.optim.Adam([W], lr=lr)
    rng = np.random.default_rng(seed)
    paths = _infer_latent_paths(model, ro, train_trials)        # per-trial filtered means
    if target_mode == "filtered":                               # heterogeneous single-trial paths
        mus = [torch.as_tensor(p, dtype=torch.float32) for p in paths]
        start_floor = t0
    elif target_mode in ("trialavg", "cycle"):
        T = min(len(p) for p in paths)
        avg = np.stack([p[:T] for p in paths], 0).mean(0)       # (T, L) trial-average orbit
        if target_mode == "trialavg":
            mus, start_floor = [torch.as_tensor(avg, dtype=torch.float32)], t0
        else:                                                   # fold steady state -> one cycle, tile
            ss = avg[t0:]
            nc = len(ss) // PERIOD
            one = ss[:nc * PERIOD].reshape(nc, PERIOD, -1).mean(0)  # (PERIOD, L) clean cycle
            mus, start_floor = [torch.as_tensor(np.tile(one, (N_TILE, 1)), dtype=torch.float32)], 0
    else:
        raise ValueError(f"unknown target_mode {target_mode!r}")
    ladder = sorted(int(x) for x in k_ladder) if k_ladder else None
    nblock = len(ladder) if ladder else 1
    for ep in range(epochs):
        allowed = ladder[: 1 + min(nblock - 1, ep * nblock // epochs)] if ladder else None
        ep_loss = 0.0
        for i in rng.permutation(len(mus)):
            mu = mus[i]
            kk = int(rng.choice(allowed)) if ladder else k      # mixture: horizon per update
            starts = np.arange(start_floor, mu.shape[0] - kk)
            if len(starts) == 0:
                continue
            if len(starts) > max_starts:                        # bound autograd memory at long k
                starts = np.sort(rng.choice(starts, max_starts, replace=False))
            starts = torch.as_tensor(starts)
            opt.zero_grad()
            x = mu[starts]                                      # (S, L) free-run starts, batched over phases
            loss = 0.0
            for j in range(1, kk + 1):                          # vectorized free-run, differentiable in W
                x = x + model.transition.velocity(x, sampling=False)
                loss = loss + ((x - mu[starts + j]) ** 2).sum()
            (loss / (len(starts) * kk)).backward()
            torch.nn.utils.clip_grad_norm_([W], clip)
            opt.step()
            ep_loss += float(loss.detach()) / (len(starts) * kk)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            tag = f"k in {allowed}" if ladder else f"k={k}"
            print(f"  rollout ep {ep:2d} ({tag}): mean L_roll {ep_loss / len(mus):.4f}", flush=True)
    return model
