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


def rollout_finetune(model, ro, train_trials, *, k: int = 12, epochs: int = 25,
                     lr: float = 5e-4, t0: int = 32, seed: int = 20260615):
    """Fine-tune ONLY the flow weights so the autonomous free-run reproduces the filtered
    trajectory over k steps on the steady-state window (bins >= t0). sgd flow only. Returns model."""
    if model.transition.flow_learner != "sgd":
        return model                                            # rollout trains the sgd flow weights
    W = model.transition.velocity.w_mean
    opt = torch.optim.Adam([W], lr=lr)
    rng = np.random.default_rng(seed)
    mus = [torch.as_tensor(p, dtype=torch.float32)              # frozen filtered means (detached targets)
           for p in _infer_latent_paths(model, ro, train_trials)]
    for _ in range(epochs):
        for i in rng.permutation(len(mus)):
            mu = mus[i]
            starts = torch.arange(t0, mu.shape[0] - k)
            if len(starts) == 0:
                continue
            opt.zero_grad()
            x = mu[starts]                                      # (S, L) free-run starts, batched over phases
            loss = 0.0
            for j in range(1, k + 1):                           # vectorized free-run, differentiable in W
                x = x + model.transition.velocity(x, sampling=False)
                loss = loss + ((x - mu[starts + j]) ** 2).sum()
            (loss / (len(starts) * k)).backward()
            torch.nn.utils.clip_grad_norm_([W], 1.0)
            opt.step()
    return model
