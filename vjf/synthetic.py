"""Self-contained synthetic data for VJF demos and tests: a 2-D limit cycle
observed through a log-linear Poisson population.

Unlike the validation experiment (which calibrates the loading to a target
Fisher-information SNR via the external ``neurofisherSNR`` package), this module
is dependency-free: the loading ``C`` and bias ``b`` are set by a simple analytic
gain calibration that hits a target mean and peak firing rate. That is enough for
a tutorial/test stream; it is not an SNR-calibrated benchmark.

The latent dynamics (``LimitCycle``) are adapted from
``meta-dynamical-ssm/examples/limit_cycle`` (catniplab).
"""
from __future__ import annotations

import numpy as np
import torch


class LimitCycle:
    """2-D limit cycle: radius relaxes to sqrt(radius_scale), constant omega."""

    def __init__(self, radius_scale: float, angular_velocity: float, dt: float = 1e-2):
        self.radius_scale = radius_scale
        self.angular_velocity = angular_velocity
        self.dt = dt

    def step(self, x: torch.Tensor) -> torch.Tensor:
        radius = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        theta = torch.atan2(x[:, 1], x[:, 0])
        radius = radius + radius * (self.radius_scale - radius**2) * self.dt
        theta = theta + self.angular_velocity * self.dt
        return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)


def limit_cycle(t_eff: int, *, dt: float = 5e-3, angular_velocity: float = 30.0,
                radius_scale: float = 2.0, process_noise: float = 0.5,
                seed: int = 20260605) -> np.ndarray:
    """One single-trial limit-cycle latent, per-dim zero-mean unit-variance.

    Returns z (t_eff, 2) float32 -- the ground-truth latent (identifiable only up
    to an affine transform).
    """
    g = torch.Generator().manual_seed(seed)
    dyn = LimitCycle(radius_scale, angular_velocity, dt)
    x = torch.empty(1, t_eff, 2)
    x[:, 0] = torch.randn(1, 2, generator=g)
    for t in range(t_eff - 1):
        noise = dt * process_noise * torch.randn(1, 2, generator=g)
        x[:, t + 1] = dyn.step(x[:, t]) + noise
    z = x[0].numpy()
    z = (z - z.mean(0, keepdims=True)) / (z.std(0, keepdims=True) + 1e-8)
    return z.astype("float32")


def poisson_readout(z: np.ndarray, n_neurons: int, *, mean_rate: float = 0.1,
                    peak_rate: float = 0.5, sparsity: float = 0.1,
                    seed: int = 20260605):
    """Log-linear Poisson loading (C, b) with rate = exp(z C^T + b).

    Simple analytic gain calibration (NOT Fisher-SNR): random sparse unit-norm
    rows are scaled by a shared gain so the median per-neuron peak/mean rate ratio
    matches ``peak_rate/mean_rate``, then the per-neuron bias is set so each
    neuron's time-mean rate equals ``mean_rate``. Rates are per bin.

    Returns C (n_neurons, 2) float32 and b (1, n_neurons) float32.
    """
    if not (mean_rate > 0):
        raise ValueError(f"mean_rate must be > 0, got {mean_rate}")
    if peak_rate < mean_rate:
        raise ValueError(f"peak_rate ({peak_rate}) must be >= mean_rate ({mean_rate})")
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((n_neurons, 2))
    mask = rng.random((n_neurons, 2)) > sparsity                 # keep with prob 1-sparsity
    C *= mask
    zero_rows = ~C.any(axis=1)                                   # all-zero loading: no modulation
    while zero_rows.any():                                       # resample so every neuron is tuned
        C[zero_rows] = rng.standard_normal((int(zero_rows.sum()), 2))
        zero_rows = ~C.any(axis=1)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)        # unit rows
    s = z @ C.T                                                   # (T, n) modulation

    target_ratio = peak_rate / mean_rate
    def median_ratio(g):                                          # median peak/mean over neurons
        e = np.exp(g * s)
        return float(np.median(e.max(0) / (e.mean(0) + 1e-12)))
    lo, hi = 0.05, 5.0                                            # bisection on the shared gain
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if median_ratio(mid) < target_ratio:
            lo = mid
        else:
            hi = mid
    C = (C * (0.5 * (lo + hi))).astype("float32")

    s = z @ C.T
    b = (np.log(mean_rate) - np.log(np.exp(s).mean(0) + 1e-12)).astype("float32")
    return C, b.reshape(1, -1)


def rate_at(z_slice: np.ndarray, C: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Poisson rate exp(z C^T + b) (no sampling)."""
    return np.exp(z_slice @ C.T + b)


def stream(z: np.ndarray, C: np.ndarray, b: np.ndarray, *, seed: int = 20260605):
    """Yield one Poisson count vector y_t (n_neurons,) float32 per latent sample.

    A per-sample generator suitable for driving ``vjf.realtime.online_filter``."""
    rng = np.random.default_rng(seed)
    for t in range(z.shape[0]):
        yield rng.poisson(rate_at(z[t], C, b).reshape(-1)).astype("float32")
