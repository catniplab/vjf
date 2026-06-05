"""Self-contained limit-cycle latent + log-linear Poisson calibration.

The LimitCycle dynamics are adapted from
``meta-dynamical-ssm/examples/limit_cycle/data.py`` (catniplab). The Poisson
loading matrix is calibrated to a target Fisher-information SNR by
``neurofisherSNR`` (catniplab, Jeon & Park, EUSIPCO 2024, arXiv:2408.08752),
pinned to a commit in requirements-exp.txt.

Observations are NOT materialized here: at 1 ms bins x 1e6 steps x thousands of
neurons the full count matrix is tens of GB. Instead we return the latent `z`
and the calibrated `(C, b)`; the caller streams Poisson counts in chunks via
`sample_counts`. Latent generation and calibration run on CPU with explicit
seeding so the stream is bit-identical regardless of training device.
"""
from __future__ import annotations

import contextlib
import io

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

    def trajectory(self, x0, num_steps, process_noise, generator):
        x = torch.empty(x0.shape[0], num_steps, x0.shape[1])
        x[:, 0] = x0
        for t in range(num_steps - 1):
            noise = self.dt * process_noise * torch.randn(x0.shape, generator=generator)
            x[:, t + 1] = self.step(x[:, t]) + noise
        return x


def generate_latent(
    *,
    t_eff: int,
    stride: int = 1,
    dt: float = 1e-3,
    angular_velocity: float = 3.0,
    radius_scale: float = 2.0,
    process_noise: float = 0.5,
    seed: int = 20260602,
) -> np.ndarray:
    """One long single-trial limit-cycle latent, per-dim zero-mean unit-var.

    Returns z (t_eff, 2) float32 -- the ground-truth latent (identifiable up to
    an affine transform).
    """
    g = torch.Generator().manual_seed(seed)
    dyn = LimitCycle(radius_scale, angular_velocity, dt)
    x0 = torch.randn(1, 2, generator=g)
    x = dyn.trajectory(x0, t_eff * stride, process_noise, g)[:, ::stride].float()[0]
    x_np = x.numpy()
    z = (x_np - x_np.mean(0, keepdims=True)) / (x_np.std(0, keepdims=True) + 1e-8)
    return z.astype("float32")


def calibrate_poisson(
    z: np.ndarray,
    *,
    n_neurons: int,
    snr_db: float,
    target_mean_rate: float = 0.02,
    target_max_rate: float = 0.1,
    p_coh: float = 0.95,
    p_sparse: float = 0.1,
    priority: str = "mean",
    seed: int = 20260602,
) -> dict:
    """Calibrate a log-linear Poisson loading (C, b) to a target SNR.

    Calibrates on a representative subsample of z (the neurofisherSNR fitter is
    O(n_neurons^2) per row). Returns C (n_neurons, 2), b (1, n_neurons), and the
    realized SNR. Rates/counts are produced later by `sample_counts`.
    """
    from neurofisherSNR.observation import gen_poisson_observations

    n_calib = min(80, z.shape[0])
    idx = np.linspace(0, z.shape[0] - 1, n_calib).round().astype(int)
    z_calib = z[idx]

    np.random.seed(seed)
    C_init = np.random.randn(n_neurons, 2)
    C_init = C_init * (np.random.rand(n_neurons, 2) > p_sparse)
    C_init = C_init / (np.linalg.norm(C_init, axis=1, keepdims=True) + 1e-8)

    with contextlib.redirect_stdout(io.StringIO()):
        _obs, C_np, b_np, _rates, snr_realized = gen_poisson_observations(
            x=z_calib, C=C_init, d_neurons=n_neurons,
            tgt_rate_per_bin=target_mean_rate, max_rate_per_bin=target_max_rate,
            priority=priority, p_coh=p_coh, p_sparse=p_sparse, tgt_snr=snr_db,
        )
    return {
        "C": C_np.astype("float32"),      # (n_neurons, 2)
        "b": b_np.astype("float32"),      # (1, n_neurons)
        "snr_target": float(snr_db),
        "snr_realized": float(snr_realized),
    }


def rate_at(z_slice: np.ndarray, C: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Poisson rate exp(z C^T + b) for a slice of latents (no sampling)."""
    return np.exp(z_slice @ C.T + b)


def sample_counts(z_slice: np.ndarray, C: np.ndarray, b: np.ndarray, rng) -> np.ndarray:
    """Sample Poisson spike counts for a slice of latents (streaming chunk)."""
    return rng.poisson(rate_at(z_slice, C, b)).astype("float32")
