"""Evaluation metrics for sVJF on the Graf V1 dataset."""
from __future__ import annotations
import numpy as np


def predictive_ll_bits_per_spike(y: np.ndarray, lam: np.ndarray, ybar: float) -> float:
    """Poisson predictive log-likelihood normalized to a homogeneous-Poisson baseline,
    in bits/spike (vLGP Eq. 39). y, lam: (trial, bin, neuron) counts and predicted rates."""
    lam = np.clip(lam, 1e-6, None)
    model_ll = (y * np.log(lam) - lam).sum()
    base_ll = (y * np.log(ybar) - ybar).sum()
    n_spk = y.sum()
    return float((model_ll - base_ll) / (n_spk * np.log(2.0) + 1e-12))


import torch


@torch.no_grad()
def leave_one_neuron_rates(model, readout, trial_counts: np.ndarray) -> np.ndarray:
    """Predicted rates (bin, neuron) for one test trial under leave-one-neuron-out:
    for each held-out neuron n, filter the trial from the OTHER neurons' projection and
    predict lam[:, n] = exp(C[n].x + b[n]). Model/readout frozen (no learning, no refresh)."""
    C = model.decoder.decode.weight.detach().cpu().numpy()      # (N, m)
    b = model.decoder.decode.bias.detach().cpu().numpy()        # (N,)
    n_bin, N = trial_counts.shape
    lam = np.zeros((n_bin, N), dtype=np.float64)
    nu_snapshot = readout.nu.copy()                             # freeze EMA entry state
    for n in range(N):
        readout.nu = nu_snapshot.copy()                         # each pass sees the same EMA
        keep = np.arange(N) != n
        Csub = C[keep]; Cpinv = np.linalg.pinv(Csub).astype(np.float32)
        q = None
        for t in range(n_bin):
            y = trial_counts[t]
            g = readout.feature(y, update_mean=False)           # frozen feature
            x_enc = torch.as_tensor((Cpinv @ (g[keep] - readout.mean_b[keep])).astype(np.float32))
            qt, *_ = model.filter(y, None, q, sgd=False, update=False, verbose=False, y_enc=x_enc)
            q = qt
            xm = qt.mean.detach().cpu().numpy()[0]
            lam[t, n] = np.exp(C[n] @ xm + b[n])
    readout.nu = nu_snapshot                                    # restore caller's EMA state
    return lam
