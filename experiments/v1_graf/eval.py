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


def orientation_decode_acc(latent_per_trial: np.ndarray, dirs: np.ndarray, *,
                           n_splits: int = 5, seed: int = 20260609) -> float:
    """Cross-validated direction-decoding accuracy from per-trial latent summaries
    (latent_per_trial: (n_trial, feat)). Multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=2000)
    y = np.round(dirs).astype(int)
    return float(cross_val_score(clf, latent_per_trial, y, cv=n_splits).mean())


def torus_embedding(latent: np.ndarray, dirs: np.ndarray):
    """Trial-averaged latent per direction, projected to its first 3 singular vectors
    (reproduces vLGP Fig 8). latent (n_trial, m). Returns (72, 3) and the direction axis."""
    axis = np.unique(dirs)
    avg = np.stack([latent[dirs == d].mean(0) for d in axis], 0)   # (72, m)
    u, s, vt = np.linalg.svd(avg - avg.mean(0), full_matrices=False)
    return (avg - avg.mean(0)) @ vt[:3].T, axis


@torch.no_grad()
def forecast_r2(model, x0: np.ndarray, true_path: np.ndarray, k: int) -> float:
    """Affine-aligned R^2 of a k-step free run of the learned flow from x0 vs true_path
    (true_path: (k, m)). Mirrors the synthetic forecast metric."""
    x, _ = model.forecast(torch.as_tensor(x0[None].astype(np.float32)), n_step=k)
    pred = x.detach().cpu().numpy()[1:, 0, :]
    A = np.concatenate([pred, np.ones((k, 1))], 1)
    W, *_ = np.linalg.lstsq(A, true_path, rcond=None)
    sse = ((true_path - A @ W) ** 2).sum(); tss = ((true_path - true_path.mean(0)) ** 2).sum() + 1e-12
    return float(1 - sse / tss)
