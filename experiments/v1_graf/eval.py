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


import contextlib

import torch


@contextlib.contextmanager
def frozen_dynamics(model):
    """Make a held-out evaluation a true no-op on model state: disable the training-time
    denoising bump (``dyn_noise``) and restore both it and the schedule counter
    ``_dyn_step`` on exit. The bump perturbs only the discarded prediction ``pt`` (not the
    filtered posterior ``qt`` we read), so this does not change any eval value; it just
    stops eval from advancing ``_dyn_step`` / consuming the global RNG, so the frozen-eval
    metrics are correct by construction rather than by accident."""
    dn = getattr(model, "dyn_noise", 0.0)
    step = getattr(model, "_dyn_step", 0)
    model.dyn_noise = 0.0
    try:
        yield
    finally:
        model.dyn_noise = dn
        model._dyn_step = step


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
    with frozen_dynamics(model):
        for n in range(N):
            readout.nu = nu_snapshot.copy()                     # each pass sees the same EMA
            keep = np.arange(N) != n
            Csub = C[keep]; Cpinv = np.linalg.pinv(Csub).astype(np.float32)
            q = None
            for t in range(n_bin):
                y = trial_counts[t]
                g = readout.feature(y, update_mean=False)       # frozen feature
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
    (true_path: (k, m)). DEPRECATED for selection (affine-overfits at short k / high L);
    kept for the latent-space diagnostic only. Selection uses
    forecast_reconstruction_deviance."""
    x, _ = model.forecast(torch.as_tensor(x0[None].astype(np.float32)), n_step=k)
    pred = x.detach().cpu().numpy()[1:, 0, :]
    A = np.concatenate([pred, np.ones((k, 1))], 1)
    W, *_ = np.linalg.lstsq(A, true_path, rcond=None)
    sse = ((true_path - A @ W) ** 2).sum(); tss = ((true_path - true_path.mean(0)) ** 2).sum() + 1e-12
    return float(1 - sse / tss)


def _poisson_deviance(n: np.ndarray, lam: np.ndarray) -> float:
    """Poisson deviance 2*sum[n log(n/lam) - (n - lam)] (n log(n/lam):=0 at n=0). >= 0,
    additive over bins/neurons/trials; 0 iff lam == n everywhere."""
    lam = np.clip(lam, 1e-6, None)
    term = np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / lam), 0.0) - (n - lam)
    return float(2.0 * term.sum())


@torch.no_grad()
def forecast_reconstruction_deviance(model, readout, trial_counts: np.ndarray,
                                     psth_counts: np.ndarray, starts, horizons) -> dict:
    """GOLD-STANDARD metric building block: future forecasted reconstruction for one
    held-out trial. Filter the trial (frozen, all neurons) to each start t0, free-run the
    flow with NO observations after t0, decode to a predicted per-bin rate, and accumulate
    the Poisson deviance of that forecast against the FUTURE spikes over each horizon k,
    alongside two baselines -- persistence (hold the model's t0 rate) and the stimulus-locked
    PSTH (``psth_counts``: trial-averaged counts/bin, (n_bin, N)). Deviances are additive, so
    returns ``{k: {'model','persist','psth','spikes'}}`` summed over the given starts; the
    caller sums over trials/directions then forms skill_k = 1 - sum(model)/sum(baseline)."""
    C = model.decoder.decode.weight.detach().cpu().numpy()
    b = model.decoder.decode.bias.detach().cpu().numpy()
    n_bin, N = trial_counts.shape
    Cpinv = np.linalg.pinv(C).astype(np.float32)
    nu0 = readout.nu.copy(); readout.nu = nu0.copy()
    q = None
    xfilt = np.zeros((n_bin, C.shape[1]))
    with frozen_dynamics(model):
        for t in range(n_bin):                                  # filter the full trial, frozen
            g = readout.feature(trial_counts[t], update_mean=False)
            x_enc = torch.as_tensor((Cpinv @ (g - readout.mean_b)).astype(np.float32))
            qt, *_ = model.filter(trial_counts[t], None, q, sgd=False, update=False, y_enc=x_enc)
            q = qt
            xfilt[t] = qt.mean.detach().cpu().numpy()[0]
    readout.nu = nu0
    kmax = max(horizons)
    out = {k: {"model": 0.0, "persist": 0.0, "psth": 0.0, "spikes": 0.0} for k in horizons}
    for t0 in starts:
        if t0 + kmax >= n_bin:
            continue
        _, ylog = model.forecast(torch.as_tensor(xfilt[t0][None].astype(np.float32)), n_step=kmax)
        lam_fc = np.exp(ylog.detach().cpu().numpy()[1:kmax + 1, 0, :])     # (kmax, N) forecast rate
        lam_persist = np.exp(C @ xfilt[t0] + b)                           # (N,) model rate at t0, held
        for k in horizons:
            fut = trial_counts[t0 + 1:t0 + 1 + k]                         # (k, N) future spikes
            out[k]["model"] += _poisson_deviance(fut, lam_fc[:k])
            out[k]["persist"] += _poisson_deviance(fut, np.broadcast_to(lam_persist, (k, N)))
            out[k]["psth"] += _poisson_deviance(fut, psth_counts[t0 + 1:t0 + 1 + k])
            out[k]["spikes"] += float(fut.sum())
    return out


def forecast_skill_summary(devs: list, horizons=(8, 16, 32), weights=(0.5, 0.3, 0.2)) -> dict:
    """Aggregate per-trial/per-direction deviance dicts (from
    forecast_reconstruction_deviance) into skill_k = 1 - sum(model)/sum(baseline) vs the
    persistence and PSTH baselines, and the pre-declared weighted persistence-skill S (the
    selection score). Deviances summed across all entries before forming the ratio.

    Guards against a spurious perfect score: with no evaluated starts / no future spikes the
    baseline deviance is ~0, which would read as skill 1.0 -- report nan instead so an empty
    or diverged config cannot win the search. A non-finite model deviance (a diverged
    free-run) likewise propagates to nan."""
    keys = ("model", "persist", "psth", "spikes")
    agg = {k: {key: sum(d[k][key] for d in devs) for key in keys} for k in horizons}

    def _skill(model_d, base_d, spikes):
        if spikes <= 0 or base_d <= 1e-9 or not np.isfinite(model_d):
            return float("nan")
        return 1.0 - model_d / base_d

    skill = {k: {"vs_persist": _skill(agg[k]["model"], agg[k]["persist"], agg[k]["spikes"]),
                 "vs_psth": _skill(agg[k]["model"], agg[k]["psth"], agg[k]["spikes"])}
             for k in horizons}
    sp = [skill[k]["vs_persist"] for k in horizons]
    S = float(sum(w * s for w, s in zip(weights, sp))) if np.all(np.isfinite(sp)) else float("nan")
    return {"skill": skill, "weighted_persist_skill": S,
            "skill8_persist": skill[horizons[0]]["vs_persist"],
            "total_spikes": float(sum(agg[k]["spikes"] for k in horizons))}
