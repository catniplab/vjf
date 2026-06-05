"""Real-time online VJF tutorial: learn a rotating limit cycle from a streaming
log-linear Poisson population, online, one sample at a time, with an *unknown*
observation model recovered on the fly.

Run::

    uv run python script/realtime_tutorial.py

What it shows:
- a dependency-free synthetic spike stream (``vjf.synthetic``);
- the projection encoder + decoupled online readout (``OnlineReadout``) so the
  recognition reads ``pinv(C)(g~(y)-b)`` instead of raw spikes, with ``C`` learned
  online (no oracle);
- the reusable per-sample loop ``vjf.realtime.online_filter`` (posterior threading,
  warm-up -> dynamics, divergence guards, Procrustes refresh, per-bin timing);
- an online trailing-window R^2 diagnostic and per-bin compute time (real-time check).

The first ``init_w`` bins warm-start the readout; the model then goes *live* on the
unseen remainder of the stream (no look-ahead, no double-counting). ``run()`` returns a
results dict and is the smoke-test entry point; ``__main__`` prints a summary and saves
two plots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vjf import synthetic as syn
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter


def affine_r2(mu: np.ndarray, z: np.ndarray) -> float:
    """R^2 of the best affine image of mu onto z (VJF latent is affine-identifiable)."""
    X = np.concatenate([mu, np.ones((len(mu), 1))], axis=1)
    W, *_ = np.linalg.lstsq(X, z, rcond=None)
    sse = float(((z - X @ W) ** 2).sum())
    tss = float(((z - z.mean(0)) ** 2).sum())
    return 1.0 - sse / (tss + 1e-12)


def run(*, t_eff: int = 8000, n_neurons: int = 80, n_rbf: int = 100,
        warmup_steps: int = 1500, refresh_K: int = 500, align_window: int = 2000,
        log_every: int = 200, seed: int = 20260605, plot: bool = True,
        outdir: str | None = None) -> dict:
    """Drive one online pass and return diagnostics.

    The first ``init_w = min(10*n_neurons, t_eff//2)`` bins warm-start the readout; the
    online loop then runs on ``counts[init_w:]`` only (the unseen remainder), so the scored
    samples never include warm-start data. Returns a dict with the trailing-R^2 curve, steady
    per-bin compute time (mean + p95, ms), divergence/refresh counts, and arrays for plotting.
    """
    if log_every <= 0:
        raise ValueError("log_every must be > 0")
    if align_window < 10:
        raise ValueError("align_window must be >= 10")
    init_w = min(10 * n_neurons, t_eff // 2)                  # cheap PCA warm-start window
    if t_eff <= init_w + warmup_steps:
        raise ValueError(f"t_eff ({t_eff}) must exceed init_w ({init_w}) + warmup_steps ({warmup_steps})")

    # --- synthetic stream (unknown C, b to the model) ---
    z = syn.limit_cycle(t_eff, dt=5e-3, angular_velocity=30.0, seed=seed)
    C, b = syn.poisson_readout(z, n_neurons, mean_rate=0.1, peak_rate=0.5, seed=seed)
    counts = np.array(list(syn.stream(z, C, b, seed=seed + 1)))

    # --- model: projection encoder + square-root-RLS dynamics ---
    torch.manual_seed(seed)
    model = VJF.make_model(n_neurons, 2, 0, n_rbf, hidden_sizes=[100, 100],
                           likelihood="poisson", transition_flow="srrls", encoder="projection")

    # --- cheap small-window PCA warm-start of the readout into the decoder ---
    ro = OnlineReadout(n_neurons, 2, smooth_tau=8.0, refresh_K=refresh_K, link="log")
    Cp, bp = ro.warm_start(counts[:init_w])                  # consume the first init_w bins
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cp))
        model.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1)))
    model.decoder.requires_grad_(False)                     # the readout owns (C, b), not SGD

    # --- go live on the unseen remainder: one OnlineResult per arriving sample ---
    live = counts[init_w:]
    z_live = z[init_w:init_w + len(live)]
    means = np.zeros((len(live), 2), dtype=np.float32)
    r2_steps, r2_vals, per_bin_ms = [], [], []
    n_diverge = n_refresh = 0
    for r in online_filter(model, live, readout=ro, warmup_steps=warmup_steps,
                           rbf_width_scale=0.5):
        means[r.step] = r.mean
        n_diverge += int(r.diverged)
        n_refresh += int(r.refreshed)
        if not r.warming_up and not r.diverged:
            per_bin_ms.append(r.elapsed_s * 1e3)
        if (r.step + 1) % log_every == 0 and r.step + 1 > warmup_steps:
            lo = max(0, r.step + 1 - align_window)
            r2_steps.append(init_w + r.step + 1)
            r2_vals.append(affine_r2(means[lo:r.step + 1], z_live[lo:r.step + 1]))

    pbm = np.array(per_bin_ms)
    out = {
        "encoder": "projection", "stream_start": init_w,
        "r2_steps": np.array(r2_steps), "r2_vals": np.array(r2_vals),
        "final_r2": float(r2_vals[-1]) if r2_vals else float("nan"),
        "per_bin_ms": float(pbm.mean()) if pbm.size else float("nan"),       # steady (excl. warm-up/init/diagnostics)
        "per_bin_p95_ms": float(np.percentile(pbm, 95)) if pbm.size else float("nan"),
        "n_diverge": n_diverge, "n_refresh": n_refresh,
        "_means": means, "_z": z_live,
    }
    if plot:
        _plots(out, Path(outdir) if outdir else Path(__file__).resolve().parent / "tutorial_out")
    return out


def _plots(out: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    z, mu, r2 = out["_z"], out["_means"], out["r2_vals"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(out["r2_steps"], r2, lw=2)
    ax.set_xlabel("stream position (bins)"); ax.set_ylabel("trailing affine R^2")
    lo_y = min(-0.1, float(np.nanmin(r2)) - 0.05) if r2.size else -0.1   # don't clip a bad run
    ax.set_ylim(lo_y, 1.0); ax.set_title("Online latent recovery (unknown readout)")
    fig.tight_layout(); fig.savefig(outdir / "trailing_r2.png", dpi=130); plt.close(fig)

    seg = slice(max(0, len(z) - 2000), len(z))
    X = np.concatenate([mu[seg], np.ones((len(mu[seg]), 1))], 1)
    aligned = X @ np.linalg.lstsq(X, z[seg], rcond=None)[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(z[seg, 0], z[seg, 1], lw=0.8, alpha=0.7, label="true z")
    ax.plot(aligned[:, 0], aligned[:, 1], lw=0.8, alpha=0.7, label="VJF (aligned)")
    ax.set_aspect("equal"); ax.legend(); ax.set_title("Phase portrait (last 2000 bins)")
    fig.tight_layout(); fig.savefig(outdir / "phase_portrait.png", dpi=130); plt.close(fig)
    print(f"wrote plots to {outdir}")


if __name__ == "__main__":
    res = run()
    print(f"final trailing R^2 = {res['final_r2']:.3f}   "
          f"per-bin online cost = {res['per_bin_ms']:.3f} ms (p95 {res['per_bin_p95_ms']:.3f})   "
          f"refreshes = {res['n_refresh']}   diverged = {res['n_diverge']}")
