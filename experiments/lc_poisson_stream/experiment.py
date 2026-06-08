"""VJF online learning of a log-linear Poisson limit-cycle stream.

Runs one long single-trial stream per SNR level, trains VJF in a single online
pass (warm-up then dynamics-on), tracks learning vs stream position, and writes
diagnostics plots + a provenance-stamped summary.

Reproducibility: deterministic data (CPU, seeded), pinned deps
(requirements-exp.txt), recorded git commit / versions / config in summary.json.

Usage:
    python experiment.py [--device cpu|cuda] [--t-eff 10000] [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lc_data import generate_latent, calibrate_poisson  # noqa: E402,F401

RESULTS = HERE / "results"

# State-noise variance floor. Over a long online pass the RLS velocity fit can
# drive the dynamics residual (and hence transition.logvar) toward zero, making
# exp(-0.5*logvar) overflow and gaussian_loss return NaN. This floor sits below
# the healthy regime (~e^-6 at 1 ms bins) but blocks the collapse-to-zero.
LOGVAR_FLOOR = math.log(1e-6)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def affine_align(mu: np.ndarray, z: np.ndarray):
    """Least-squares affine map mu -> z. Returns (A, c, r2) over the rows given.

    The VJF latent is identifiable only up to an affine transform, so we score
    the best affine image of the filtered mean against the true latent.
    """
    n = mu.shape[0]
    X = np.concatenate([mu, np.ones((n, 1))], axis=1)  # (n, 3)
    W, *_ = np.linalg.lstsq(X, z, rcond=None)           # (3, 2)
    pred = X @ W
    sse = float(((z - pred) ** 2).sum())
    tss = float(((z - z.mean(0, keepdims=True)) ** 2).sum())
    r2 = 1.0 - sse / (tss + 1e-12)
    A, c = W[:2].T, W[2]  # z ~ A @ mu + c
    return A, c, r2


def r2_with(A, c, mu, z):
    pred = mu @ A.T + c
    sse = float(((z - pred) ** 2).sum())
    tss = float(((z - z.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / (tss + 1e-12)


def _mean(out):
    """Mean of a transition output: Gaussian (RLS flow) -> .mean; Tensor (SGD flow) -> itself."""
    return out.mean if isinstance(out, tuple) else out


def pca_readout_init(counts_win, sigma=8.0, d=2):
    """Causal PLDS/GPFA-style readout warm-start from an initial spike window.

    Gaussian-smooth counts in time, log-transform, PCA across neurons -> (C, b),
    with the latent rescaled to unit variance per dim (scale folded into C so the
    rate exp(x C^T + b) is unchanged). Uses ONLY the given initial window -- no
    look-ahead. References: Macke et al. 2011 (PLDS init), Yu et al. 2009 (GPFA).
    """
    from scipy.ndimage import gaussian_filter1d
    Y = gaussian_filter1d(counts_win.astype(np.float64), sigma, axis=0)
    L = np.log(Y + 1e-2)
    b = L.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(L - b, full_matrices=False)
    X = U[:, :d] * S[:d]
    C = Vt[:d].T
    std = X.std(0, keepdims=True) + 1e-8
    return (C * std).astype("float32"), b.astype("float32")


def transition_mean(model, mu, device, chunk=2000):
    """Mean one-step prediction over many states, chunked.

    VJF's LinearRegression.forward(sampling=False) forms an N x N matrix
    (FL @ FL.T) to read its diagonal, so calling it on the full stream is
    O(T^2) memory. Chunking keeps it bounded.
    """
    out = np.empty_like(mu)
    with torch.no_grad():
        for i in range(0, len(mu), chunk):
            o = model.transition(torch.as_tensor(mu[i:i + chunk], device=device), None, sampling=False)
            out[i:i + chunk] = _mean(o).cpu().numpy()
    return out


def grid_field(model, A, c, device, lim=2.4, n=16):
    """Learned velocity field on a z-space grid (arrows mapped from mu-space)."""
    gx = np.linspace(-lim, lim, n)
    grid_z = np.stack(np.meshgrid(gx, gx), -1).reshape(-1, 2).astype(np.float32)
    grid_mu = ((grid_z - c) @ np.linalg.pinv(A).T).astype(np.float32)
    with torch.no_grad():
        nxt = _mean(model.transition(torch.as_tensor(grid_mu, device=device), None,
                                     sampling=False)).cpu().numpy()
    return grid_z, (nxt - grid_mu) @ A.T


def kstep_skill(model, mu_all, z, A, c, lo, hi, K, M, device):
    """k-step free-run forecast vs zero-flow (constant) baseline, k=1..K.

    Rolls the learned dynamics from M start states in [lo, hi-K), maps to
    z-space, and scores against true z[start+k]. Baseline predicts z[start].
    """
    valid_hi = hi - K - 1
    if valid_hi <= lo + 1:
        return None
    starts = np.unique(np.linspace(lo, valid_hi, min(M, valid_hi - lo)).astype(int))
    z_start = z[starts]
    r2m = np.zeros(K); r2b = np.zeros(K); msem = np.zeros(K); mseb = np.zeros(K)
    with torch.no_grad():
        state = torch.as_tensor(mu_all[starts], device=device)
        for k in range(1, K + 1):
            state = _mean(model.transition(state, None, sampling=False))
            pred_z = state.cpu().numpy() @ A.T + c
            true_z = z[starts + k]
            sse_m = ((true_z - pred_z) ** 2).sum()
            sse_b = ((true_z - z_start) ** 2).sum()
            tss = ((true_z - true_z.mean(0)) ** 2).sum() + 1e-12
            r2m[k - 1] = 1 - sse_m / tss
            r2b[k - 1] = 1 - sse_b / tss
            msem[k - 1] = sse_m / true_z.size
            mseb[k - 1] = sse_b / true_z.size
    return {"r2_model": r2m, "r2_base": r2b, "mse_model": msem, "mse_base": mseb}


# --------------------------------------------------------------------------- #
# One SNR condition
# --------------------------------------------------------------------------- #
def run_condition(z, cal, n_neurons, cfg, device):
    from vjf.model import VJF
    from lc_data import rate_at, sample_counts

    C, b = cal["C"], cal["b"]
    T, N = z.shape[0], n_neurons
    warmup_steps = min(int(cfg["warmup_frac"] * T), cfg["warmup_cap"])
    log_every, align_w, chunk = cfg["log_every"], cfg["align_window"], cfg["obs_chunk"]
    snap_steps = sorted({min(int(f * T), T) for f in cfg["snapshots"]})
    rng = np.random.default_rng(cfg["seed"] + 1)  # streaming Poisson sampler

    # Readout: how the decoder (C, b) is obtained.
    #   'oracle'  - pin to the generator's true (C, b), frozen (upper bound).
    #   'learned' - default init, trained by SGD (collapses under low-rate Poisson).
    #   'pca'     - causal PLDS/GPFA-style warm-start from the FIRST ~mult*N spike
    #               bins (Gaussian-smooth -> log -> PCA across neurons), then frozen.
    #               Recovers the oracle bound without knowing (C,b) at high SNR;
    #               expected to degrade at low SNR. Fine-tuning only hurts.
    #   'pca_proj_online' - cheap small-window PCA init + ONLINE incremental-PCA
    #               refinement (vjf.readout.OnlineReadout); the recognition reads the
    #               subspace projection pinv(C)(g~(y)-b) instead of raw spikes. Recovers
    #               the asymptotic readout online without a large init window.
    #   encoder ('spikes'|'projection') is an independent axis: 'projection' feeds the
    #   recognition pinv(C)(g~(y)-b) for ANY readout (so e.g. an oracle-C projection
    #   ceiling is comparable to the online estimate). 'pca_proj_online' forces it.
    readout = cfg["readout"]
    encoder = ("projection" if readout == "pca_proj_online"
               or cfg.get("encoder") == "projection" else "spikes")
    front = None  # OnlineReadout for the projection encoder (None for spike encoders)

    torch.manual_seed(cfg["seed"])
    model = VJF.make_model(
        ydim=N, xdim=2, udim=0, n_rbf=cfg["n_rbf"],
        hidden_sizes=cfg["hidden_sizes"], likelihood="poisson",
        lr=cfg["lr"], lr_decay=1.0, transition_flow=cfg["transition_flow"],
        encoder=encoder,
    ).to(device)

    if readout == "oracle":
        with torch.no_grad():
            model.decoder.decode.weight.copy_(torch.as_tensor(C, device=device))
            model.decoder.decode.bias.copy_(torch.as_tensor(b.reshape(-1), device=device))
        model.decoder.requires_grad_(False)
    elif readout == "pca":
        n_init = cfg["pca_init_mult"] * N                       # causal: first ~10*N bins only
        cw = sample_counts(z[:n_init], C, b, np.random.default_rng(cfg["seed"] + 5))
        Cp, bp = pca_readout_init(cw, sigma=cfg["pca_smooth_sigma"])
        with torch.no_grad():
            model.decoder.decode.weight.copy_(torch.as_tensor(Cp, device=device))
            model.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1), device=device))
        model.decoder.requires_grad_(False)
    elif readout == "pca_proj_online":
        from vjf.readout import OnlineReadout
        n_init = cfg["proj_init_mult"] * N                      # cheap small init window
        cw = sample_counts(z[:n_init], C, b, np.random.default_rng(cfg["seed"] + 5))
        front = OnlineReadout(N, 2, smooth_tau=cfg["proj_tau"],
                              refresh_K=cfg["proj_refresh_K"], link="log")
        Cp, bp = front.warm_start(cw)                           # batch-PCA init of (C, b)
        with torch.no_grad():
            model.decoder.decode.weight.copy_(torch.as_tensor(Cp, device=device))
            model.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1), device=device))
        model.decoder.requires_grad_(False)                    # readout owns C, b (not SGD)
    elif readout != "learned":
        raise ValueError("readout must be 'oracle', 'learned', 'pca' or "
                         f"'pca_proj_online', got {readout!r}")
    # 'learned' leaves the default-initialized decoder trainable (expect collapse).

    # Projection encoder with a FIXED (C, b) (oracle/pca/learned): build a frozen front
    # that only forms pi_t = pinv(C)(g~(y)-b); no online PCA update or refresh.
    front_online = readout == "pca_proj_online"
    if encoder == "projection" and front is None:
        from vjf.readout import OnlineReadout
        front = OnlineReadout(N, 2, smooth_tau=cfg["proj_tau"], refresh_K=0, link="log")
        front.set_fixed(model.decoder.decode.weight.detach().cpu().numpy(),
                        model.decoder.decode.bias.detach().cpu().numpy())

    mu_all = np.zeros((T, 2), dtype=np.float32)
    log = {k: [] for k in ("step", "wall", "recon", "dynamics", "entropy", "r2")}
    win = {k: [] for k in ("recon", "dynamics", "entropy")}
    snapshots = []
    n_diverge = 0
    filter_time = 0.0  # cumulative time inside model.filter() over post-warmup bins
    n_filter = 0       # number of post-warmup filter steps timed
    bin_ms, refresh_ms, ord_ms = [], [], []  # E5: per-bin latency (all / refresh-bin / ordinary-bin)

    q = None
    t0 = time.time()
    t = 0
    for s0 in range(0, T, chunk):  # stream Poisson counts in chunks (never store full T x N)
        cc = torch.as_tensor(sample_counts(z[s0:min(s0 + chunk, T)], C, b, rng), device=device)
        for i in range(cc.shape[0]):
            warm = t < warmup_steps
            if t == warmup_steps:  # warm-up -> dynamics learning
                m = torch.as_tensor(mu_all[:warmup_steps], device=device)
                model.transition.initialize(m[1:], m[:-1], None)
                # Narrow the RBF bumps: initialize() sets width to the full state
                # radius regardless of count; smaller widths sharpen the flow and
                # greatly extend the forecast horizon.
                model.transition.velocity.feature.logwidth.data += math.log(cfg["rbf_width_scale"])
            # Projection encoder: feature -> incremental-PCA update -> recognition input.
            # All inside the timed region (part of the per-bin online cost).
            _tf = time.perf_counter()
            y_enc = None
            if front is not None:
                g = front.feature(cc[i].cpu().numpy(), update_mean=front_online)
                if front_online:
                    front.update(g)
                y_enc = torch.as_tensor(front.project(g), device=device)
            try:
                qt, loss, recon, dynamics, entropy = model.filter(
                    cc[i], None, q, sgd=True, update=True, verbose=True, warm_up=warm,
                    y_enc=y_enc,
                )
            except AssertionError:  # gaussian_loss non-finite guard tripped
                n_diverge += 1
                model.transition.logvar.data.clamp_(min=LOGVAR_FLOOR)
                q = None; mu_all[t] = mu_all[t - 1]; t += 1; continue
            refreshed = False
            if front is not None and front_online:
                refreshed = front.maybe_refresh(model.decoder, t) is not None  # Procrustes refresh
            if not warm:  # time the steady online-filtering cost (post warm-up)
                dt_bin = time.perf_counter() - _tf
                filter_time += dt_bin
                n_filter += 1
                bin_ms.append(dt_bin * 1e3)
                (refresh_ms if refreshed else ord_ms).append(dt_bin * 1e3)
            model.transition.logvar.data.clamp_(min=LOGVAR_FLOOR)  # keep long run stable
            mu = qt.mean.detach()
            if not torch.isfinite(mu).all() or mu.abs().max() > 1e3:
                n_diverge += 1
                q = None; mu_all[t] = mu_all[t - 1]; t += 1; continue
            q = qt
            mu_all[t] = mu.cpu().numpy()[0]
            for key, val in (("recon", recon), ("dynamics", dynamics), ("entropy", entropy)):
                win[key].append(float(val.detach()))

            if (t + 1) % log_every == 0:
                lo, hi = t + 1 - log_every, t + 1
                aw = max(hi - align_w, 0)                     # trailing-window fit (online)
                A, c, _ = affine_align(mu_all[aw:hi], z[aw:hi])
                r2 = r2_with(A, c, mu_all[lo:hi], z[lo:hi])
                log["step"].append(hi); log["wall"].append(time.time() - t0); log["r2"].append(r2)
                for key in ("recon", "dynamics", "entropy"):
                    log[key].append(float(np.mean(win[key])) if win[key] else float("nan"))
                    win[key].clear()

            if (t + 1) in snap_steps:  # capture learned dynamics at this stream position
                s = t + 1
                aw = max(s - align_w, 0)
                As, cs, _ = affine_align(mu_all[aw:s], z[aw:s])
                gz, az = grid_field(model, As, cs, device)
                kp = kstep_skill(model, mu_all, z, As, cs, warmup_steps, s,
                                 cfg["kpred_max"], cfg["kpred_starts"], device)
                snapshots.append({"step": s, "grid_z": gz, "arrow_z": az, "kpred": kp})
            t += 1

    # Steady-state metrics on the second half of the stream.
    half = T // 2
    A, c, _ = affine_align(mu_all[half:], z[half:])
    r2_final = r2_with(A, c, mu_all[half:], z[half:])

    # One-step prediction R^2 in z-space (mask any non-finite predictions from
    # skipped/diverged steps so a single bad value doesn't NaN the metric).
    nxt = transition_mean(model, mu_all, device)
    pred_z = nxt[:-1] @ A.T + c
    ok = np.isfinite(pred_z).all(1)
    zt = z[1:][ok]; pz = pred_z[ok]
    sse = ((zt - pz) ** 2).sum(); tss = ((zt - zt.mean(0)) ** 2).sum()
    onestep_r2 = float(1.0 - sse / (tss + 1e-12))

    # Rate reconstruction (subsample; true rate computed on the fly from z, C, b).
    sub = max(1, (T - half) // cfg["rate_sub_n"])
    idx = np.arange(half, T, sub)
    with torch.no_grad():
        rate_hat_s = np.exp(model.decoder(torch.as_tensor(mu_all[idx], device=device)).cpu().numpy())
    rates_s = rate_at(z[idx], C, b)
    rate_corr = float(np.corrcoef(rate_hat_s.ravel(), rates_s.ravel())[0, 1])

    # Small plot slices (regenerated deterministically; never the full T x N).
    raster_n = int(round(3.0 / (cfg["dt"] * cfg["stride"])))
    counts_win = sample_counts(z[half:half + raster_n], C, b, np.random.default_rng(cfg["seed"] + 2))
    nrn = int(np.argmax(rates_s.var(0)))                              # a lively neuron
    eg = slice(half, half + min(600, T - half))
    with torch.no_grad():
        rate_eg_hat = np.exp(model.decoder(torch.as_tensor(mu_all[eg], device=device)).cpu().numpy())[:, nrn]
    rate_eg_true = rate_at(z[eg], C, b)[:, nrn]

    thr = cfg["r2_threshold"]
    steps_arr, r2_arr, wall_arr = (np.array(log[k]) for k in ("step", "r2", "wall"))
    hit = np.where(r2_arr >= thr)[0]
    ttt_step = int(steps_arr[hit[0]]) if len(hit) else -1
    ttt_wall = float(wall_arr[hit[0]]) if len(hit) else -1.0

    # k-step forecast horizon from the final snapshot (max k with model R^2 >= 0).
    kp_final = snapshots[-1]["kpred"] if snapshots and snapshots[-1]["kpred"] else None
    if kp_final is not None:
        r2m = np.asarray(kp_final["r2_model"])
        below = np.where(r2m < 0)[0]
        khorizon = int(below[0]) if len(below) else int(len(r2m))
        kpred_r2_k1 = float(r2m[0]); kpred_r2_k100 = float(r2m[-1])
    else:
        khorizon, kpred_r2_k1, kpred_r2_k100 = -1, float("nan"), float("nan")

    return {
        "n_neurons": int(N),
        "snr_target": cal["snr_target"],
        "snr_realized": cal["snr_realized"],
        "label": f"{N}n ~{cal['snr_realized']:.0f}dB",
        "r2_final": r2_final,
        "onestep_r2": onestep_r2,
        "rate_corr": rate_corr,
        "kpred_horizon": khorizon,
        "kpred_r2_k1": kpred_r2_k1,
        "kpred_r2_k100": kpred_r2_k100,
        "ttt_step": ttt_step,
        "ttt_wall": ttt_wall,
        "n_diverge": n_diverge,
        "still_rising": bool(len(r2_arr) >= 5 and r2_arr[-1] > r2_arr[-5] + 0.02),
        "wall_total": time.time() - t0,
        "per_bin_filter_ms": float(filter_time / max(n_filter, 1) * 1e3),  # mean per bin
        "per_bin_p50_ms": float(np.percentile(bin_ms, 50)) if bin_ms else None,   # E5
        "per_bin_p95_ms": float(np.percentile(bin_ms, 95)) if bin_ms else None,
        "per_bin_max_ms": float(np.max(bin_ms)) if bin_ms else None,
        "per_bin_refresh_p95_ms": float(np.percentile(refresh_ms, 95)) if refresh_ms else None,
        "per_bin_ordinary_p95_ms": float(np.percentile(ord_ms, 95)) if ord_ms else None,
        "timing_sample_ms": (np.random.default_rng(0).choice(
            np.asarray(bin_ms), size=min(len(bin_ms), 8000), replace=False).tolist()
            if cfg.get("dump_timing") and bin_ms else None),  # for a violin/box (E5)
        "n_filter_steps": int(n_filter),
        "log": {k: np.asarray(v).tolist() for k, v in log.items()},
        "_mu": mu_all, "_z": z, "_A": A, "_c": c, "_model": model,
        "_counts_win": counts_win, "_C": C,
        "_rate_scatter": (rates_s.ravel()[::5], rate_hat_s.ravel()[::5]),
        "_rate_eg": (rate_eg_true, rate_eg_hat, nrn),
        "_snapshots": snapshots,
        "_kpred_final": kp_final,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(results, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(results)))

    # 1. Learning curves (R^2 + ELBO components vs stream position).
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for r, col in zip(results, colors):
        lg = r["log"]
        lab = r["label"]
        axes[0].plot(lg["step"], lg["r2"], color=col, label=lab)
        axes[1].plot(lg["step"], lg["recon"], color=col, label=lab)
        axes[2].plot(lg["step"], lg["dynamics"], color=col, label=lab)
        axes[3].plot(lg["step"], lg["entropy"], color=col, label=lab)
    for ax, ttl in zip(axes, ["latent R^2 (aligned)", "recon ELBO", "dynamics ELBO", "entropy"]):
        ax.set_xlabel("stream position (steps)"); ax.set_title(ttl); ax.legend(fontsize=8)
    axes[0].axhline(cfg["r2_threshold"], ls="--", c="grey", lw=0.8)
    axes[0].set_ylim(-0.1, 1.05)
    fig.tight_layout(); fig.savefig(RESULTS / "learning_curves.png", dpi=130); plt.close(fig)

    # 2. Phase portraits: true orbit vs aligned VJF latent (late window), per SNR.
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.6), squeeze=False)
    for ax, r in zip(axes[0], results):
        z, mu, A, c = r["_z"], r["_mu"], r["_A"], r["_c"]
        seg = slice(len(z) - 2000, len(z))
        aligned = mu[seg] @ A.T + c
        ax.plot(z[seg, 0], z[seg, 1], lw=0.8, alpha=0.7, label="true z")
        ax.plot(aligned[:, 0], aligned[:, 1], lw=0.8, alpha=0.7, label="VJF (aligned)")
        ax.set_title(f"{r['label']}  R^2={r['r2_final']:.2f}")
        ax.set_aspect("equal"); ax.legend(fontsize=8)
    fig.suptitle("Phase portrait (last 2000 steps)"); fig.tight_layout()
    fig.savefig(RESULTS / "phase_portraits.png", dpi=130); plt.close(fig)

    # 3. Learned vector field EVOLUTION: rows = conditions, cols = snapshot times.
    #    Shows the rotational flow emerging as VJF learns online.
    nsnap = max(len(r["_snapshots"]) for r in results)
    fig, axes = plt.subplots(len(results), nsnap,
                             figsize=(3.2 * nsnap, 3.2 * len(results)), squeeze=False)
    for i, r in enumerate(results):
        z = r["_z"]
        for j, snap in enumerate(r["_snapshots"]):
            ax = axes[i][j]
            gz, az = snap["grid_z"], snap["arrow_z"]
            ax.plot(z[-2000:, 0], z[-2000:, 1], lw=0.5, alpha=0.4, color="k")
            # quiver autoscale divides by the field's max magnitude; an early snapshot
            # whose learned flow is still ~identity (velocity ~ 0 everywhere) makes that
            # scale 0 -> divide-by-zero / 0*inf NaN. Nothing to draw, so skip it.
            if float(np.max(np.hypot(az[:, 0], az[:, 1]))) > 1e-8:
                ax.quiver(gz[:, 0], gz[:, 1], az[:, 0], az[:, 1], color="C0", alpha=0.85, angles="xy")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t={snap['step']}")
            if j == 0:
                ax.set_ylabel(r["label"], fontsize=9)
    fig.suptitle("Learned velocity field over training (aligned to z-space)")
    fig.tight_layout(); fig.savefig(RESULTS / "vector_field_evolution.png", dpi=130); plt.close(fig)

    # 4. Forecast rollout from a late state vs true orbit.
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.6), squeeze=False)
    for ax, r in zip(axes[0], results):
        z, mu, A, c, model = r["_z"], r["_mu"], r["_A"], r["_c"], r["_model"]
        dev = next(model.parameters()).device
        # Deterministic free-run of the learned mean flow (no RBF sampling noise).
        with torch.no_grad():
            st = torch.as_tensor(mu[len(mu) // 2], device=dev).reshape(1, 2)
            traj = [st]
            for _ in range(1000):
                st = _mean(model.transition(st, None, sampling=False))
                traj.append(st)
            roll = torch.cat(traj, 0).cpu().numpy()
        roll_z = roll @ A.T + c
        ax.plot(z[-2000:, 0], z[-2000:, 1], lw=0.8, alpha=0.5, label="true z")
        ax.plot(roll_z[:, 0], roll_z[:, 1], lw=0.8, alpha=0.8, label="forecast")
        ax.set_title(f"{r['label']} forecast"); ax.set_aspect("equal"); ax.legend(fontsize=8)
    fig.suptitle("Free-run forecast of learned dynamics"); fig.tight_layout()
    fig.savefig(RESULTS / "forecast.png", dpi=130); plt.close(fig)

    # 4b. Single-trial forecast examples: from several marked start states, free-run
    #     the learned dynamics K steps (dashed) vs the true trajectory (solid).
    K = cfg["kpred_max"]
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.8), squeeze=False)
    for ax, r in zip(axes[0], results):
        z, mu, A, c, model = r["_z"], r["_mu"], r["_A"], r["_c"], r["_model"]
        dev = next(model.parameters()).device
        half = len(z) // 2
        starts = np.linspace(half, len(z) - K - 2, 6).astype(int)
        ax.plot(z[half:half + 4000, 0], z[half:half + 4000, 1], lw=0.4, alpha=0.25, color="grey")
        cols = plt.cm.turbo(np.linspace(0.08, 0.92, len(starts)))
        for s, col in zip(starts, cols):
            with torch.no_grad():
                st = torch.as_tensor(mu[s], device=dev).reshape(1, 2)
                traj = [st]
                for _ in range(K):
                    st = _mean(model.transition(st, None, sampling=False))
                    traj.append(st)
                roll_z = torch.cat(traj, 0).cpu().numpy() @ A.T + c
            true_seg = z[s:s + K + 1]
            ax.plot(true_seg[:, 0], true_seg[:, 1], color=col, lw=1.3, alpha=0.85)
            ax.plot(roll_z[:, 0], roll_z[:, 1], color=col, lw=1.1, ls="--", alpha=0.95)
            ax.plot(true_seg[0, 0], true_seg[0, 1], "o", color=col, ms=4)
        ax.set_title(f"{r['label']}  ({K}-step)"); ax.set_aspect("equal")
    fig.suptitle("Single-trial forecast examples (solid=true, dashed=VJF free-run, o=start)")
    fig.tight_layout(); fig.savefig(RESULTS / "forecast_examples.png", dpi=130); plt.close(fig)

    # 5. Rate reconstruction (predicted vs true, example neuron trace + scatter).
    fig, axes = plt.subplots(2, len(results), figsize=(5 * len(results), 7), squeeze=False)
    for j, r in enumerate(results):
        eg_true, eg_hat, nrn = r["_rate_eg"]
        axes[0][j].plot(eg_true, lw=0.8, label="true rate")
        axes[0][j].plot(eg_hat, lw=0.8, alpha=0.8, label="VJF rate")
        axes[0][j].set_title(f"{r['label']} neuron {nrn}"); axes[0][j].legend(fontsize=8)
        st, sh = r["_rate_scatter"]
        axes[1][j].scatter(st, sh, s=2, alpha=0.3)
        mx = float(st.max()) if st.size else 1.0
        axes[1][j].plot([0, mx], [0, mx], "k--", lw=0.8)
        axes[1][j].set_xlabel("true rate"); axes[1][j].set_ylabel("VJF rate")
        axes[1][j].set_title(f"corr={r['rate_corr']:.2f}")
    fig.suptitle("Rate reconstruction (steady state)"); fig.tight_layout()
    fig.savefig(RESULTS / "rate_reconstruction.png", dpi=130); plt.close(fig)

    # 6. Convergence summary across conditions.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(results))
    axes[0].bar(x, [r["r2_final"] for r in results], color=colors)
    axes[0].set_title("final latent R^2"); axes[0].axhline(cfg["r2_threshold"], ls="--", c="grey")
    axes[1].bar(x, [r["ttt_step"] if r["ttt_step"] > 0 else 0 for r in results], color=colors)
    axes[1].set_title(f"steps to R^2>={cfg['r2_threshold']}")
    axes[2].bar(x, [r["rate_corr"] for r in results], color=colors)
    axes[2].set_title("rate corr")
    axes[3].bar(x, [max(0, r["kpred_horizon"]) for r in results], color=colors)
    axes[3].set_title("k-step horizon (R^2>=0)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "convergence_summary.png", dpi=130); plt.close(fig)

    # 7. Spike raster, 3 s steady-state window, per condition (ticks = spike bins).
    bin_s = cfg["dt"] * cfg["stride"]
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.8), squeeze=False)
    for ax, r in zip(axes[0], results):
        counts, C = r["_counts_win"], r["_C"]
        order = np.argsort(np.arctan2(C[:, 1], C[:, 0]))  # sort by preferred phase
        win_c = counts[:, order].T                          # (N, n_bins)
        N, nb = win_c.shape
        neuron_idx, bin_idx = np.nonzero(win_c)         # spike bins
        ax.scatter(bin_idx * bin_s, neuron_idx, s=4, c="k", alpha=0.6, marker="|", linewidths=0.5)
        ax.set_xlabel("time (s)"); ax.set_ylabel("neuron (sorted by preferred phase)")
        ax.set_title(r["label"])
        ax.set_xlim(0, nb * bin_s); ax.set_ylim(-1, N)
    fig.suptitle(f"Spike raster (3 s, {bin_s * 1e3:.0f} ms bins, steady state)")
    fig.tight_layout(); fig.savefig(RESULTS / "spike_raster.png", dpi=130); plt.close(fig)

    # 8. k-step forecast accuracy vs horizon (final model). Solid = VJF free-run,
    #    dotted = zero-flow baseline (predict z_{t+k}=z_t). Both degrade with k;
    #    the gap is VJF's advantage. (Accuracy DECREASES with horizon, as expected.)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for r, col in zip(results, colors):
        kp = r["_kpred_final"]
        if kp is None:
            continue
        ks = np.arange(1, len(kp["r2_model"]) + 1)
        axes[0].plot(ks, kp["r2_model"], color=col, label=r["label"])
        axes[0].plot(ks, kp["r2_base"], color=col, ls=":", alpha=0.6)
        axes[1].plot(ks, np.sqrt(kp["mse_model"]), color=col, label=r["label"])
        axes[1].plot(ks, np.sqrt(kp["mse_base"]), color=col, ls=":", alpha=0.6)
    axes[0].set_xlabel("forecast horizon k (steps)")
    axes[0].set_ylabel("k-step forecast R^2")
    axes[0].set_ylim(-0.5, 1.05); axes[0].axhline(0, c="grey", lw=0.6)
    axes[0].set_title("forecast R^2 (solid=VJF, dotted=zero-flow baseline)"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("forecast horizon k (steps)")
    axes[1].set_ylabel("k-step forecast RMSE (z units)")
    axes[1].set_title("forecast error (solid=VJF, dotted=zero-flow baseline)"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "kstep_prediction.png", dpi=130); plt.close(fig)

    # 9. k-step forecast R^2 vs horizon at each snapshot time (improves over
    #    training); final zero-flow baseline shown dotted for reference.
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.4), squeeze=False)
    for ax, r in zip(axes[0], results):
        snaps = [s for s in r["_snapshots"] if s["kpred"] is not None]
        scol = plt.cm.plasma(np.linspace(0.1, 0.85, max(len(snaps), 1)))
        for s, col in zip(snaps, scol):
            kp = s["kpred"]; ks = np.arange(1, len(kp["r2_model"]) + 1)
            ax.plot(ks, kp["r2_model"], color=col, label=f"t={s['step']}")
        if snaps:
            kp = snaps[-1]["kpred"]; ks = np.arange(1, len(kp["r2_base"]) + 1)
            ax.plot(ks, kp["r2_base"], color="grey", ls=":", lw=1, label="zero-flow")
        ax.axhline(0, c="grey", lw=0.6); ax.set_ylim(-0.5, 1.05)
        ax.set_xlabel("forecast horizon k (steps)"); ax.set_ylabel("k-step forecast R^2")
        ax.set_title(r["label"]); ax.legend(fontsize=7)
    fig.suptitle("k-step forecast accuracy over training")
    fig.tight_layout(); fig.savefig(RESULTS / "kstep_evolution.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
def provenance(cfg):
    def _git(args, cwd):
        try:
            return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
        except Exception:
            return "unknown"
    info = {
        "vjf_commit": _git(["rev-parse", "HEAD"], HERE),
        "vjf_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], HERE),
        "vjf_dirty": _git(["status", "--porcelain"], HERE) != "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
    }
    # CPU model (Linux /proc/cpuinfo, else platform.processor()).
    cpu_model = platform.processor()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    info["cpu_model"] = cpu_model
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        info["scipy"] = "unknown"
    try:
        import neurofisherSNR
        info["neurofisherSNR"] = getattr(neurofisherSNR, "__version__", "unknown")
        info["neurofisherSNR_path"] = neurofisherSNR.__file__
    except Exception:
        info["neurofisherSNR"] = "unknown"
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--t-eff", type=int, default=200_000)  # 5 ms bins, 1000 s
    ap.add_argument("--readout", default=None,
                    choices=["oracle", "learned", "pca", "pca_proj_online"],
                    help="override cfg['readout']")
    ap.add_argument("--encoder", default=None, choices=["spikes", "projection"],
                    help="override cfg['encoder'] (recognition input)")
    ap.add_argument("--tag", default=None,
                    help="write results to results/<tag>/ (for multi-mode sweeps)")
    ap.add_argument("--proj-tau", type=float, default=None,
                    help="override cfg['proj_tau']; tau=1 disables the causal EMA smoothing")
    ap.add_argument("--refresh-K", type=int, default=None,
                    help="override cfg['proj_refresh_K'] (readout refresh interval; E2 K sweep)")
    ap.add_argument("--snr", type=float, default=None,
                    help="run a single SNR condition (dB); picks the matching population size")
    ap.add_argument("--seed", type=int, default=None, help="override cfg['seed']")
    ap.add_argument("--dump-timing", action="store_true",
                    help="save a downsampled per-bin latency sample (for a violin/box plot)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    args = ap.parse_args()
    global RESULTS
    if args.tag:
        RESULTS = HERE / "results" / args.tag

    cfg = {
        # SNR is set by population size at biological firing rates (mean ~20 Hz,
        # peak <=~100 Hz at 5 ms bins). (n_neurons, target_snr_db) per condition;
        # realized SNR ~ 3 / 6 / 9 dB. Capped at 300 neurons: under oracle readout
        # VJF's online filter is numerically stable up to ~300, diverges at ~800.
        # (n_neurons, target_snr_db); extended down to the neural regime (-3, 0 dB) via
        # smaller populations. Realized SNR ~ target.
        "conditions": [(15, -3.0), (30, 0.0), (50, 3.0), (150, 6.0), (250, 8.0)],
        "t_eff": 2000 if args.quick else args.t_eff,
        "stride": 1,
        "dt": 5e-3,                       # 5 ms bins
        "angular_velocity": 30.0,         # ~42 steps/cycle: strong rotational signal,
                                          # safe from aliasing (omega*dt=0.15 rad/step)
        "target_mean_rate": 0.1,          # 20 Hz at 5 ms bins
        "target_max_rate": 0.5,           # 100 Hz peak
        "obs_chunk": 20000,               # streaming Poisson chunk size
        "n_rbf": 100,                     # richer flow basis (sharper field); srrls handles it
        "rbf_width_scale": 0.5,           # narrow the RBF bumps vs the default (=state radius)
        "transition_flow": "srrls",       # square-root RLS: stable over long streams AND
                                          # RLS-speed convergence (forecast horizon ~k=100).
                                          # 'rls' explodes at long T; 'sgd' is stable but slow.
        "hidden_sizes": [100, 100],
        "lr": 1e-3,
        "warmup_frac": 0.15,
        "warmup_cap": 20000,              # cap warm-up steps for very long streams
        "readout": "pca",                 # 'oracle' | 'learned' | 'pca' (causal warm-start, frozen)
        "encoder": "spikes",              # 'spikes' | 'projection' (recognition input)
        "pca_init_mult": 60,             # PCA warm-start window = first pca_init_mult*N bins (causal).
                                          # 60-100*N gives a stable ~oracle C across SNR; 10*N is
                                          # high-variance at low SNR (Phase-0 subspace-angle study).
        "pca_smooth_sigma": 8.0,         # Gaussian smoothing (bins) for the PCA warm-start
        "proj_init_mult": 10,            # pca_proj_online: cheap small init window (first 10*N bins);
                                          # the online estimator recovers the asymptotic C from here
        "proj_tau": 8.0,                 # causal EMA timescale (bins) for the link-matched feature
        "proj_refresh_K": 1000,          # refresh decoder (C,b) from the online PCA every K bins
        "log_every": 50 if args.quick else 1000,
        "align_window": 5000,            # trailing window for online R^2 alignment
        "rate_sub_n": 5000,              # subsample size for rate-recon metrics
        "kpred_max": 100,                # k-step forecast horizon (steps)
        "kpred_starts": 500,             # number of forecast start points
        "snapshots": [0.1, 0.3, 0.6, 1.0],   # capture learned dynamics at these stream fractions
        "r2_threshold": 0.8,
        "seed": 20260602,
    }
    if args.readout is not None:
        cfg["readout"] = args.readout
    if args.encoder is not None:
        cfg["encoder"] = args.encoder
    if args.proj_tau is not None:
        cfg["proj_tau"] = args.proj_tau
    if args.refresh_K is not None:
        cfg["proj_refresh_K"] = args.refresh_K
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["dump_timing"] = bool(args.dump_timing)
    if args.snr is not None:                       # single SNR condition (for sharding sweeps)
        cfg["conditions"] = [c for c in cfg["conditions"] if abs(c[1] - args.snr) < 1e-6]
        if not cfg["conditions"]:
            raise SystemExit(f"--snr {args.snr} not in conditions")
    if args.quick:
        cfg["conditions"] = [(50, 3.0), (150, 6.0)]
        cfg["warmup_cap"] = 300
        cfg["align_window"] = 400
        cfg["obs_chunk"] = 1000

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_default_device("cuda")  # VJF creates some tensors via default device

    torch.set_default_dtype(torch.float32)
    print(f"device={device}  t_eff={cfg['t_eff']}  conditions={cfg['conditions']}", flush=True)

    # Shared latent across conditions: same dynamics, different observation populations.
    from lc_data import generate_latent, calibrate_poisson
    z = generate_latent(
        t_eff=cfg["t_eff"], stride=cfg["stride"], dt=cfg["dt"],
        angular_velocity=cfg["angular_velocity"], seed=cfg["seed"],
    )
    np.savez_compressed(RESULTS / "latent.npz", z=z)

    results = []
    for n_neurons, snr in cfg["conditions"]:
        print(f"\n=== {n_neurons} neurons (target {snr:.0f} dB) : calibrating ===", flush=True)
        cal = calibrate_poisson(
            z, n_neurons=n_neurons, snr_db=snr,
            target_mean_rate=cfg["target_mean_rate"], target_max_rate=cfg["target_max_rate"],
            seed=cfg["seed"],
        )
        print(f"    realized SNR={cal['snr_realized']:.1f} dB", flush=True)
        np.savez_compressed(RESULTS / f"calib_n{n_neurons}.npz", C=cal["C"], b=cal["b"])
        r = run_condition(z, cal, n_neurons, cfg, device)
        print(f"    R^2_final={r['r2_final']:.3f}  onestep_R^2={r['onestep_r2']:.3f}  "
              f"rate_corr={r['rate_corr']:.3f}  kpred_horizon={r['kpred_horizon']}  "
              f"diverge={r['n_diverge']}  wall={r['wall_total']:.0f}s  "
              f"per_bin_filter={r['per_bin_filter_ms']:.3f}ms", flush=True)
        results.append(r)

    make_plots(results, cfg)

    summary = {
        "config": cfg,
        "provenance": provenance(cfg),
        "conditions": [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in results
        ],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote results to", RESULTS, flush=True)


if __name__ == "__main__":
    main()
