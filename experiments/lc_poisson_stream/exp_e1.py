"""E1 + E0: causal test of subspace drift (C3) with direct drift measurement.

Arms (high-SNR condition, projection encoder, square-root-RLS flow), one seed per run:
  proj_oracle          - oracle C, fixed (the ceiling)
  online_base          - online readout, Procrustes refresh every K, NO flow tracking (current)
  online_track         - online readout, refresh every K, WITH subspace-tracking flow rotation
  oracle_imposed       - oracle C, but a known rotation Q imposed every K, NO tracking (degrade)
  oracle_imposed_track - oracle C + imposed Q every K, WITH tracking (should be invariant: the fix)
  frozen_pca           - frozen batch-PCA readout (no refresh)
  freeze_after         - online readout, refresh until step S then stop writing C

Per run we log the E0 drift series (principal angles, per-refresh rotation) and the standard
recovery/forecast metrics, and save summary.json. Usage:
    python exp_e1.py --arm online_track --seed 20260605 [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import affine_align, r2_with, _mean, kstep_skill, transition_mean, pca_readout_init  # noqa: E402
from lc_data import generate_latent, calibrate_poisson, sample_counts, rate_at  # noqa: E402
from vjf.model import VJF  # noqa: E402
from vjf.readout import OnlineReadout  # noqa: E402

RESULTS = HERE / "results_e1"
LOGVAR_FLOOR = math.log(1e-6)
ONLINE_ARMS = {"online_base", "online_track", "freeze_after"}
IMPOSED_ARMS = {"oracle_imposed", "oracle_imposed_track"}
TRACK_ARMS = {"online_track", "oracle_imposed_track"}


def _rot(theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]], dtype="float32")


def run_arm(arm, z, C, b, N, cfg, seed, device):
    T = z.shape[0]
    warmup = min(int(cfg["warmup_frac"] * T), cfg["warmup_cap"])
    K = cfg["refresh_K"]
    online = arm in ONLINE_ARMS
    imposed = arm in IMPOSED_ARMS
    track = arm in TRACK_ARMS
    rng = np.random.default_rng(seed + 1)

    torch.manual_seed(seed)
    model = VJF.make_model(ydim=N, xdim=2, udim=0, n_rbf=cfg["n_rbf"],
                           hidden_sizes=cfg["hidden_sizes"], likelihood="poisson",
                           lr=cfg["lr"], lr_decay=1.0, transition_flow="srrls",
                           encoder="projection").to(device)
    dec = model.decoder

    front = OnlineReadout(N, 2, smooth_tau=cfg["proj_tau"], refresh_K=K, link="log")
    if online:
        n_init = cfg["proj_init_mult"] * N
        Cp, bp = front.warm_start(sample_counts(z[:n_init], C, b, np.random.default_rng(seed + 5)))
    elif arm == "frozen_pca":
        n_init = cfg["pca_init_mult"] * N
        Cp, bp = pca_readout_init(sample_counts(z[:n_init], C, b, np.random.default_rng(seed + 5)),
                                  sigma=cfg["pca_smooth_sigma"])
        front.set_fixed(Cp, bp)
    else:  # proj_oracle, oracle_imposed(_track): true readout
        Cp, bp = C.astype("float32"), b.reshape(-1).astype("float32")
        front.set_fixed(Cp, bp)
    with torch.no_grad():
        dec.decode.weight.copy_(torch.as_tensor(Cp, device=device))
        dec.decode.bias.copy_(torch.as_tensor(np.asarray(bp).reshape(-1), device=device))
    dec.requires_grad_(False)

    Q_imp = _rot(cfg["imposed_angle_deg"])
    S = int(cfg["freeze_after_frac"] * T)
    mu_all = np.zeros((T, 2), dtype=np.float32)
    drift, refresh_steps = [], []
    per_bin_ms = []
    n_diverge = 0
    q = None
    chunk = cfg["obs_chunk"]
    t0 = time.time()
    t = 0
    for s0 in range(0, T, chunk):
        cc = torch.as_tensor(sample_counts(z[s0:min(s0 + chunk, T)], C, b, rng), device=device)
        for i in range(cc.shape[0]):
            warm = t < warmup
            if t == warmup:
                m = torch.as_tensor(mu_all[:warmup], device=device)
                model.transition.initialize(m[1:], m[:-1], None)
                model.transition.velocity.feature.logwidth.data += math.log(cfg["rbf_width_scale"])
            _tf = time.perf_counter()
            g = front.feature(cc[i].cpu().numpy(), update_mean=online)
            if online:
                front.update(g)
            y_enc = torch.as_tensor(front.project(g), device=device)
            try:
                qt, loss = model.filter(cc[i], None, q, sgd=True, update=True, warm_up=warm, y_enc=y_enc)
            except AssertionError:
                n_diverge += 1
                model.transition.logvar.data.clamp_(min=LOGVAR_FLOOR)
                q = None; mu_all[t] = mu_all[t - 1]; t += 1; continue
            if not warm:
                per_bin_ms.append((time.perf_counter() - _tf) * 1e3)
            model.transition.logvar.data.clamp_(min=LOGVAR_FLOOR)
            mu = qt.mean.detach()
            if not torch.isfinite(mu).all() or mu.abs().max() > 1e3:
                n_diverge += 1; q = None; mu_all[t] = mu_all[t - 1]; t += 1; continue
            q = qt
            mu_all[t] = mu.cpu().numpy()[0]
            # --- refresh / imposed-rotation behavior by arm ---
            if not warm and t > 0 and t % K == 0:
                if online and not (arm == "freeze_after" and t >= S):
                    md = front.maybe_refresh(dec, t, transition=model.transition if track else None,
                                             track_subspace=track, oracle_C=C)
                    if md is not None:
                        drift.append(md); refresh_steps.append(t)
                elif imposed:
                    front.impose_rotation(dec, Q_imp, transition=model.transition if track else None,
                                          track_subspace=track)
                    refresh_steps.append(t)
                    drift.append({"step": int(t), "refresh_rot_deg": float(cfg["imposed_angle_deg"]),
                                  "angle_oracle_deg": None})
            t += 1

    # --- metrics ---
    half = T // 2
    A, c, _ = affine_align(mu_all[half:], z[half:])
    r2_final = r2_with(A, c, mu_all[half:], z[half:])
    nxt = transition_mean(model, mu_all, device)
    pred_z = nxt[:-1] @ A.T + c
    ok = np.isfinite(pred_z).all(1)
    zt, pz = z[1:][ok], pred_z[ok]
    onestep_r2 = float(1 - ((zt - pz) ** 2).sum() / (((zt - zt.mean(0)) ** 2).sum() + 1e-12))
    # Drift-sensitive metric: fix the latent->z alignment on an EARLY post-warmup window and score
    # one-step prediction on the SECOND HALF with it. If the latent frame drifts, this early-fixed
    # alignment no longer fits the late data, so the gap (onestep_r2 - onestep_r2_fixed) exposes
    # frame motion that the re-fit (own-frame) onestep_r2 absorbs. (cf. affine-R2 masking, E0.)
    ew = slice(warmup, min(warmup + 5000, half))
    Ae, ce, _ = affine_align(mu_all[ew], z[ew])
    pe = (nxt[:-1] @ Ae.T + ce)[ok]
    onestep_r2_fixed = float(1 - ((zt - pe) ** 2).sum() / (((zt - zt.mean(0)) ** 2).sum() + 1e-12))
    kp = kstep_skill(model, mu_all, z, A, c, warmup, T, cfg["kpred_max"], cfg["kpred_starts"], device)
    pbm = np.array(per_bin_ms)
    return {
        "arm": arm, "seed": int(seed), "snr_realized": cal_snr,
        "r2_final": float(r2_final), "onestep_r2": onestep_r2,
        "onestep_r2_fixed": onestep_r2_fixed,
        "kpred_r2": kp["r2_model"].tolist() if kp else None,
        "kpred_base": kp["r2_base"].tolist() if kp else None,
        "drift": drift, "refresh_steps": refresh_steps,
        "n_refresh": len(refresh_steps), "n_diverge": int(n_diverge),
        "per_bin_ms_p50": float(np.percentile(pbm, 50)) if pbm.size else None,
        "per_bin_ms_p95": float(np.percentile(pbm, 95)) if pbm.size else None,
        "wall_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["proj_oracle", "online_base", "online_track", "oracle_imposed",
                             "oracle_imposed_track", "frozen_pca", "freeze_after"])
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--t-eff", type=int, default=200_000)
    ap.add_argument("--n-neurons", type=int, default=250)
    ap.add_argument("--snr-db", type=float, default=8.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = {
        "warmup_frac": 0.15, "warmup_cap": 20000, "obs_chunk": 20000,
        "n_rbf": 100, "rbf_width_scale": 0.5, "hidden_sizes": [100, 100], "lr": 1e-3,
        "proj_tau": 8.0, "proj_init_mult": 10, "pca_init_mult": 60, "pca_smooth_sigma": 8.0,
        "refresh_K": 1000, "imposed_angle_deg": 3.0, "freeze_after_frac": 0.4,
        "kpred_max": 200, "kpred_starts": 500,
        "dt": 5e-3, "stride": 1, "angular_velocity": 30.0,
    }
    t_eff = 2000 if args.quick else args.t_eff
    if args.quick:
        cfg.update(warmup_cap=300, obs_chunk=1000, refresh_K=200, kpred_max=50)

    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    z = generate_latent(t_eff=t_eff, stride=cfg["stride"], dt=cfg["dt"],
                        angular_velocity=cfg["angular_velocity"], seed=args.seed)
    cal = calibrate_poisson(z, n_neurons=args.n_neurons, snr_db=args.snr_db,
                            target_mean_rate=0.1, target_max_rate=0.5, seed=args.seed)
    global cal_snr
    cal_snr = cal["snr_realized"]
    res = run_arm(args.arm, z, cal["C"], cal["b"], args.n_neurons, cfg, args.seed, device)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.arm}_n{args.n_neurons}_seed{args.seed}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[{args.arm} seed={args.seed}] r2={res['r2_final']:.3f} onestep={res['onestep_r2']:.3f} "
          f"k1={res['kpred_r2'][0] if res['kpred_r2'] else float('nan'):.3f} "
          f"n_refresh={res['n_refresh']} diverge={res['n_diverge']} "
          f"p95={res['per_bin_ms_p95']} wall={res['wall_s']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
