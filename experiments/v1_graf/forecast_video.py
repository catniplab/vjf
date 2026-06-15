"""Beautiful 3D rotating video of the BEST single-dir sVJF free-run forecast.

Retrains the search-best config (loaded from best_single.json) on the 30 train trials,
then on the best-forecasting HELD-OUT TEST trial shows, in one rotating 3D frame:
  - the FILTERED latent trajectory (inferred WITH observations), and
  - the autonomous FREE-RUN FORECAST launched at t0 = end of the first grating cycle
    (no observations after t0).
The 5-D latent is projected to 3D by PCA of the filtered path (the latent lives on a
low-D cycle). Also records per-bin VJF wall-clock (online-learning step + frozen-inference
step) -- the per-step timing the M1 report tracked -- and writes a static money-shot PNG.

Run: uv run python -m experiments.v1_graf.forecast_video
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval, SEED
from experiments.v1_graf.eval import (
    frozen_dynamics, forecast_reconstruction_deviance, forecast_skill_summary)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
VID = os.path.join(HERE, "report_m1", "videos")
BEST_JSON = os.path.join(HERE, "..", "..", "gcp_runs",
                         "graf-search-single-xl-20260614-164113", "results", "best_single.json")
BINS_PER_CYCLE = 16                       # 16 bins/cycle (10 ms bins)
T0 = BINS_PER_CYCLE                        # forecast launches after the first cycle

# Config presets. "metricbest" = the search winner (largest basis; overfits the flow into a
# jagged free-run). "l3"/"l4" = lower-capacity configs whose free-run is a clean limit cycle
# (more interpretable dynamics). All are top-of-ranking, scale-fixed CCIPCA + Adam + grow.
PRESETS = {
    "metricbest": {"latent_dim": 5, "max_rbf": 2400, "epochs": 100, "lr": 0.001,
                   "rbf_base": 100, "dyn_noise": 0.3, "dyn_noise_decay": 0.97, "dyn_noise_fit_ref": 0.3},
    "l3": {"latent_dim": 3, "max_rbf": None, "epochs": 100, "lr": 3e-4,
           "rbf_base": 100, "dyn_noise": 0.3, "dyn_noise_decay": 0.97, "dyn_noise_fit_ref": 0.3},
    "l4": {"latent_dim": 4, "max_rbf": 1600, "epochs": 100, "lr": 0.001,
           "rbf_base": 100, "dyn_noise": 0.3, "dyn_noise_decay": 0.97, "dyn_noise_fit_ref": 0.3},
}


def load_best():
    """The search-best config (fallback to the known xl winner if the JSON is absent)."""
    try:
        with open(os.path.abspath(BEST_JSON)) as fh:
            return json.load(fh)["best"]["config"]
    except Exception:
        return PRESETS["metricbest"]


@torch.no_grad()
def filtered_path(model, ro, tc):
    """Frozen filtered latent path (T, L) for one trial + per-bin frozen-inference timing."""
    nu0 = ro.nu.copy()
    means, dt = [], []
    with frozen_dynamics(model):
        ro.nu = nu0.copy()
        q = None
        for t in range(tc.shape[0]):
            g = ro.feature(tc[t], update_mean=False)
            x_enc = torch.as_tensor(ro.project(g))
            t1 = time.perf_counter()
            qt, *_ = model.filter(tc[t], None, q, sgd=False, update=False, y_enc=x_enc)
            dt.append(time.perf_counter() - t1)
            q = qt
            means.append(qt.mean.detach().cpu().numpy()[0])
    ro.nu = nu0
    return np.asarray(means), np.asarray(dt)


@torch.no_grad()
def online_step_timing(model, ro, tc):
    """Per-bin wall-clock of the FULL online (learning) VJF step -- the real-time primitive
    the M1 report timed. Runs on a copy-free trial with sgd+update ON (no readout refresh)."""
    nu0 = ro.nu.copy()
    ro.nu = nu0.copy()
    q, dt = None, []
    for t in range(tc.shape[0]):
        g = ro.feature(tc[t], update_mean=False)
        x_enc = torch.as_tensor(ro.project(g))
        t1 = time.perf_counter()
        qt, *_ = model.filter(tc[t], None, q, sgd=True, update=True, y_enc=x_enc)
        dt.append(time.perf_counter() - t1)
        q = qt
    ro.nu = nu0
    return np.asarray(dt)


def main():
    os.makedirs(VID, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    cfg = load_best()
    print(f"[video] best config: {cfg}")

    # 1. Retrain the best config on the 30 train trials; eval on the 10 reserved TEST trials.
    data = prepare_single_dir_data(direction=225.0, n_val=10)
    test_ybar = float(data["test_counts"].mean())
    res = _train_eval(
        data["train_trials"], data["test_trials"], data["test_counts"], test_ybar,
        epochs=cfg["epochs"], latent_dim=cfg["latent_dim"], N=data["N"], grow=True,
        grow_weight_init="residual", flow="sgd", optimizer="adam", lr=cfg["lr"],
        rbf_base=cfg.get("rbf_base", 100), max_rbf=cfg.get("max_rbf"),
        dyn_noise=cfg["dyn_noise"], dyn_noise_decay=cfg["dyn_noise_decay"],
        dyn_noise_fit_ref=cfg["dyn_noise_fit_ref"], smooth_lambda=cfg.get("smooth_lambda", 0.0),
        psth_counts=data["psth_counts"], return_model=True)
    model, ro = res["model"], res["ro"]
    L = cfg["latent_dim"]
    print(f"[video] trained: test PLL {res['pll']:.3f} (ceiling {data['pll_psth']:.3f}), "
          f"S_persist {res['fc_weighted_persist_skill']:+.4f}, n_basis {res['n_basis']}")

    # 2. Per-bin VJF timing (online-learning step + frozen-inference step), like the M1 report.
    #    The online step mutates the model (sgd+update), so time a DEEPCOPY on a TRAIN trial
    #    (never the held-out model/test trials used for the figure); frozen timing is a no-op.
    on = online_step_timing(copy.deepcopy(model), copy.deepcopy(ro), data["train_trials"][0]) * 1e3
    _, fr = filtered_path(model, ro, data["test_trials"][0])
    fr = fr * 1e3
    timing = {"online_median_ms": float(np.median(on)), "online_p95_ms": float(np.percentile(on, 95)),
              "frozen_median_ms": float(np.median(fr)), "frozen_p95_ms": float(np.percentile(fr, 95)),
              "bin_ms": 10.0, "n_basis": int(res["n_basis"]), "latent_dim": L}
    print(f"[video] per-bin timing: online median {timing['online_median_ms']:.2f} ms "
          f"(p95 {timing['online_p95_ms']:.2f}); frozen median {timing['frozen_median_ms']:.2f} ms "
          f"(p95 {timing['frozen_p95_ms']:.2f}) | n_basis {res['n_basis']}")

    # 3. Pick the best-forecasting held-out TEST trial (highest weighted persist skill at t0).
    starts = [T0]
    per_trial = []
    for i, tc in enumerate(data["test_trials"]):
        dev = forecast_reconstruction_deviance(model, ro, tc, data["psth_counts"], starts, (8, 16, 32))
        s = forecast_skill_summary([dev])["weighted_persist_skill"]
        per_trial.append((s, i))
    per_trial.sort(reverse=True)
    best_s, best_i = per_trial[0]
    print(f"[video] best test trial = #{best_i} (S_persist {best_s:+.4f} at t0={T0})")
    tc = data["test_trials"][best_i]
    T = tc.shape[0]

    # 4. Filtered path + free-run forecast from t0.
    xfilt, _ = filtered_path(model, ro, tc)                    # (T, L)
    with torch.no_grad():
        x, _ = model.forecast(torch.as_tensor(xfilt[T0][None].astype(np.float32)), n_step=T - 1 - T0)
    xfc = x.detach().cpu().numpy()[:, 0, :]                     # (T-T0, L); xfc[0] == xfilt[T0]

    # 4b. High-SNR reference: trial-averaged filtered latent over ALL trials of this direction
    #     (stimulus-locked, shared frozen readout -> averaging cancels per-trial noise and
    #     reveals the clean underlying cycle: "how smooth it could be").
    all_trials = list(data["train_trials"]) + list(data["val_trials"]) + list(data["test_trials"])
    xbar = np.stack([filtered_path(model, ro, t)[0] for t in all_trials], 0).mean(0)  # (T, L)

    # 5. Project to 3D by PCA of the SMOOTH trial-average (the clean-cycle frame).
    ctr = xbar.mean(0)
    _, _, vt = np.linalg.svd(xbar - ctr, full_matrices=False)
    B = vt[:3]
    pf = (xfilt - ctr) @ B.T                                    # single-trial filtered, (T,3)
    pc = (xfc - ctr) @ B.T                                      # free-run forecast, (T-T0,3)
    pbar = (xbar - ctr) @ B.T                                   # trial-average reference, (T,3)
    render(pf, pc, pbar, T0, cfg, res, data, best_i, timing)

    with open(os.path.join(VID, "forecast_3d_timing.json"), "w") as fh:
        json.dump({"config": cfg, "test_trial": best_i, "best_trial_S_persist": best_s,
                   "test_pll": res["pll"], "pll_ceiling": data["pll_psth"], **timing}, fh, indent=2)
    print(f"[video] done -> {VID}/forecast_3d.mp4 (+ .png money shot, timing.json)")


def render(pf, pc, pbar, t0, cfg, res, data, trial_i, timing):
    """3D rotating animation: a gold trial-AVERAGE reference loop (high-SNR, "how smooth it
    could be") stays drawn throughout; the single-trial filtered latent (teal) advances ->
    at t0 its held-out continuation goes faint and the crimson free-run forecast unfolds;
    camera rotates."""
    plt.style.use("dark_background")
    allpts = np.vstack([pf, pc, pbar])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.08 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad
    T = pf.shape[0]
    TAIL = 80                                                  # extra pure-rotation frames at the end
    nframes = T + TAIL
    fig = plt.figure(figsize=(7.2, 7.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    teal, crim, ghost, gold = "#27d3c4", "#ff5470", "#5a6b78", "#f5c542"

    def draw(f):
        ax.cla()
        t = min(f, T - 1)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel("PC1", labelpad=-8); ax.set_ylabel("PC2", labelpad=-8); ax.set_zlabel("PC3", labelpad=-8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.04); pane.pane.set_edgecolor((1, 1, 1, 0.08))
        # trial-average reference (gold): the high-SNR clean cycle, drawn in full throughout
        ax.plot(pbar[:, 0], pbar[:, 1], pbar[:, 2], color=gold, lw=3.0, alpha=0.85, zorder=2)
        # filtered: observed part [0:t0] solid teal; held-out continuation [t0:t] faint ghost
        obs = min(t, t0)
        ax.plot(pf[:obs + 1, 0], pf[:obs + 1, 1], pf[:obs + 1, 2], color=teal, lw=2.4, alpha=0.95)
        if t > t0:
            ax.plot(pf[t0:t + 1, 0], pf[t0:t + 1, 1], pf[t0:t + 1, 2], color=ghost, lw=1.4,
                    alpha=0.7, ls="--")
            ax.scatter(*pf[t], color=ghost, s=22, alpha=0.9)
        else:
            ax.scatter(*pf[t], color=teal, s=30)
        ax.scatter(*pf[t0], color="white", s=55, marker="*", zorder=6,
                   edgecolors=crim, linewidths=0.8)
        # free-run forecast from t0
        if t >= t0:
            j = t - t0
            ax.plot(pc[:j + 1, 0], pc[:j + 1, 1], pc[:j + 1, 2], color=crim, lw=2.6, alpha=0.97)
            ax.scatter(*pc[j], color=crim, s=46, zorder=7)
        ax.view_init(elev=22, azim=-70 + 1.05 * f)            # slow continuous rotation
        cyc = t / BINS_PER_CYCLE
        ax.set_title(
            f"sVJF free-run forecast  -  dir 225, L={cfg['latent_dim']}, {res['n_basis']} RBF\n"
            f"gold = trial-average (high-SNR reference)   teal = single-trial filtered   "
            f"crimson = free-run forecast\n"
            f"star = forecast start (1 cycle)   t = {cyc:0.2f} cycles   |   test trial #{trial_i}   |   "
            f"PLL {res['pll']:.2f} (ceil {data['pll_psth']:.2f})",
            fontsize=8.5, color="0.92")
        return []

    print(f"[video] rendering {nframes} frames...")
    anim = animation.FuncAnimation(fig, draw, frames=nframes, interval=50, blit=False)
    anim.save(os.path.join(VID, "forecast_3d.mp4"), writer=animation.FFMpegWriter(
        fps=20, bitrate=4000), dpi=120)
    # money-shot PNG: full trajectories, a flattering angle
    draw(T - 1)
    ax.view_init(elev=24, azim=48)
    fig.savefig(os.path.join(VID, "forecast_3d.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
