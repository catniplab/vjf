"""Gallery of Fig-6-like free-run slices for the top forecast-accuracy configs of a search.

Reads a search's shard results, takes the top-N configs by weighted forecast accuracy
(gated by reconstruction), RE-FITS each (the searches save only metrics), and renders each
model's autonomous free-run of the leading decoded-variance factor -- the trial-average
reference (gold) vs the free-run forecast (crimson), launched after the onset transient -- in
one grid, so the spectrum of fitted models is comparable at a glance.

Run: uv run python -m experiments.v1_graf.gallery [results_dir]
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval
from experiments.v1_graf.forecast_video import filtered_path, T0
from experiments.v1_graf.factor_order import decoded_variance_basis, project
from experiments.v1_graf.eval import forecast_reconstruction_deviance, forecast_skill_summary
from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
DEFAULT_RESULTS = os.path.join(HERE, "..", "..", "gcp_runs",
                               "graf-search-single-roll-20260615-195016", "results")
N_TOP = 8


def top_configs(results_dir, n):
    recs = []
    for f in glob.glob(os.path.join(results_dir, "search_single_shard*.json")):
        b = json.load(open(f))
        recs += (b["records"] if isinstance(b, dict) else b)
    ok = [r for r in recs if "error" not in r
          and isinstance(r.get("weighted_persist_skill"), (int, float))
          and np.isfinite(r["weighted_persist_skill"])]
    ok.sort(key=lambda r: r["weighted_persist_skill"], reverse=True)
    return ok[:n]


def fit_and_freerun(cfg, data):
    """Re-fit one config and return (pbar, pf, pc, fve, S, lab) for its leading-factor free-run."""
    test, psth = data["test_trials"], data["psth_counts"]
    res = _train_eval(
        data["train_trials"], test, data["test_counts"], float(data["test_counts"].mean()),
        epochs=cfg["epochs"], latent_dim=cfg["latent_dim"], N=data["N"], grow=True,
        grow_weight_init="residual", flow="sgd", optimizer="adam", lr=cfg["lr"],
        rbf_base=cfg.get("rbf_base", 100), max_rbf=cfg.get("max_rbf"),
        dyn_noise=cfg["dyn_noise"], smooth_lambda=cfg.get("smooth_lambda", 0.0),
        rollout=cfg.get("rollout", False), rollout_k=cfg.get("rollout_k", 12),
        rollout_epochs=cfg.get("rollout_epochs", 25), psth_counts=psth, return_model=True)
    model, ro = res["model"], res["ro"]
    starts = [T0]
    per = [(forecast_skill_summary([forecast_reconstruction_deviance(model, ro, tc, psth, starts, (8, 16, 32))])
            ["weighted_persist_skill"], i) for i, tc in enumerate(test)]
    bi = max(per)[1]
    tc = test[bi]
    Tn = tc.shape[0]
    xfilt, _ = filtered_path(model, ro, tc)
    with torch.no_grad():
        x, _ = model.forecast(torch.as_tensor(xfilt[T0][None].astype(np.float32)), n_step=Tn - 1 - T0)
    xfc = x.detach().cpu().numpy()[:, 0, :]
    allt = list(data["train_trials"]) + list(data["val_trials"]) + list(test)
    xbar = np.stack([filtered_path(model, ro, t)[0] for t in allt], 0).mean(0)
    C = model.decoder.decode.weight.detach().cpu().numpy()
    W, fve, ctr = decoded_variance_basis(xbar, C)
    S = forecast_skill_summary([forecast_reconstruction_deviance(model, ro, tc, psth, list(range(T0, Tn - 32, 8)) or [T0], (8, 16, 32)) for tc in test])["weighted_persist_skill"]
    roll = f"+roll(k{cfg['rollout_k']})" if cfg.get("rollout") else "1-step"
    lab = f"L{cfg['latent_dim']} λ{cfg.get('smooth_lambda', 0):g} {roll}\nS={S:+.3f} (test PLL {res['pll']:.2f})"
    return project(xbar, W, ctr), project(xfilt, W, ctr), project(xfc, W, ctr), fve, lab


def main():
    torch.set_default_dtype(torch.float32)
    results = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(DEFAULT_RESULTS)
    cfgs = top_configs(results, N_TOP)
    print(f"top {len(cfgs)} configs from {results}", flush=True)
    data = prepare_single_dir_data(direction=225.0, n_val=10)
    set_style()
    ncol = 4
    nrow = int(np.ceil(len(cfgs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FW(1.0), 1.5 * nrow + 0.3), sharex=True, squeeze=False)
    gold, crim = "#c8920a", "#d6336c"
    for ci, rec in enumerate(cfgs):
        pbar, pf, pc, fve, lab = fit_and_freerun(rec["config"], data)
        ax = axes[ci // ncol][ci % ncol]
        t = np.arange(pbar.shape[0]); tfc = np.arange(T0, T0 + pc.shape[0])
        ax.plot(t, pbar[:, 0], color=gold, lw=2.0, alpha=0.9)          # trial-average factor 1
        ax.plot(tfc, pc[:, 0], color=crim, lw=1.8)                     # free-run forecast factor 1
        ax.axvline(T0, color="0.6", ls="--", lw=0.7)
        ax.set_title(lab, fontsize=6.5)
        ax.set_yticks([])
        print(f"  {ci + 1}/{len(cfgs)}: {lab.splitlines()[0]}  {lab.splitlines()[1]}", flush=True)
    for j in range(len(cfgs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Free-run forecast (gold = trial-average, crimson = autonomous free-run) -- "
                 "leading decoded-variance factor, top configs", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"gallery.{ext}"), dpi=200)
    plt.close(fig)
    print(f"gallery -> {FIGS}/gallery.png", flush=True)


if __name__ == "__main__":
    main()
