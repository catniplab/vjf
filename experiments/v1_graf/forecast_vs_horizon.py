"""Forecast accuracy vs. horizon (single-seed, exploratory).

SUPERSEDED for the report by rollout_study.py, which runs this multi-seed (mean +/- s.d.) and is
the canonical generator of report_m1/figs/forecast_vs_horizon.png. This script writes a separate
*_singleseed.png and is kept only as the quick single-fit probe that first tested the (refuted)
hypothesis that a sustained-cycle flow would overtake a contractive one at long horizons -- it
does not; see rollout_study.py and the report.

Scores single-fit held-out forecast accuracy (deviance reduction vs persistence AND vs the
near-oracle PSTH) at k = 8..96 bins (0.5..6 cycles) for four flows -- one-step, single-trial
rollout, trial-average rollout, and periodic long-horizon rollout. The contractive single-trial
rollout leads at every horizon; the sustaining flows do not overtake it.
Run: uv run python -m experiments.v1_graf.forecast_vs_horizon
"""
from __future__ import annotations
import copy
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval
from experiments.v1_graf.eval import forecast_reconstruction_deviance, forecast_skill_summary
from experiments.v1_graf.forecast_video import T0, BINS_PER_CYCLE
from experiments.v1_graf.rollout import rollout_finetune
from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
HORIZONS = (8, 16, 24, 32, 48, 64, 96)          # bins (0.5 .. 6 cycles)
COLORS = {"before": "#7a7a7a", "filtered": "#2ca02c", "trialavg": "#1b6fd6", "cycle k160": "#d6336c"}


def accuracy_by_k(model, ro, trials, psth):
    n_bin = trials[0].shape[0]
    starts = list(range(T0, n_bin, 2))                                   # fn skips t0+max(H)>=n_bin
    devs = [forecast_reconstruction_deviance(model, ro, tc, psth, starts, HORIZONS) for tc in trials]
    s = forecast_skill_summary(devs, horizons=HORIZONS, weights=tuple([1 / len(HORIZONS)] * len(HORIZONS)))
    vp = [s["skill"][k]["vs_persist"] for k in HORIZONS]
    vq = [s["skill"][k]["vs_psth"] for k in HORIZONS]
    return vp, vq


def main():
    torch.set_default_dtype(torch.float32)
    data = prepare_single_dir_data(direction=225.0, n_val=10)
    test, psth = data["test_trials"], data["psth_counts"]
    ybar = float(data["test_counts"].mean())
    res = _train_eval(data["train_trials"], test, data["test_counts"], ybar, epochs=100,
                      latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
                      flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
                      dyn_noise=0.0, smooth_lambda=0.0, psth_counts=psth, return_model=True)
    model0, ro = res["model"], res["ro"]
    print(f"fitted one-step: test PLL {res['pll']:.3f} (ceil {data['pll_psth']:.3f})", flush=True)

    models = {"before": copy.deepcopy(model0)}
    print("[filtered] rollout ...", flush=True)
    models["filtered"] = rollout_finetune(copy.deepcopy(model0), ro, data["train_trials"],
                                          epochs=40, k_ladder=[8, 16, 32, 48], target_mode="filtered")
    print("[trialavg] rollout ...", flush=True)
    models["trialavg"] = rollout_finetune(copy.deepcopy(model0), ro, data["train_trials"],
                                          epochs=300, k_ladder=[8, 16, 32, 48], target_mode="trialavg")
    print("[cycle k160] rollout ...", flush=True)
    models["cycle k160"] = rollout_finetune(copy.deepcopy(model0), ro, data["train_trials"],
                                            epochs=250, k_ladder=[32, 64, 128, 160], target_mode="cycle", clip=10.0)

    accs = {}
    for tag, m in models.items():
        vp, vq = accuracy_by_k(m, ro, test, psth)
        accs[tag] = (vp, vq)
        print(f"  {tag:11s} vs-persist " + "/".join(f"{x:+.3f}" for x in vp), flush=True)
        print(f"  {tag:11s} vs-PSTH    " + "/".join(f"{x:+.3f}" for x in vq), flush=True)

    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(FW(1.0), FW(1.0) * 0.42))
    kc = np.array(HORIZONS) / BINS_PER_CYCLE
    for tag, (vp, vq) in accs.items():
        ax[0].plot(kc, vp, "-o", color=COLORS[tag], lw=1.6, ms=3, label=tag)
        ax[1].plot(kc, vq, "-o", color=COLORS[tag], lw=1.6, ms=3, label=tag)
    for a, base in zip(ax, ("persistence", "near-oracle PSTH")):
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlabel("forecast horizon (cycles)")
        a.set_ylabel(f"forecast accuracy vs {base}")
    ax[0].legend(fontsize=6, loc="best")
    ax[0].set_title("vs persistence", fontsize=8); ax[1].set_title("vs PSTH (near-oracle)", fontsize=8)
    fig.savefig(os.path.join(FIGS, "forecast_vs_horizon_singleseed.png"), dpi=200)
    plt.close(fig)
    print(f"\n-> {FIGS}/forecast_vs_horizon.png", flush=True)


if __name__ == "__main__":
    main()
