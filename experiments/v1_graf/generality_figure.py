"""Generality across grating directions: forecast accuracy vs horizon for the single-trial rollout.

Pools the rollout_study raw results for four directions -- 225 (3 local + 5 GCP = 8 seeds), and
45/105/295 (3 GCP seeds each) -- and, for the selected single-trial rollout, plots held-out
forecast accuracy vs horizon against persistence and the near-oracle PSTH, one line per direction.
Shows that beating the PSTH is direction-dependent: the free-run beats it on the noisier
directions (105/295/45) but not on the strongest, cleanest one (225). Also prints a per-direction
summary table. Run: uv run python -m experiments.v1_graf.generality_figure
"""
from __future__ import annotations
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
G = os.path.join(HERE, "..", "..", "gcp_runs")
HORIZONS = (8, 16, 24, 32, 48, 64, 96)
BINS_PER_CYCLE = 16
STRAT = "single-trial, k<=48"
# direction -> list of result JSONs to pool
SOURCES = {
    225: [os.path.join(HERE, "rollout_study_results.json"),
          os.path.join(G, "rollout-study-20260625", "results", "rollout_study_results_d225.json")],
    105: [os.path.join(G, "rollout-study-20260626-generality", "results", "rollout_study_results_d105.json")],
    295: [os.path.join(G, "rollout-study-20260626-generality", "results", "rollout_study_results_d295.json")],
    45:  [os.path.join(G, "rollout-study-20260626-generality", "results", "rollout_study_results_d45.json")],
}
PLL = {225: 0.73, 105: 0.66, 45: 0.62, 295: 0.58}             # PSTH-ceiling PLL (data SNR proxy)


def _v(rec, field, k):
    d = rec[field]; return d.get(str(k), d.get(k))


def load(dirn):
    recs = []
    for f in SOURCES[dirn]:
        recs += json.load(open(f))["raw"].get(STRAT, [])
    return recs


def main():
    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(FW(1.0), FW(1.0) * 0.42))
    kc = np.array(HORIZONS) / BINS_PER_CYCLE
    order = [225, 105, 295, 45]
    cols = {225: "#d6336c", 105: "#1b9e9a", 295: "#1b6fd6", 45: "#c8920a"}
    print(f"{'dir':>4s} {'seeds':>5s} {'S':>14s} {'a8 vsPSTH':>11s} {'a96 vsPSTH':>11s}")
    for dirn in order:
        rs = load(dirn)
        vp = np.array([[_v(r, "vp", k) for k in HORIZONS] for r in rs], float)
        vq = np.array([[_v(r, "vq", k) for k in HORIZONS] for r in rs], float)
        lab = f"dir {dirn} (PLL {PLL[dirn]:.2f})"
        for a, m in ((ax[0], vp), (ax[1], vq)):
            a.plot(kc, m.mean(0), "-o", color=cols[dirn], lw=1.5, ms=2.8, label=lab)
            a.fill_between(kc, m.mean(0) - m.std(0), m.mean(0) + m.std(0), color=cols[dirn], alpha=0.10)
        S = np.mean([r["S"] for r in rs])
        print(f"{dirn:>4d} {len(rs):>5d} {S:+.4f}±{np.std([r['S'] for r in rs]):.3f} "
              f"{vq[:, 0].mean():+.3f}      {vq[:, -1].mean():+.3f}")
    for a, base in zip(ax, ("persistence", "near-oracle PSTH")):
        a.axhline(0, color="0.6", lw=0.8); a.set_xlabel("forecast horizon (cycles)")
        a.set_ylabel(f"forecast accuracy vs {base}")
    ax[0].legend(fontsize=6, loc="best"); ax[0].set_title("vs persistence", fontsize=8)
    ax[1].set_title("vs PSTH (near-oracle)", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"generality.{ext}"), dpi=200)
    plt.close(fig)
    print(f"\n-> {FIGS}/generality.png")


if __name__ == "__main__":
    main()
