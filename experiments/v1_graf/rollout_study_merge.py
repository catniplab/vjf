"""Merge rollout_study raw results across seeds/directions -> report table + figure (no re-fit).

Each rollout_study run stores per-seed raw records (S, vp/vq per horizon, ring_data, rho) in its
results JSON. This merges any number of those JSONs (e.g. the 3 local + 5 GCP seeds for dir 225,
plus the other-direction runs), recomputes the mean+-s.d. strategy table, and re-renders the
forecast-accuracy-vs-horizon figure from the stored curves -- so the report's Table + figure
reflect all seeds without retraining.

Usage: uv run python -m experiments.v1_graf.rollout_study_merge <results.json> [more.json ...]
  (direction 225 JSONs are pooled for the in-report table/figure; other directions are summarized
   for the generality note.)
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
TEX = os.path.join(HERE, "report_m1", "rollout_study_table.tex")
HORIZONS = (8, 16, 24, 32, 48, 64, 96)
BINS_PER_CYCLE = 16
STRATEGIES = ["one-step (no rollout)", "single-trial, k<=48", "trial-avg, k<=48",
              "periodic, k<=48", "periodic, k<=96", "periodic, k<=160"]
TGT = {"one-step (no rollout)": ("--", "--", "--"), "single-trial, k<=48": ("filtered", 48, 1),
       "trial-avg, k<=48": ("trialavg", 48, 1), "periodic, k<=48": ("cycle", 48, 1),
       "periodic, k<=96": ("cycle", 96, 10), "periodic, k<=160": ("cycle", 160, 10)}


def _vq(rec, k):                                            # JSON keys may be str or int
    d = rec["vq"]; return d.get(str(k), d.get(k))


def _vp(rec, k):
    d = rec["vp"]; return d.get(str(k), d.get(k))


def main():
    paths = sys.argv[1:]
    if not paths:
        print("need >=1 results JSON"); return
    by_dir = {}                                             # direction -> {strategy: [recs]}
    for p in paths:
        d = json.load(open(p))
        dirn = int(d.get("direction", 225))
        raw = d["raw"]
        dd = by_dir.setdefault(dirn, {s: [] for s in STRATEGIES})
        for s in STRATEGIES:
            dd[s] += raw.get(s, [])
    for dirn in sorted(by_dir):
        n = len(by_dir[dirn][STRATEGIES[0]])
        print(f"direction {dirn}: {n} seeds pooled")

    main_dir = 225 if 225 in by_dir else sorted(by_dir)[0]
    runs = by_dir[main_dir]
    nseed = len(runs[STRATEGIES[0]])

    def ms(a): a = np.array(a, float); return a.mean(), a.std()
    print(f"\n=== dir {main_dir}, {nseed} seeds: report table ===")
    rows = []
    for s in STRATEGIES:
        rs = runs[s]
        S = ms([r["S"] for r in rs]); ring = ms([r["ring_data"] for r in rs]); rho = ms([r["rho"] for r in rs])
        a8 = np.mean([_vq(r, 8) for r in rs]); a96 = np.mean([_vq(r, 96) for r in rs])
        tm, mk, cl = TGT[s]
        Sbold = f"\\mathbf{{{S[0]:+.3f}{{\\scriptstyle\\pm{S[1]:.3f}}}}}" if S[0] > 0 and S[1] < abs(S[0]) else f"{S[0]:+.3f}{{\\scriptstyle\\pm{S[1]:.3f}}}"
        rows.append(f"    {tm:13s} & {str(mk):>3s} & {str(cl):>3s} & ${Sbold}$ & ${a8:+.3f}$ & ${a96:+.3f}$ "
                    f"& ${ring[0]:.2f}{{\\scriptstyle\\pm{ring[1]:.2f}}}$ & ${rho[0]:.2f}$ \\\\")
        print(f"  {s:22s} S {S[0]:+.4f}±{S[1]:.3f} | a8 {a8:+.3f} a96 {a96:+.3f} | ring {ring[0]:.2f}±{ring[1]:.2f} | rho {rho[0]:.2f}±{rho[1]:.2f}")
    rhos = sorted(round(r["rho"], 2) for r in runs["one-step (no rollout)"])
    print(f"  one-step rho range over {nseed} seeds: {rhos}")

    # figure: forecast accuracy vs horizon, mean+-sd over the pooled seeds
    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(FW(1.0), FW(1.0) * 0.42))
    kc = np.array(HORIZONS) / BINS_PER_CYCLE
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(STRATEGIES)))
    short = {"one-step (no rollout)": "one-step", "single-trial, k<=48": "single-trial",
             "trial-avg, k<=48": "trial-avg", "periodic, k<=48": "periodic k48",
             "periodic, k<=96": "periodic k96", "periodic, k<=160": "periodic k160"}
    for s, col in zip(STRATEGIES, cols):
        rs = runs[s]
        vp = np.array([[_vp(r, k) for k in HORIZONS] for r in rs], float)
        vq = np.array([[_vq(r, k) for k in HORIZONS] for r in rs], float)
        ax[0].plot(kc, vp.mean(0), "-o", color=col, lw=1.4, ms=2.5, label=short[s])
        ax[0].fill_between(kc, vp.mean(0) - vp.std(0), vp.mean(0) + vp.std(0), color=col, alpha=0.12)
        ax[1].plot(kc, vq.mean(0), "-o", color=col, lw=1.4, ms=2.5, label=short[s])
        ax[1].fill_between(kc, vq.mean(0) - vq.std(0), vq.mean(0) + vq.std(0), color=col, alpha=0.12)
    for a, base in zip(ax, ("persistence", "near-oracle PSTH")):
        a.axhline(0, color="0.6", lw=0.8); a.set_xlabel("forecast horizon (cycles)")
        a.set_ylabel(f"forecast accuracy vs {base}")
    ax[0].legend(fontsize=5, loc="best"); ax[0].set_title("vs persistence", fontsize=8)
    ax[1].set_title("vs PSTH (near-oracle)", fontsize=8)
    fig.savefig(os.path.join(FIGS, "forecast_vs_horizon.png"), dpi=200)
    fig.savefig(os.path.join(FIGS, "forecast_vs_horizon.pdf"))
    plt.close(fig)
    print(f"\n-> {FIGS}/forecast_vs_horizon.png ({nseed} seeds)")
    print("\n-- table rows (paste into tab:rollout) --\n" + "\n".join(rows))


if __name__ == "__main__":
    main()
