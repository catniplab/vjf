"""Systematic rollout study: scan targets x horizon x clip across seeds -> the report's tables.

For each fit seed we train the one-step model once, then fine-tune copies with each rollout
STRATEGY (target * horizon ladder * grad clip) and measure two orthogonal things:

  forecasting  : held-out free-run accuracy -- the locked score S (weighted k=8/16/32 vs
                 persistence) and accuracy vs the near-oracle PSTH at a short (k=8) and long
                 (k=96) horizon;
  dynamics     : the autonomous attractor -- ring amplitude (relative to the data orbit) from a
                 dense-initial-condition return map, the flow-map spectral radius at the ring
                 center (>1 => unstable center => limit cycle), and a focus/limit-cycle verdict.

Aggregates mean +/- std over seeds, prints a markdown table, writes a LaTeX fragment for the
report, and renders the multi-seed forecast-accuracy-vs-horizon figure.
Run: uv run python -m experiments.v1_graf.rollout_study
"""
from __future__ import annotations
import copy
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval
from experiments.v1_graf.eval import forecast_reconstruction_deviance, forecast_skill_summary
from experiments.v1_graf.forecast_video import filtered_path, T0, BINS_PER_CYCLE
from experiments.v1_graf.factor_order import decoded_variance_basis, project
from experiments.v1_graf.rollout import rollout_finetune
from experiments.v1_graf.rollout_diag import free_run, cycle_amplitude
from experiments.v1_graf.limit_cycle_class import map_jacobian_eig
from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
# Direction + seeds are env-parameterized for the GCP fleet (one VM per direction); outputs are
# direction-tagged so a fleet does not clobber. Defaults reproduce the in-report dir-225 study.
DIRECTION = float(os.environ.get("ROLLOUT_DIR", "225"))
SEEDS = [int(s) for s in os.environ.get("ROLLOUT_SEEDS", "20260609,20260610,20260611").split(",")]
_TAG = f"_d{int(DIRECTION)}" if os.environ.get("ROLLOUT_DIR") else ""
OUT = os.path.join(HERE, f"rollout_study_results{_TAG}.json")
TEX = os.path.join(HERE, "report_m1", f"rollout_study_table{_TAG}.tex")
FIG_NAME = f"forecast_vs_horizon{_TAG}.png"
HORIZONS = (8, 16, 24, 32, 48, 64, 96)          # bins (0.5 .. 6 cycles)
RADII = np.linspace(0.2, 2.3, 9)                # initial radii for the return map
N_CYC = 300
TAIL = 30

# strategy = (label, target_mode, k_ladder, clip); None target = one-step (no rollout)
STRATEGIES = [
    ("one-step (no rollout)", None, None, None),
    ("single-trial, k<=48", "filtered", [8, 16, 32, 48], 1.0),
    ("trial-avg, k<=48", "trialavg", [8, 16, 32, 48], 1.0),
    ("periodic, k<=48", "cycle", [16, 32, 48], 1.0),
    ("periodic, k<=96", "cycle", [24, 48, 96], 10.0),
    ("periodic, k<=160", "cycle", [32, 64, 128, 160], 10.0),
]
EP = {"filtered": 40, "trialavg": 300, "cycle": 250}


def evaluate(model, ro, test, psth, W, ctr, off, data_amp):
    n_bin = test[0].shape[0]
    starts = list(range(T0, n_bin, 2))
    devs = [forecast_reconstruction_deviance(model, ro, tc, psth, starts, HORIZONS) for tc in test]
    s = forecast_skill_summary(devs, horizons=HORIZONS, weights=tuple([1 / len(HORIZONS)] * len(HORIZONS)))
    vp = {k: s["skill"][k]["vs_persist"] for k in HORIZONS}
    vq = {k: s["skill"][k]["vs_psth"] for k in HORIZONS}
    S = 0.5 * vp[8] + 0.3 * vp[16] + 0.2 * vp[32]                 # locked selection score
    # return map: dense ICs -> tail amplitude; ring center for the eigenvalue test
    tails, settle = [], None
    for r in RADII:
        xr = free_run(model, ctr + r * off, N_CYC * BINS_PER_CYCLE)
        tails.append(float(cycle_amplitude(project(xr, W, ctr))[-TAIL:].mean()))
        if abs(r - 1.0) < 0.25 or settle is None:
            settle = xr[-TAIL * BINS_PER_CYCLE:].mean(0)
    tails = np.array(tails)
    ring = float(np.median(tails))
    rho = float(np.max(np.abs(map_jacobian_eig(model, settle))))
    collapsed = tails.std() / (ring + 1e-9) < 0.2
    verdict = ("focus" if (rho <= 1.0 or ring < 0.2 * data_amp)
               else ("limit cycle" if collapsed else "unclear"))
    return dict(S=S, vp=vp, vq=vq, ring_data=ring / data_amp, rho=rho, verdict=verdict)


def main():
    torch.set_default_dtype(torch.float32)
    print(f"=== direction {DIRECTION:.0f}, seeds {SEEDS} ===", flush=True)
    data = prepare_single_dir_data(direction=DIRECTION, n_val=10)
    test, psth = data["test_trials"], data["psth_counts"]
    ybar = float(data["test_counts"].mean())
    runs = {lab: [] for lab, *_ in STRATEGIES}
    plls = []

    for sd in SEEDS:
        print(f"\n=== fit seed {sd} ===", flush=True)
        res = _train_eval(data["train_trials"], test, data["test_counts"], ybar, epochs=100,
                          latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
                          flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
                          dyn_noise=0.0, smooth_lambda=0.0, psth_counts=psth, return_model=True, seed=sd)
        model0, ro = res["model"], res["ro"]
        plls.append(res["pll"])
        allt = list(data["train_trials"]) + list(data["val_trials"]) + list(test)
        xbar = np.stack([filtered_path(model0, ro, t)[0] for t in allt], 0).mean(0)
        C = model0.decoder.decode.weight.detach().cpu().numpy()
        W, fve, ctr = decoded_variance_basis(xbar, C)
        data_amp = float(cycle_amplitude(project(xbar, W, ctr)).mean())
        off = filtered_path(model0, ro, test[0])[0][T0] - ctr
        for lab, tm, lad, clip in STRATEGIES:
            m = copy.deepcopy(model0)
            if tm is not None:
                rollout_finetune(m, ro, data["train_trials"], epochs=EP[tm], k_ladder=lad,
                                 target_mode=tm, clip=clip, lr=1e-3, seed=sd + 1)
            r = evaluate(m, ro, test, psth, W, ctr, off, data_amp)
            runs[lab].append(r)
            print(f"  {lab:24s} S {r['S']:+.4f} | vsPSTH k8 {r['vq'][8]:+.3f} k96 {r['vq'][96]:+.3f} "
                  f"| ring {r['ring_data']:.2f}x | rho {r['rho']:.3f} | {r['verdict']}", flush=True)

    # aggregate mean +/- std
    def ms(vals):
        a = np.array(vals); return float(a.mean()), float(a.std())
    agg = {}
    for lab, *_ in STRATEGIES:
        rs = runs[lab]
        agg[lab] = dict(
            S=ms([r["S"] for r in rs]), vq8=ms([r["vq"][8] for r in rs]), vq96=ms([r["vq"][96] for r in rs]),
            ring=ms([r["ring_data"] for r in rs]), rho=ms([r["rho"] for r in rs]),
            verdict=max(set(r["verdict"] for r in rs), key=[r["verdict"] for r in rs].count))
    pll_m, pll_s = ms(plls)
    json.dump({"seeds": SEEDS, "pll": [pll_m, pll_s],
               "agg": {k: {kk: vv for kk, vv in v.items()} for k, v in agg.items()},
               "raw": {k: [{kk: (vv if kk != "vp" and kk != "vq" else vv) for kk, vv in r.items()}
                           for r in runs[k]] for k in runs}},
              open(OUT, "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    # markdown table
    print(f"\nfit recon PLL {pll_m:.3f} +/- {pll_s:.3f} (ceiling {data['pll_psth']:.3f})\n", flush=True)
    hdr = f"| {'strategy':24s} | target | maxk | clip |   S (locked)  | a_k vsPSTH k=8 | k=96 | ring/data | rho  | verdict |"
    print(hdr); print("|" + "-" * (len(hdr) - 2) + "|")
    cfg = {lab: (tm, lad, clip) for lab, tm, lad, clip in STRATEGIES}
    for lab, *_ in STRATEGIES:
        tm, lad, clip = cfg[lab]
        a = agg[lab]
        mk = "-" if not lad else str(max(lad))
        print(f"| {lab:24s} | {tm or '-':10s} | {mk:>4s} | {('%g'%clip) if clip else '-':>4s} | "
              f"{a['S'][0]:+.4f}+-{a['S'][1]:.3f} | {a['vq8'][0]:+.3f} | {a['vq96'][0]:+.3f} | "
              f"{a['ring'][0]:.2f}+-{a['ring'][1]:.2f} | {a['rho'][0]:.2f} | {a['verdict']} |", flush=True)

    # LaTeX fragment (booktabs)
    rows = []
    for lab, tm, lad, clip in STRATEGIES:
        a = agg[lab]
        mk = "--" if not lad else str(max(lad))
        cl = "--" if not clip else "%g" % clip
        rows.append(f"{lab} & {tm or '--'} & {mk} & {cl} & "
                    f"${a['S'][0]:+.3f}{{\\scriptstyle\\pm{a['S'][1]:.3f}}}$ & "
                    f"${a['vq8'][0]:+.3f}$ & ${a['vq96'][0]:+.3f}$ & "
                    f"${a['ring'][0]:.2f}$ & ${a['rho'][0]:.2f}$ & {a['verdict']} \\\\")
    with open(TEX, "w") as fh:
        fh.write("% auto-generated by rollout_study.py\n\\begin{tabular}{lcccccccl}\n\\toprule\n"
                 "strategy & target & $k_{\\max}$ & clip & $S$ & $a_8$ vs PSTH & $a_{96}$ & ring/data & $\\rho$ & dynamics\\\\\n"
                 "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # multi-seed forecast-accuracy-vs-horizon figure
    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(FW(1.0), FW(1.0) * 0.42))
    kc = np.array(HORIZONS) / BINS_PER_CYCLE
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(STRATEGIES)))
    for (lab, *_), col in zip(STRATEGIES, cols):
        rs = runs[lab]
        vp = np.array([[r["vp"][k] for k in HORIZONS] for r in rs])
        vq = np.array([[r["vq"][k] for k in HORIZONS] for r in rs])
        ax[0].plot(kc, vp.mean(0), "-o", color=col, lw=1.4, ms=2.5, label=lab)
        ax[0].fill_between(kc, vp.mean(0) - vp.std(0), vp.mean(0) + vp.std(0), color=col, alpha=0.12)
        ax[1].plot(kc, vq.mean(0), "-o", color=col, lw=1.4, ms=2.5, label=lab)
        ax[1].fill_between(kc, vq.mean(0) - vq.std(0), vq.mean(0) + vq.std(0), color=col, alpha=0.12)
    for a, base in zip(ax, ("persistence", "near-oracle PSTH")):
        a.axhline(0, color="0.6", lw=0.8); a.set_xlabel("forecast horizon (cycles)")
        a.set_ylabel(f"forecast accuracy vs {base}")
    ax[0].legend(fontsize=5, loc="best"); ax[0].set_title("vs persistence", fontsize=8)
    ax[1].set_title("vs PSTH (near-oracle)", fontsize=8)
    fig.savefig(os.path.join(FIGS, FIG_NAME), dpi=200)
    fig.savefig(os.path.join(FIGS, FIG_NAME.replace(".png", ".pdf")))
    plt.close(fig)
    print(f"\n-> {OUT}\n-> {TEX}\n-> {FIGS}/{FIG_NAME}", flush=True)


if __name__ == "__main__":
    main()
