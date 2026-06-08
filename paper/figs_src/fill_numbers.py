"""Compute every number the paper cites, straight from the staged result JSONs,
so the prose/tables match the figures exactly. Prints ready-to-paste LaTeX table
rows plus the inline values. Re-run after each data drop; fill the \\TD{} slots.

Run: uv run --with numpy python paper/figs_src/fill_numbers.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SNRS = [-3, 0, 3, 6, 8]


DT = 5e-3
PERIOD = 42.0
CYC_S = PERIOD * DT
KCAP = 200
RHO = 0.5


def auc(kr):
    return float(np.clip(np.asarray(kr), 0, None).mean())


def fc_horizon_s(d):
    """First lead k with R2_forecast(k) < RHO*R2_filter, as a TIME in s (censored at cap)."""
    r2 = np.asarray(d["kpred_r2"]); thr = RHO * d["r2_final"]
    below = np.where(r2 < thr)[0]
    k = int(below[0]) if len(below) else KCAP
    return k * DT, (k >= KCAP)


def main_runs():
    by = defaultdict(list)
    for f in glob.glob(os.path.join(DATA, "main", "mo_*.json")):
        by[os.path.basename(f).split("_")[1]].append(json.load(open(f)))
    return by


def cell(by, mode, snr, field):
    vals = [c[field] for r in by.get(mode, []) for c in r["conditions"]
            if round(c["snr_target"]) == snr and field in c]
    return (np.mean(vals), len(vals)) if vals else (None, 0)


def tab(by, field, title):
    rows = [("projoracle", "projection enc., oracle (ceiling)"),
            ("online", "projection enc., online readout ($\\tau{=}8$)"),
            ("spikeoracle", "spike enc., oracle (vanilla-VJF bound)"),
            ("frozenpca", "spike enc., frozen PCA")]
    print(f"\n% ---- {title} ({field}) ----")
    for mk, lab in rows:
        cells = []
        for s in SNRS:
            m, n = cell(by, mk, s, field)
            cells.append(f"{m:.2f}" if m is not None else "\\TD{}")
        print(f"{lab:44s} & " + " & ".join(cells) + "\\\\")


def readout_arms(data_dir, n):
    by = defaultdict(list)
    for f in glob.glob(os.path.join(data_dir, "*.json")):
        if f"_n{n}_" in os.path.basename(f):
            d = json.load(open(f)); by[d["arm"]].append(d)
    return by


def agg(by, arm, fn):
    if not by.get(arm):
        return None
    return float(np.mean([fn(r) for r in by[arm]]))


def report_readout(tag, by):
    print(f"\n% ---- readout schedule, {tag} ----")
    if not by:
        print("  (no data yet)"); return
    for arm in ("proj_oracle", "frozen_pca", "freeze_after", "online_base"):
        one = agg(by, arm, lambda r: r["onestep_r2"])
        if one is None:
            continue
        hs = [fc_horizon_s(r) for r in by[arm]]
        hmean = float(np.mean([h for h, _ in hs])); cens = all(c for _, c in hs)
        hstr = f">{KCAP*DT:.1f}s" if cens else f"{hmean:.2f}s ({hmean/CYC_S:.1f} cyc)"
        print(f"  {arm:14s} one-step R2={one:.2f}  forecast horizon={hstr}  (n={len(by[arm])})")
    # subspace drift: per-refresh rotation and net change in angle-to-oracle (online arm)
    for arm in ("online_base", "freeze_after"):
        runs = by.get(arm, [])
        if runs and runs[0].get("drift"):
            per = [d["angle_prev_deg"] for r in runs for d in r["drift"]]
            nets = []
            for r in runs:
                ao = [d["angle_oracle_deg"] for d in r["drift"]]
                nets.append(max(ao) - min(ao))
            print(f"  {arm:14s} per-refresh rot median={np.median(per):.3f} deg, "
                  f"cumulative |Delta angle-to-oracle|={np.mean(nets):.2f} deg")


def report_timing():
    f = os.path.join(DATA, "extra", "x_timing.json")
    print("\n% ---- timing (C5) ----")
    if not os.path.exists(f):
        print("  (no timing data)"); return
    d = sorted(json.load(open(f))["conditions"], key=lambda c: c["snr_target"])
    meds = []
    for c in d:
        s = np.asarray(c.get("timing_sample_ms") or [])
        if len(s):
            meds.append(np.percentile(s, 50))
            print(f"  n={c['n_neurons']:>3} ({round(c['snr_target'])} dB): "
                  f"p50={np.percentile(s,50):.2f}  p95={np.percentile(s,95):.2f}  max={s.max():.2f} ms")
    if meds:
        print(f"  median across conditions: {np.median(meds):.2f} ms; "
              f"#conditions with p95<5ms: "
              f"{sum(np.percentile(np.asarray(c.get('timing_sample_ms') or [0]),95)<5 for c in d)}/{len(d)}")


if __name__ == "__main__":
    by = main_runs()
    print(f"main modes present: {sorted(by)} "
          f"(seeds: {{ {', '.join(f'{k}:{len(v)}' for k,v in by.items())} }})")
    tab(by, "r2_final", "Table 1: filtered-latent R2")
    tab(by, "onestep_r2", "Table 2: one-step prediction R2")
    print("\n% ---- inline (C1/C2) ----")
    for s in (-3, 0):
        po, _ = cell(by, "projoracle", s, "onestep_r2")
        so, _ = cell(by, "spikeoracle", s, "onestep_r2")
        if po is not None:
            print(f"  one-step R2 @ {s} dB: proj+oracle={po:.2f} vs spike+oracle={so:.2f}")
    for s in (0, 8):
        on, _ = cell(by, "online", s, "r2_final")
        so, _ = cell(by, "spikeoracle", s, "r2_final")
        fp, _ = cell(by, "frozenpca", s, "r2_final")
        po, _ = cell(by, "projoracle", s, "r2_final")
        if on is not None:
            print(f"  filtered R2 @ {s} dB: online={on:.2f} spike+oracle={so:.2f} "
                  f"frozenPCA={fp:.2f} ceiling={po:.2f}")
    report_readout("high SNR (n=250, 8 dB)", readout_arms(os.path.join(DATA, "e1"), 250))
    report_readout("low SNR (n=30, 0 dB)", readout_arms(os.path.join(DATA, "e1_low"), 30))
    report_timing()
