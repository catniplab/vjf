"""Slim the staged result JSONs in paper/data/ down to ONLY the fields the figures and
fill_numbers.py read, so the figure inputs stay reproducible without committing large
experimental outcome data (raw runs live in gcp_runs/, which is gitignored). Run in place.

Run: uv run --with numpy python paper/figs_src/slim_data.py
"""
import glob
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def keep(d, fields):
    return {k: d[k] for k in fields if k in d}


def _round(o, nd=5):                       # figures need ~5 sig digits, not float64 text
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, list):
        return [_round(x, nd) for x in o]
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    return o


def write(path, obj):
    with open(path, "w") as f:
        json.dump(_round(obj), f)


def slim():
    # main 5-SNR comparison: per-condition metrics + the convergence log (Fig A/B, tables)
    for f in glob.glob(os.path.join(DATA, "main", "mo_*.json")):
        d = json.load(open(f)); conds = []
        for c in d.get("conditions", []):
            cc = keep(c, ["snr_target", "n_neurons", "r2_final", "onestep_r2", "rate_corr"])
            if isinstance(c.get("log"), dict):
                cc["log"] = keep(c["log"], ["step", "r2"])
            conds.append(cc)
        write(f, {"conditions": conds})
    # timing: keep a downsampled sample (<=800 pts) for the box plot + the full-series percentiles
    for f in glob.glob(os.path.join(DATA, "extra", "x_timing.json")):
        d = json.load(open(f)); conds = []
        for c in d.get("conditions", []):
            s = np.asarray(c.get("timing_sample_ms") or [])
            idx = np.linspace(0, len(s) - 1, min(800, len(s))).astype(int) if len(s) else []
            cc = keep(c, ["snr_target", "n_neurons", "per_bin_p50_ms", "per_bin_p95_ms", "per_bin_max_ms"])
            cc["timing_sample_ms"] = s[idx].tolist() if len(idx) else []
            conds.append(cc)
        write(f, {"conditions": conds})
    # readout-schedule (E1 family): metrics + k-step curve + (small) drift series
    for sub in ("e1", "e1_low", "e1_mid", "e2"):
        for f in glob.glob(os.path.join(DATA, sub, "*.json")):
            d = json.load(open(f))
            if "conditions" in d:                                  # e2-style summary
                conds = [keep(c, ["snr_target", "n_neurons", "r2_final", "onestep_r2",
                                  "rate_corr", "refresh_K"]) for c in d["conditions"]]
                write(f, {"conditions": conds, **keep(d, ["refresh_K"])})
            else:                                                  # exp_e1-style per-arm
                slim = keep(d, ["arm", "seed", "snr_realized", "r2_final", "onestep_r2",
                                "onestep_r2_fixed", "kpred_r2"])
                slim["drift"] = [keep(e, ["angle_prev_deg", "angle_oracle_deg"])
                                 for e in d.get("drift", [])]      # only the two fields used
                write(f, slim)
    # tau sweep (E3): per-condition rate_corr + filtered R^2
    for f in glob.glob(os.path.join(DATA, "e3", "*.json")):
        d = json.load(open(f))
        conds = [keep(c, ["snr_target", "n_neurons", "rate_corr", "r2_final"]) for c in d.get("conditions", [])]
        write(f, {"conditions": conds})
    # E6/E7 comparison curves (step + per-seed R^2 arrays): just round the floats
    for name in ("motivation_compare.json", "flow_compare.json"):
        p = os.path.join(DATA, "extra", name)
        if os.path.exists(p):
            write(p, json.load(open(p)))


if __name__ == "__main__":
    before = sum(os.path.getsize(p) for p in glob.glob(os.path.join(DATA, "*", "*.json")))
    slim()
    after = sum(os.path.getsize(p) for p in glob.glob(os.path.join(DATA, "*", "*.json")))
    print(f"paper/data: {before/1e6:.2f} MB -> {after/1e6:.2f} MB")
