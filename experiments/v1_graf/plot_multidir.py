"""Reproducible multi-direction summary figure from a scan_multidir*.json.

decode-accuracy-relative-to-chance, leave-one-neuron PLL, and single-trial forecast R2
vs direction count, one line per arm (mean over seeds), at L=3. Title-free (the report
caption carries it); rendered at the width it is included at (scale 1.0).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import FIGS
from experiments.v1_graf.figstyle import set_style, FW

ARM_STYLE = {                                              # consistent color/label per arm
    "srrls_base": ("C0", "srrls (scale-fix)"),
    "sgd_noise": ("C1", "sgd+noise (fixed)"),
    "sgd_grow_noise": ("C2", "sgd grow+noise"),
    "sgd_grow_refined": ("C3", "sgd grow refined"),
}


def main(json_path, out_name="scan_multidir_summary.png", latent_dim=3):
    set_style()
    d = json.load(open(json_path))
    mean = lambda xs: sum(xs) / len(xs)
    arms = [a for a in ARM_STYLE if any(r["arm"] == a for r in d)]
    ndirs = sorted({r["n_dir"] for r in d if r["latent_dim"] == latent_dim})

    def agg(arm, nd, key):
        xs = [r[key] for r in d if r["arm"] == arm and r["latent_dim"] == latent_dim and r["n_dir"] == nd]
        return mean(xs) if xs else None

    def series(arm, key, scale_by_dir=False):                 # skip missing (arm, n_dir) cells
        xs, ys = [], []
        for nd in ndirs:
            v = agg(arm, nd, key)
            if v is None:
                continue
            xs.append(nd); ys.append(v * nd if scale_by_dir else v)
        return xs, ys

    fig, ax = plt.subplots(1, 3, figsize=(FW(1.0), 2.7))
    for arm in arms:
        c, lab = ARM_STYLE[arm]
        x, y = series(arm, "decode_acc", scale_by_dir=True); ax[0].plot(x, y, "o-", color=c, label=lab)
        x, y = series(arm, "pll"); ax[1].plot(x, y, "o-", color=c)
        x, y = series(arm, "forecast_r2"); ax[2].plot(x, y, "o-", color=c)
    ax[0].axhline(1, color="0.6", lw=0.6, ls=":"); ax[0].set_title("decode / chance")
    ax[0].set_ylabel(r"$\times$ chance"); ax[0].legend(fontsize=6.5)
    ax[1].set_title("leave-1-neuron PLL"); ax[1].set_ylabel("bits/spk")
    ax[2].axhline(0, color="0.6", lw=0.6, ls=":"); ax[2].set_title("forecast R2 (single-trial)")
    for a in ax:
        a.set_xlabel("# directions"); a.set_xticks(ndirs)
    out = os.path.join(FIGS, out_name)
    fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
    print("fig ->", out, "| arms:", arms, "| ndirs:", ndirs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--out", default="scan_multidir_summary.png")
    ap.add_argument("--latent-dim", type=int, default=3)
    a = ap.parse_args()
    main(a.json_path, out_name=a.out, latent_dim=a.latent_dim)
