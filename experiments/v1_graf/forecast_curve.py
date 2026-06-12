"""Free-run forecast R2 over training epochs for selected single-dir sVJF configs.

Runs each config once to E=50 with DENSE snapshot checkpoints (vs the scan's 3) and
plots forecast R2 vs epoch, so the trajectory (late bloomer? early collapse? stable?)
is visible. Configs are the scan winners.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import torch
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval, FIGS
from experiments.v1_graf.figstyle import set_style, FW

CHK = (1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50)
CONFIGS = [
    dict(label="adam/grow L=3 lr1e-4", latent_dim=3, optimizer="adam", lr=1e-4,
         grow=True, grow_weight_init="residual"),
    dict(label="adam/grow L=4 lr1e-4", latent_dim=4, optimizer="adam", lr=1e-4,
         grow=True, grow_weight_init="residual"),
    dict(label="sgd/grow L=4 lr5e-4", latent_dim=4, optimizer="sgd", lr=5e-4,
         grow=True, grow_weight_init="residual"),
    dict(label="adam/fixed L=3 lr5e-4", latent_dim=3, optimizer="adam", lr=5e-4,
         grow=False),
]

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        _DATA = prepare_single_dir_data(None)
    return _DATA


def _run(cfg):
    torch.set_num_threads(1)
    d = _data()
    kw = dict(cfg)
    label, ld = kw.pop("label"), kw.pop("latent_dim")
    res = _train_eval(d["train_trials"], d["test_trials"], d["test_counts"], d["ybar"],
                      max(CHK), ld, d["N"], flow="sgd", snapshot_epochs=CHK, **kw)
    return label, [(s["epoch"], s["fc"], s["pll"]) for s in res["snapshots"]]


def main():
    set_style()
    d = _data()
    with ProcessPoolExecutor(max_workers=min(len(CONFIGS), max(1, (os.cpu_count() or 2) - 1))) as ex:
        out = list(ex.map(_run, CONFIGS))

    fig, ax = plt.subplots(figsize=(FW(0.66), 3.2))
    for label, traj in out:
        eps = [e for e, _, _ in traj]
        fcs = [f for _, f, _ in traj]
        ax.plot(eps, fcs, marker="o", ms=3, label=label)
        print(label, " ".join(f"{e:.0f}:{f:+.2f}" for e, f, _ in traj))
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("epoch"); ax.set_ylabel("free-run forecast R2"); ax.legend(fontsize=7)
    ax.set_title(f"forecast R2 over training (dir {d['d_star']:.0f} deg, PSTH-ceiling PLL {d['pll_psth']:.2f})")
    out_png = os.path.join(FIGS, "forecast_over_epochs.png")
    fig.savefig(out_png); fig.savefig(out_png.replace(".png", ".pdf"))
    print("fig ->", out_png)


if __name__ == "__main__":
    main()
