"""Does denoising noise stabilize the free-run flow? Forecast R2 over training for the
clearest collapser (adam/grow L=4 lr1e-4) with no noise vs a decaying raised sinusoid
(period 1000) at a few peak amplitudes. A stabilized config should stop bouncing/dying
and hold a high forecast across epochs.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import torch
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval, FIGS
from experiments.v1_graf.figstyle import set_style, FW

CHK = (1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50)
BASE = dict(latent_dim=4, optimizer="adam", lr=1e-4, grow=True, grow_weight_init="residual")
CONFIGS = [
    dict(label="no noise", **BASE),
    dict(label="sigma0.1 decay0.97", dyn_noise=0.1, dyn_noise_decay=0.97, **BASE),
    dict(label="sigma0.2 decay0.97", dyn_noise=0.2, dyn_noise_decay=0.97, **BASE),
    dict(label="sigma0.3 decay0.97", dyn_noise=0.3, dyn_noise_decay=0.97, **BASE),
    dict(label="sigma0.2 constant",  dyn_noise=0.2, dyn_noise_decay=1.0,  **BASE),
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
    kw = dict(cfg); label, ld = kw.pop("label"), kw.pop("latent_dim")
    res = _train_eval(d["train_trials"], d["test_trials"], d["test_counts"], d["ybar"],
                      max(CHK), ld, d["N"], flow="sgd", snapshot_epochs=CHK, **kw)
    return label, [(s["epoch"], s["fc"]) for s in res["snapshots"]]


def main():
    set_style()
    d = _data()
    with ProcessPoolExecutor(max_workers=min(len(CONFIGS), max(1, (os.cpu_count() or 2) - 1))) as ex:
        out = list(ex.map(_run, CONFIGS))
    fig, ax = plt.subplots(figsize=(FW(0.66), 3.2))
    for label, traj in out:
        ax.plot([e for e, _ in traj], [f for _, f in traj], marker="o", ms=3, label=label)
        print(label, " ".join(f"{e:.0f}:{f:+.2f}" for e, f in traj))
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("epoch"); ax.set_ylabel("free-run forecast R2"); ax.legend(fontsize=7)
    ax.set_title(f"denoising noise vs forecast stability (adam/grow L=4 lr1e-4, dir {d['d_star']:.0f})")
    out_png = os.path.join(FIGS, "noise_sweep.png")
    fig.savefig(out_png); fig.savefig(out_png.replace(".png", ".pdf"))
    print("fig ->", out_png)


if __name__ == "__main__":
    main()
