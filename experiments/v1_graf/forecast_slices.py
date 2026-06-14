"""Forecasting visualization: free-run the learned flow from several start times along a
held-out trial's inferred latent path, and overlay each forecast on the true path (time
axis). Shows the predictive horizon from different phases of the cycle.

Single direction, the stable recipe (sgd grow flow + residual init + denoising noise).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval, FIGS
from experiments.v1_graf.figstyle import set_style, FW

E = 20
STARTS = (0, 16, 32, 48, 64, 80)        # one per grating cycle (~16 bins)
K = 40                                  # forecast horizon (bins)


def main():
    set_style()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    d = prepare_single_dir_data(None)
    res = _train_eval(d["train_trials"], d["test_trials"], d["test_counts"], d["ybar"], E, 3,
                      d["N"], flow="sgd", optimizer="adam", lr=1e-4, grow=True,
                      grow_weight_init="residual", dyn_noise=0.2, dyn_noise_decay=0.97,
                      return_model=True)
    model = res["model"]; p = np.asarray(res["paths"][0])    # (nt, 3) inferred latent path
    nt, L = p.shape
    t = np.arange(nt)

    fig, ax = plt.subplots(L, 1, figsize=(FW(1.0), 4.4), sharex=True)
    cmap = plt.get_cmap("viridis")
    for t0 in STARTS:
        k = min(K, nt - 1 - t0)
        if k <= 1:
            continue
        x, _ = model.forecast(torch.as_tensor(p[t0][None].astype(np.float32)), n_step=k)
        fr = x.detach().cpu().numpy()[:, 0, :]               # (k+1, L) free-run from p[t0]
        for j in range(L):
            ax[j].plot(np.arange(t0, t0 + k + 1), fr[:, j], color=cmap(t0 / max(STARTS)), lw=1.3)
            ax[j].plot(t0, p[t0, j], "o", color=cmap(t0 / max(STARTS)), ms=4)
    for j in range(L):
        ax[j].plot(t, p[:, j], color="0.35", lw=1.0, ls="--", zorder=0)   # inferred path (truth)
        ax[j].set_ylabel(f"latent $x_{j+1}$")
    ax[0].plot([], [], color="0.35", ls="--", label="inferred path")
    ax[0].plot([], [], color=cmap(0.6), lw=1.3, label="free-run forecast")
    ax[0].legend(fontsize=7, ncol=2)
    ax[-1].set_xlabel("time in trial (bins, 10 ms)")
    fig.suptitle(f"free-run forecasts launched at successive cycle phases "
                 f"(dir {d['d_star']:.0f} deg, L=3, sgd grow + noise; color = start time)", fontsize=9)
    out = os.path.join(FIGS, "forecast_slices.png")
    fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
    print(f"forecast R2 (whole)={res['forecast_r2']:+.3f}; starts={STARTS}; fig -> {out}")


if __name__ == "__main__":
    main()
