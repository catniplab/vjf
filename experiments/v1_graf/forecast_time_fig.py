"""Fig 5 (forecast_time): single-trial free-run, two directions side by side.

Left column dir 225 (strongest, cleanest -- the PSTH/trial-average tracks the single trial better
than the autonomous free-run); right column dir 105 (noisier -- the free-run beats the near-oracle
PSTH, cf. the generality figure). Rows are the two leading decoded-variance factors. In each panel:
trial-average latent (gold, = the PSTH-equivalent predictor), single-trial filtered (teal), and the
autonomous free-run forecast from t0 (crimson). Best one-step config (L=4, 1600 RBF, lambda=0).
Run: uv run python -m experiments.v1_graf.forecast_time_fig
"""
from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval
from experiments.v1_graf.forecast_video import filtered_path, T0, BINS_PER_CYCLE
from experiments.v1_graf.factor_order import decoded_variance_basis, project
from experiments.v1_graf.eval import forecast_reconstruction_deviance, forecast_skill_summary
from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
GOLD, TEAL, CRIM = "#c8920a", "#1b9e9a", "#d6336c"


def fit_and_project(direction):
    data = prepare_single_dir_data(direction=direction, n_val=10)
    test, psth = data["test_trials"], data["psth_counts"]
    ybar = float(data["test_counts"].mean())
    res = _train_eval(data["train_trials"], test, data["test_counts"], ybar, epochs=100,
                      latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
                      flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
                      dyn_noise=0.0, smooth_lambda=0.0, psth_counts=psth, return_model=True)
    model, ro = res["model"], res["ro"]
    print(f"dir {direction:.0f}: PLL {res['pll']:.3f} (ceil {data['pll_psth']:.3f})", flush=True)
    per = [(forecast_skill_summary([forecast_reconstruction_deviance(model, ro, tc, psth, [T0], (8, 16, 32))])
            ["weighted_persist_skill"], i) for i, tc in enumerate(test)]
    bi = max(per)[1]
    tc = test[bi]; Tn = tc.shape[0]
    xfilt, _ = filtered_path(model, ro, tc)
    with torch.no_grad():
        x, _ = model.forecast(torch.as_tensor(xfilt[T0][None].astype(np.float32)), n_step=Tn - 1 - T0)
    xfc = x.detach().cpu().numpy()[:, 0, :]
    allt = list(data["train_trials"]) + list(data["val_trials"]) + list(test)
    xbar = np.stack([filtered_path(model, ro, t)[0] for t in allt], 0).mean(0)
    C = model.decoder.decode.weight.detach().cpu().numpy()
    W, fve, ctr = decoded_variance_basis(xbar, C)
    return dict(pf=project(xfilt, W, ctr), pc=project(xfc, W, ctr), pbar=project(xbar, W, ctr),
                fve=fve, Tn=Tn, trial=bi)


def main():
    torch.set_default_dtype(torch.float32)
    cols = [(225.0, "dir 225 (strongest; PSTH wins)"),
            (105.0, "dir 105 (noisier; free-run beats PSTH)")]
    res = {d: fit_and_project(d) for d, _ in cols}

    set_style()
    fig, ax = plt.subplots(2, 2, figsize=(FW(1.0), 3.4), sharex=True)
    for j, (d, title) in enumerate(cols):
        r = res[d]
        t = np.arange(r["pf"].shape[0]); tfc = np.arange(T0, T0 + r["pc"].shape[0])
        for row in range(2):                                    # two leading factors
            a = ax[row][j]
            a.plot(t, r["pbar"][:, row], color=GOLD, lw=2.2, alpha=0.9,
                   label="trial-average ($\\approx$ PSTH)")
            a.plot(t, r["pf"][:, row], color=TEAL, lw=1.1, alpha=0.95, label="single-trial (filtered)")
            a.plot(tfc, r["pc"][:, row], color=CRIM, lw=1.8, label="free-run forecast")
            a.axvline(T0, color="0.5", ls="--", lw=0.8)
            if j == 0:
                a.set_ylabel(f"factor {row + 1}")
            if row == 0:
                a.set_title(f"{title}", fontsize=8)
        ax[1][j].set_xlabel("time (10 ms bins;  dashed = forecast start, end of cycle 2)")
    ax[0][0].legend(fontsize=6, loc="upper right", framealpha=0.9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"forecast_time.{ext}"), dpi=200)
    plt.close(fig)
    print(f"-> {FIGS}/forecast_time.png (dir 225 trial #{res[225.0]['trial']}, "
          f"dir 105 trial #{res[105.0]['trial']})", flush=True)


if __name__ == "__main__":
    main()
