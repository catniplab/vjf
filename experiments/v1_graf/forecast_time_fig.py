"""Regenerate Fig 5 (forecast_time): single-trial free-run of the current best config.

Fits the report's best one-step config (L=4, 1600 RBF, lambda=0 -- the same model shown as the
"before"/one-step case in the limit-cycle and phase-portrait figures), then renders the
decoded-variance-ordered factor time-slice on the best-forecasting held-out trial: trial-average
(gold), single-trial filtered (teal), autonomous free-run from t0 (crimson). Replaces the stale
pre-investigation forecast_time.png. Run: uv run python -m experiments.v1_graf.forecast_time_fig
"""
from __future__ import annotations
import numpy as np
import torch

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval
from experiments.v1_graf.forecast_video import filtered_path, render_slices, T0
from experiments.v1_graf.factor_order import decoded_variance_basis, project
from experiments.v1_graf.eval import forecast_reconstruction_deviance, forecast_skill_summary


def main():
    torch.set_default_dtype(torch.float32)
    data = prepare_single_dir_data(direction=225.0, n_val=10)
    test, psth = data["test_trials"], data["psth_counts"]
    ybar = float(data["test_counts"].mean())
    res = _train_eval(data["train_trials"], test, data["test_counts"], ybar, epochs=100,
                      latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
                      flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
                      dyn_noise=0.0, smooth_lambda=0.0, psth_counts=psth, return_model=True)
    model, ro = res["model"], res["ro"]
    print(f"fitted: test PLL {res['pll']:.3f} (ceil {data['pll_psth']:.3f}), n_basis {res['n_basis']}",
          flush=True)

    # best-forecasting held-out trial at t0 (same selection as the other free-run figures)
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
    pf, pc, pbar = project(xfilt, W, ctr), project(xfc, W, ctr), project(xbar, W, ctr)
    render_slices(pf, pc, pbar, fve, T0, {"latent_dim": 4}, {"n_basis": res["n_basis"]}, {}, bi,
                  name="forecast_time")
    print(f"forecast_time.png regenerated (best config, test trial #{bi})", flush=True)


if __name__ == "__main__":
    main()
