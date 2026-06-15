"""Quick local bracket for the R2 curvature-penalty weight lambda_smooth.

Trains the winner capacity region (L=4, 1600 ctr, E short) at a log-sweep of lambda with
denoising OFF, and reports for each: test PLL, forecast S_persist, the field curvature R2 on
the test states (what we penalize), and the free-run JAGGEDNESS (mean ||2nd difference|| of a
2-cycle free-run = how smooth the autonomous trajectory actually is). Picks the lambda range
that smooths the free-run without killing reconstruction, to scale the 8-VM search.

Run: uv run python -m experiments.v1_graf.lambda_bracket
"""
from __future__ import annotations
import numpy as np
import torch

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval
from experiments.v1_graf.forecast_video import filtered_path

T0, KMAX = 16, 32                                  # forecast launch (1 cycle), 2-cycle horizon
EPOCHS = 30                                        # short -- relative smoothing shows early


def jaggedness(model, ro, tc):
    """Mean norm of the discrete 2nd difference of a 2-cycle free-run from t0 (lower=smoother)."""
    xfilt, _ = filtered_path(model, ro, tc)
    with torch.no_grad():
        x, _ = model.forecast(torch.as_tensor(xfilt[T0][None].astype(np.float32)), n_step=KMAX)
    fc = x.detach().cpu().numpy()[:, 0, :]
    d2 = fc[2:] - 2 * fc[1:-1] + fc[:-2]
    return float(np.linalg.norm(d2, axis=1).mean())


def main():
    torch.set_default_dtype(torch.float32)
    data = prepare_single_dir_data(direction=225.0, n_val=0)   # 40 train / 10 test
    ybar = float(data["test_counts"].mean())
    tc0 = data["test_trials"][0]
    print(f"dir 225, N={data['N']}, PSTH ceiling PLL={data['pll_psth']:.3f}, "
          f"{len(data['train_trials'])} train / {len(data['test_trials'])} test, E={EPOCHS}")
    print(f"{'lambda':>10} | {'PLL':>7} | {'S_persist':>9} | {'fieldR2':>10} | {'free-run jag':>12}")
    for lam in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        res = _train_eval(
            data["train_trials"], data["test_trials"], data["test_counts"], ybar,
            epochs=EPOCHS, latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
            flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
            dyn_noise=0.0, smooth_lambda=lam, psth_counts=data["psth_counts"], return_model=True)
        model, ro = res["model"], res["ro"]
        with torch.no_grad():
            xf = torch.as_tensor(np.stack([filtered_path(model, ro, t)[0]
                                           for t in data["test_trials"]], 0).reshape(-1, 4).astype(np.float32))
            fieldR2 = float(model.transition.curvature_penalty(xf))
        jag = jaggedness(model, ro, tc0)
        print(f"{lam:>10.0e} | {res['pll']:>7.3f} | {res['fc_weighted_persist_skill']:>+9.4f} | "
              f"{fieldR2:>10.2f} | {jag:>12.4f}", flush=True)


if __name__ == "__main__":
    main()
