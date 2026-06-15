"""Try multi-step (rollout) training of the flow to capture the autonomous limit cycle.

One-step ELBO training underfits the free-run: a contracting focus wins the
observation-corrected one-step objective yet its free-run decays. Here we fine-tune ONLY the
flow weights with a non-causal rollout loss on the buffered filtered trajectory (the eval stays
causal): from each steady-state start t, free-run the flow k steps and match the filtered means

    L_roll = (1/k) sum_t sum_{j=1..k} || xhat^j - mu_{t+j} ||^2,   xhat^j = xhat^{j-1} + f(xhat^{j-1})

(latent form). Compares test forecast accuracy and the free-run before vs after, and the value
of lowering the curvature penalty. Run: uv run python -m experiments.v1_graf.rollout_test
"""
from __future__ import annotations
import numpy as np
import torch

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval
from experiments.v1_graf.forecast_video import filtered_path
from experiments.v1_graf.eval import forecast_reconstruction_deviance, forecast_skill_summary

BINS_PER_CYCLE = 16
T0 = 2 * BINS_PER_CYCLE                      # steady-state starts begin after the onset transient


def test_accuracy(model, ro, trials, psth):
    """Weighted forecast accuracy vs persistence + per-k, on the held-out trials."""
    n_bin = trials[0].shape[0]
    starts = list(range(T0, n_bin - 32, 8)) or [T0]
    devs = [forecast_reconstruction_deviance(model, ro, tc, psth, starts, (8, 16, 32)) for tc in trials]
    s = forecast_skill_summary(devs)
    return s["weighted_persist_skill"], s["skill"]


def rollout_finetune(model, ro, train_trials, k=12, epochs=25, lr=5e-4, seed=20260615):
    """Fine-tune the flow weights so the autonomous free-run reproduces the filtered trajectory."""
    W = model.transition.velocity.w_mean
    opt = torch.optim.Adam([W], lr=lr)
    rng = np.random.default_rng(seed)
    mus = [torch.as_tensor(filtered_path(model, ro, tc)[0], dtype=torch.float32) for tc in train_trials]
    for ep in range(epochs):
        order = rng.permutation(len(mus))
        ep_loss = 0.0
        for i in order:
            mu = mus[i]                                  # (T, L) filtered means (targets, detached)
            starts = torch.arange(T0, mu.shape[0] - k)
            if len(starts) == 0:
                continue
            opt.zero_grad()
            x = mu[starts]                               # (S, L) free-run start states (batched over phases)
            loss = 0.0
            for j in range(1, k + 1):                    # vectorized free-run over all start phases
                x = x + model.transition.velocity(x, sampling=False)
                loss = loss + ((x - mu[starts + j]) ** 2).sum()
            loss = loss / (len(starts) * k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 1.0)
            opt.step()
            ep_loss += float(loss)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  rollout ep {ep:2d}: mean L_roll {ep_loss / len(mus):.4f}", flush=True)
    return model


def main():
    torch.set_default_dtype(torch.float32)
    data = prepare_single_dir_data(direction=225.0, n_val=10)
    psth, test, ybar = data["psth_counts"], data["test_trials"], float(data["test_counts"].mean())
    for lam in [1e-3, 1e-4]:
        print(f"\n=== base config L=4, 1600 RBF, E=100, lambda={lam:g} ===", flush=True)
        res = _train_eval(data["train_trials"], test, data["test_counts"], ybar, epochs=100,
                          latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
                          flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
                          dyn_noise=0.0, smooth_lambda=lam, psth_counts=psth, return_model=True)
        model, ro = res["model"], res["ro"]
        s0, k0 = test_accuracy(model, ro, test, psth)
        print(f"  BEFORE rollout: test PLL {res['pll']:.3f} (ceil {data['pll_psth']:.3f}) | "
              f"S_persist {s0:+.4f} | k8/16/32 {k0[8]['vs_persist']:+.3f}/{k0[16]['vs_persist']:+.3f}/{k0[32]['vs_persist']:+.3f}", flush=True)
        rollout_finetune(model, ro, data["train_trials"])
        s1, k1 = test_accuracy(model, ro, test, psth)
        print(f"  AFTER  rollout: S_persist {s1:+.4f} | k8/16/32 "
              f"{k1[8]['vs_persist']:+.3f}/{k1[16]['vs_persist']:+.3f}/{k1[32]['vs_persist']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
