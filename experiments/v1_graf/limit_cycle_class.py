"""Definitive classification: point attractor (focus) vs attracting limit cycle.

Two rigorous tests per model, beyond the noisy short free-run:

  1. Linear stability of the flow MAP F(x) = x + f(x) at its fixed point x* (Newton-solved from
     the orbit center). Eigenvalues of dF/dx = I + df/dx (finite-difference Jacobian):
       - all |lambda| < 1  -> x* is a stable focus; the free-run relaxes to a POINT (no cycle).
       - a complex pair |lambda| > 1 -> x* is unstable; trajectories spiral OUT, and if they
         settle at finite amplitude there is an attracting LIMIT CYCLE around x*.
  2. Amplitude return map: free-run 400 cycles from a dense grid of initial radii along the
     orbit direction; plot the tail amplitude vs the initial amplitude. A limit cycle collapses
     the whole grid onto one horizontal line at the ring radius (attraction from both sides); a
     focus collapses everything to ~0.

Also prints ||W_after - W_before|| to confirm each rollout actually changed the flow.
Run: uv run python -m experiments.v1_graf.limit_cycle_class
"""
from __future__ import annotations
import copy
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval
from experiments.v1_graf.forecast_video import filtered_path, T0, BINS_PER_CYCLE
from experiments.v1_graf.factor_order import decoded_variance_basis, project
from experiments.v1_graf.rollout import rollout_finetune
from experiments.v1_graf.rollout_test import test_accuracy
from experiments.v1_graf.rollout_diag import free_run, cycle_amplitude, per_cycle_ratio
from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
N_CYC = 400                                     # long free-run for the return map
TAIL = 40                                       # cycles averaged for tail amplitude
RADII = np.linspace(0.15, 2.5, 14)              # initial radii (x the t0 offset) for the return map

CONFIGS = [
    dict(name="before", rollout=False),
    dict(name="trialavg k48", target_mode="trialavg", k_ladder=[8, 16, 32, 48], clip=1.0, epochs=300),
    dict(name="cycle k96 c10", target_mode="cycle", k_ladder=[24, 48, 96], clip=10.0, epochs=250),
    dict(name="cycle k160 c10", target_mode="cycle", k_ladder=[32, 64, 128, 160], clip=10.0, epochs=250),
]


@torch.no_grad()
def velocity(model, x):
    """f(x) for x a (S, L) numpy array -> (S, L) numpy."""
    v = model.transition.velocity(torch.as_tensor(x.astype(np.float32)), sampling=False)
    return v.detach().cpu().numpy()


def map_jacobian_eig(model, x, h=1e-2):
    """Eigenvalues of the flow MAP F(x)=x+f(x) linearized at x (L,) -- finite-difference df/dx.
    Evaluated at the RING CENTER (settled free-run mean): an attracting limit cycle encloses an
    unstable spiral, so a complex pair has |lambda|>1; a point attractor has all |lambda|<1."""
    L = x.shape[0]
    J = np.zeros((L, L))
    for i in range(L):
        e = np.zeros(L); e[i] = h
        J[:, i] = (velocity(model, (x + e)[None])[0] - velocity(model, (x - e)[None])[0]) / (2 * h)
    return np.linalg.eigvals(np.eye(L) + J)


def main():
    torch.set_default_dtype(torch.float32)
    data = prepare_single_dir_data(direction=225.0, n_val=10)
    test, psth = data["test_trials"], data["psth_counts"]
    ybar = float(data["test_counts"].mean())
    res = _train_eval(data["train_trials"], test, data["test_counts"], ybar, epochs=100,
                      latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
                      flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
                      dyn_noise=0.0, smooth_lambda=0.0, psth_counts=psth, return_model=True)
    model0, ro = res["model"], res["ro"]
    W0 = model0.transition.velocity.w_mean.detach().cpu().numpy().copy()
    print(f"fitted one-step: test PLL {res['pll']:.3f} (ceil {data['pll_psth']:.3f})", flush=True)

    allt = list(data["train_trials"]) + list(data["val_trials"]) + list(test)
    xbar = np.stack([filtered_path(model0, ro, t)[0] for t in allt], 0).mean(0)
    C = model0.decoder.decode.weight.detach().cpu().numpy()
    W, fve, ctr = decoded_variance_basis(xbar, C)
    data_amp = float(cycle_amplitude(project(xbar, W, ctr)).mean())
    off = filtered_path(model0, ro, test[0])[0][T0] - ctr        # t0 offset = radial direction
    n_long = N_CYC * BINS_PER_CYCLE

    recs = []
    for cfg in CONFIGS:
        m = copy.deepcopy(model0)
        if cfg.get("rollout", True):
            print(f"[{cfg['name']}] rollout ...", flush=True)
            rollout_finetune(m, ro, data["train_trials"], epochs=cfg["epochs"], k_ladder=cfg["k_ladder"],
                             target_mode=cfg["target_mode"], clip=cfg["clip"], lr=1e-3)
        dW = float(np.linalg.norm(m.transition.velocity.w_mean.detach().cpu().numpy() - W0))
        # amplitude return map: tail amplitude vs initial radius (the global attractor test)
        init_amp, tail_amp = [], []
        settle = None
        for s in RADII:
            xr = free_run(m, ctr + s * off, n_long)               # (n_long+1, L) full-latent free-run
            amp = cycle_amplitude(project(xr, W, ctr))
            init_amp.append(amp[0]); tail_amp.append(float(amp[-TAIL:].mean()))
            if abs(s - 1.0) < 0.2 or settle is None:
                settle = xr[-TAIL * BINS_PER_CYCLE:].mean(0)       # ring center = settled free-run mean
        eig = map_jacobian_eig(m, settle)                         # linearize at the ring center
        rho = float(np.max(np.abs(eig)))
        s_metric, _ = test_accuracy(m, ro, test, psth)
        tail = np.array(tail_amp)
        ring = float(np.median(tail))
        collapsed = bool(tail.std() / (ring + 1e-9) < 0.15)       # all ICs -> one common amplitude
        if rho <= 1.0 or ring < 0.15 * data_amp:                  # stable center => point attractor
            verdict = "focus"
        elif collapsed:
            verdict = "limit cycle"
        else:
            verdict = "marginal"
        recs.append(dict(cfg=cfg, dW=dW, eig=eig, rho=rho, init_amp=np.array(init_amp),
                         tail_amp=tail, ring=ring, ring_data=ring / data_amp, S=s_metric, verdict=verdict))
        print(f"  {cfg['name']:15s} dW {dW:.3f} | center spectral radius {rho:.4f} "
              f"| eig|.| {np.sort(np.abs(eig))[::-1].round(3)} | ring {ring:.2f} = {ring/data_amp:.2f}x data "
              f"| collapsed {collapsed} | {verdict} | S {s_metric:+.4f}", flush=True)

    # ---- figure: row1 return maps, row2 eigenvalue spectra ----
    set_style()
    n = len(recs)
    fig, ax = plt.subplots(2, n, figsize=(FW(1.0), FW(1.0) * 0.52), squeeze=False)
    for ci, r in enumerate(recs):
        a = ax[0][ci]
        a.plot([0, data_amp * 2.6], [0, data_amp * 2.6], color="0.7", lw=0.8, ls=":")  # tail==init
        a.axhline(data_amp, color="#c8920a", lw=1.3, alpha=0.8)
        a.scatter(r["init_amp"], r["tail_amp"], color="#d6336c", s=14, zorder=5)
        a.set_title(f"{r['cfg']['name']}\nrho {r['rho']:.3f} | {r['verdict']}", fontsize=6.3)
        a.set_xlabel("initial amplitude"); a.set_ylabel("tail amplitude (400 cyc)")
        a.set_xlim(0); a.set_ylim(0)
        b = ax[1][ci]                                             # eigenvalues of the map vs unit circle
        th = np.linspace(0, 2 * np.pi, 100)
        b.plot(np.cos(th), np.sin(th), color="0.7", lw=0.8)
        b.scatter(r["eig"].real, r["eig"].imag, color="#1b6fd6", s=20, zorder=5)
        b.set_aspect("equal", "box"); b.set_title(f"map eigenvalues (|.|max {r['rho']:.3f})", fontsize=6.3)
        b.set_xlabel("Re"); b.set_ylabel("Im")
    fig.suptitle("Limit-cycle classification: amplitude return map (top) + flow-map eigenvalues (bottom)",
                 fontsize=8)
    fig.savefig(os.path.join(FIGS, "limit_cycle_class.png"), dpi=200)
    plt.close(fig)
    print(f"\nclass -> {FIGS}/limit_cycle_class.png", flush=True)


if __name__ == "__main__":
    main()
