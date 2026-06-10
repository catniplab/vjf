"""Scaled-back task: model a SINGLE drifting-grating direction with sVJF.

One direction (auto: strongest population response), its ~50 trials, 74 well-tuned
neurons, 10 ms bins, 0-1400 ms. Replay the 40 train trials for several EPOCHS
(per-trial latent reset; readout + flow keep learning across epochs). L=2 (a clean
limit cycle). Sweep epoch counts. Eval on 10 held-out trials: leave-one-neuron-out
PLL vs a stimulus-locked PSTH ceiling, the inferred latent cycle, and a free-run
forecast. The closest real-data analog to the synthetic limit-cycle benchmark.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask, kmeans_centers
from experiments.v1_graf.run_m1 import _infer_latent_paths
from experiments.v1_graf.eval import (
    leave_one_neuron_rates, predictive_ll_bits_per_spike, forecast_r2)
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter_trials

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
RESULTS = os.path.join(HERE, "results")
BIN_MS, T_MAX_MS, SEED = 10.0, 1400.0, 20260609
WARMUP_TRIALS, N_RBF, HIDDEN = 8, 100, [64, 64]
matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 130, "savefig.bbox": "tight"})


def _train_eval(train_trials, test_trials, test_counts, ybar, epochs, latent_dim, N,
                grow=False, grow_thresh=0.5, max_rbf=400, grow_min_gap=1):
    torch.manual_seed(SEED)
    rep = train_trials * epochs
    cover_window = np.concatenate(train_trials[:WARMUP_TRIALS], 0)
    n_rbf = min(N_RBF, cover_window.shape[0])
    model = VJF.make_model(ydim=N, xdim=latent_dim, udim=0, n_rbf=n_rbf, hidden_sizes=HIDDEN,
                           likelihood="poisson", transition_flow="srrls", encoder="projection")
    if grow:
        model.transition.grow_rbf = True
        model.transition.grow_thresh = grow_thresh
        model.transition.max_rbf = max_rbf
        model.transition.grow_min_gap = grow_min_gap
    ro = OnlineReadout(N, latent_dim, smooth_tau=8.0, refresh_K=500, link="log")
    Cw, bw = ro.warm_start(cover_window)
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cw))
        model.decoder.decode.bias.copy_(torch.as_tensor(bw.reshape(-1)))
    loss_trace = []
    for res in online_filter_trials(model, rep, readout=ro, warmup_trials=WARMUP_TRIALS,
                                    seed_centers=lambda s: kmeans_centers(s, n_rbf)):
        if res.step % 100 == 0 and not res.diverged:
            loss_trace.append(res.loss)
    lam = np.stack([leave_one_neuron_rates(model, ro, tc) for tc in test_trials], 0)
    pll = predictive_ll_bits_per_spike(test_counts, lam, ybar)
    paths = _infer_latent_paths(model, ro, test_trials)
    k = min(50, paths[0].shape[0] - 1)
    fc = forecast_r2(model, paths[0][0], paths[0][1:k + 1], k)
    centers = model.transition.velocity.feature.centroid.detach().cpu().numpy()
    widths = np.exp(model.transition.velocity.feature.logwidth.detach().cpu().numpy())
    return dict(pll=float(pll), forecast_r2=float(fc), paths=paths, centers=centers,
                widths=widths, loss_trace=np.asarray(loss_trace),
                n_basis=int(model.transition.velocity.feature.n_basis))


def _plot_latent(res, epochs, d_star, pll, fc, pll_psth, suffix=""):
    paths, centers, widths = res["paths"], res["centers"], res["widths"]
    avg = np.mean([p for p in paths], 0)                       # (nt, 2) mean cycle
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color="0.75", lw=0.6, alpha=0.7)
    t = np.arange(avg.shape[0])
    ax.scatter(avg[:, 0], avg[:, 1], c=t, cmap="viridis", s=12, zorder=3)
    ax.plot(avg[:, 0], avg[:, 1], color="0.3", lw=0.8, zorder=2)
    sz = 15 + 200 * (widths - widths.min()) / (np.ptp(widths) + 1e-12)
    ax.scatter(centers[:, 0], centers[:, 1], s=sz, c="r", marker="x", alpha=0.5,
               label="RBF centers", zorder=1)
    ax.set_xlabel("latent 1"); ax.set_ylabel("latent 2"); ax.legend(fontsize=8)
    ax.set_title(f"dir {d_star:.0f} deg, L=2, {epochs} epochs\n"
                 f"PLL={pll:.3f} (PSTH ceiling {pll_psth:.3f}), forecast R2={fc:.2f}")
    fig.savefig(os.path.join(FIGS, f"single_dir_latent_E{epochs}{suffix}.png"))
    fig.savefig(os.path.join(FIGS, f"single_dir_latent_E{epochs}{suffix}.pdf")); plt.close(fig)


def main(epochs_list=(5, 20, 50), latent_dim=2, direction=None,
         grow=False, grow_thresh=0.5, max_rbf=400, grow_min_gap=1):
    os.makedirs(FIGS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    rng = np.random.default_rng(SEED)
    arr = load_array(5)
    counts = bin_spikes(arr["spk_times"], bin_ms=BIN_MS)
    dirs = arr["ori"]
    mask, _ = well_tuned_mask(counts, dirs)
    counts = counts[:, :, mask]
    N = counts.shape[2]
    nt = int(round(T_MAX_MS / BIN_MS))

    keep = np.unique(dirs)
    if direction is None:
        rate = {d: counts[dirs == d][:, :nt, :].mean() for d in keep}
        d_star = float(max(rate, key=rate.get))
    else:
        d_star = float(direction)
    idx = np.where(dirs == d_star)[0]
    rng.shuffle(idx)
    test_idx, train_idx = idx[:10], idx[10:]
    train_trials = [counts[i][:nt] for i in train_idx]
    test_trials = [counts[i][:nt] for i in test_idx]
    test_counts = np.stack(test_trials, 0)
    ybar = float(test_counts.mean())

    # stimulus-locked PSTH ceiling: each neuron's train-trial-averaged rate per bin
    psth = np.stack([counts[i][:nt] for i in train_idx], 0).mean(0)        # (nt, N)
    lam_psth = np.broadcast_to(psth, test_counts.shape)
    pll_psth = predictive_ll_bits_per_spike(test_counts, lam_psth, ybar)

    summary = {"direction_deg": d_star, "n_neurons": N, "n_train": len(train_trials),
               "n_test": len(test_trials), "latent_dim": latent_dim, "grow": grow,
               "grow_thresh": grow_thresh, "max_rbf": max_rbf,
               "pll_psth_ceiling": float(pll_psth), "epochs": {}}
    sfx = "_grow" if grow else ""
    for E in epochs_list:
        res = _train_eval(train_trials, test_trials, test_counts, ybar, E, latent_dim, N,
                          grow=grow, grow_thresh=grow_thresh, max_rbf=max_rbf,
                          grow_min_gap=grow_min_gap)
        _plot_latent(res, E, d_star, res["pll"], res["forecast_r2"], pll_psth, suffix=sfx)
        summary["epochs"][str(E)] = {"pll": res["pll"], "forecast_r2": res["forecast_r2"],
                                     "n_basis": res["n_basis"]}
        print(f"E={E:3d}  PLL={res['pll']:+.3f} (ceiling {pll_psth:+.3f})  "
              f"forecast R2={res['forecast_r2']:+.3f}  n_basis={res['n_basis']}")

    # summary figure: PLL + forecast vs epochs
    Es = list(epochs_list)
    plls = [summary["epochs"][str(E)]["pll"] for E in Es]
    fcs = [summary["epochs"][str(E)]["forecast_r2"] for E in Es]
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    ax[0].plot(Es, plls, "o-"); ax[0].axhline(pll_psth, ls="--", color="g", label="PSTH ceiling")
    ax[0].axhline(0, ls=":", color="k", lw=0.7, label="mean-rate baseline")
    ax[0].set_xlabel("epochs"); ax[0].set_ylabel("leave-1-neuron-out PLL (bits/spk)")
    ax[0].legend(fontsize=7); ax[0].set_title("Predictive log-likelihood")
    ax[1].plot(Es, fcs, "o-", color="C1"); ax[1].axhline(0, ls=":", color="k", lw=0.7)
    ax[1].set_xlabel("epochs"); ax[1].set_ylabel("free-run forecast R2")
    ax[1].set_title("Forecast")
    fig.suptitle(f"Single-direction sVJF (dir {d_star:.0f} deg, L=2): metrics vs epochs")
    fig.savefig(os.path.join(FIGS, f"single_dir_summary{sfx}.png"))
    fig.savefig(os.path.join(FIGS, f"single_dir_summary{sfx}.pdf")); plt.close(fig)

    with open(os.path.join(RESULTS, f"single_dir_summary{sfx}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"direction={d_star:.0f} deg, PSTH ceiling PLL={pll_psth:+.3f}; figs -> {FIGS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=[5, 20, 50])
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--direction", type=float, default=None)
    ap.add_argument("--grow", action="store_true")
    ap.add_argument("--grow-thresh", type=float, default=0.5)
    ap.add_argument("--max-rbf", type=int, default=400)
    ap.add_argument("--grow-min-gap", type=int, default=1)
    args = ap.parse_args()
    main(epochs_list=tuple(args.epochs), latent_dim=args.latent_dim, direction=args.direction,
         grow=args.grow, grow_thresh=args.grow_thresh, max_rbf=args.max_rbf,
         grow_min_gap=args.grow_min_gap)
