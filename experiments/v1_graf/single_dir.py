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
import copy
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask, kmeans_centers
from experiments.v1_graf.figstyle import set_style
from experiments.v1_graf.run_m1 import _infer_latent_paths
from experiments.v1_graf.eval import (
    leave_one_neuron_rates, predictive_ll_bits_per_spike, forecast_r2,
    forecast_reconstruction_deviance, forecast_skill_summary)
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter_trials

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
RESULTS = os.path.join(HERE, "results")
BIN_MS, T_MAX_MS, SEED = 10.0, 1400.0, 20260609
WARMUP_TRIALS, N_RBF, HIDDEN = 8, 100, [64, 64]


def prepare_single_dir_data(direction=None, n_val=0):
    """Load array_5, keep well-tuned neurons, pick one direction (auto: strongest mean
    response), split 10 test / ``n_val`` val / rest train trials, and compute the
    stimulus-locked PSTH (the forecast/ceiling baseline). The search selects on the val
    split, the clean experiment reports on test; val and test are never trained on.
    Shared by the driver and the scan."""
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
    test_idx, val_idx, train_idx = idx[:10], idx[10:10 + n_val], idx[10 + n_val:]
    train_trials = [counts[i][:nt] for i in train_idx]
    test_trials = [counts[i][:nt] for i in test_idx]
    val_trials = [counts[i][:nt] for i in val_idx]
    test_counts = np.stack(test_trials, 0)
    val_counts = np.stack(val_trials, 0) if val_trials else np.empty((0, nt, N))
    ybar = float(test_counts.mean())
    psth = np.stack([counts[i][:nt] for i in train_idx], 0).mean(0)        # (nt, N) train PSTH
    pll_psth = predictive_ll_bits_per_spike(test_counts, np.broadcast_to(psth, test_counts.shape), ybar)
    return dict(train_trials=train_trials, test_trials=test_trials, test_counts=test_counts,
                val_trials=val_trials, val_counts=val_counts,
                ybar=ybar, N=N, d_star=d_star, pll_psth=float(pll_psth), psth_counts=psth)


def _forecast_skill(model, ro, eval_trials, psth_counts):
    """Gold-standard selection metric on the held-out (val/test) trials: future forecasted
    reconstruction -- free-run the flow from many start phases, decode, and score the
    forecast spikes by Poisson-deviance skill vs persistence + the stimulus-locked PSTH.
    ``psth_counts`` is the single-direction train PSTH (nt, N)."""
    n_bin = eval_trials[0].shape[0]
    starts = list(range(n_bin // 8, n_bin - 32, 8)) or [0]
    devs = [forecast_reconstruction_deviance(model, ro, tc, psth_counts, starts, (8, 16, 32))
            for tc in eval_trials]
    return forecast_skill_summary(devs)


def _eval_model(model, ro, test_trials, test_counts, ybar):
    """PLL + inferred latent paths + a free-run forecast (and its trajectory) for the
    current model. Mutates ro's EMA state, so callers that need to keep training pass a
    deepcopy (snapshots) -- the end-of-training call can use the live model."""
    lam = np.stack([leave_one_neuron_rates(model, ro, tc) for tc in test_trials], 0)
    pll = predictive_ll_bits_per_spike(test_counts, lam, ybar)
    paths = _infer_latent_paths(model, ro, test_trials)
    k = min(50, paths[0].shape[0] - 1)
    fc = forecast_r2(model, paths[0][0], paths[0][1:k + 1], k)
    x, _ = model.forecast(torch.as_tensor(paths[0][0][None].astype(np.float32)), n_step=k)
    freerun = x.detach().cpu().numpy()[:, 0, :]            # (k+1, m) free-run from trial-0 start
    return float(pll), float(fc), paths, freerun


def _train_eval(train_trials, test_trials, test_counts, ybar, epochs, latent_dim, N,
                grow=False, grow_thresh=0.5, max_rbf=None, grow_min_gap=None, refresh_k=500,
                column_norm="eig", rbf_base=25, width_scale=0.5, flow="srrls",
                snapshot_epochs=(), optimizer="sgd", lr=1e-4, grow_weight_init="zero", seed=SEED,
                dyn_noise=0.0, dyn_noise_period=1000, dyn_noise_decay=1.0,
                dyn_noise_fit_ref=0.0, probe_fn=None, return_model=False, psth_counts=None):
    torch.manual_seed(seed)
    rng_order = np.random.default_rng(seed + 777)              # randomize replay order each epoch
    rep = []
    for _ in range(epochs):
        rep += [train_trials[i] for i in rng_order.permutation(len(train_trials))]
    steps_per_epoch = len(train_trials) * train_trials[0].shape[0]
    cover_window = np.concatenate(train_trials[:WARMUP_TRIALS], 0)
    # Data-driven seeding + growth now work for the sgd flow too (VJF.filter re-points
    # the optimizer at grown weights); 'rls' keeps a fixed basis.
    if flow not in ("srrls", "sgd"):
        grow = False
    # O(2^d) RBF budget (curse of dimensionality): grow seeds small + adds via novelty up
    # to the cap; fixed seeds the whole O(2^d) basis at once (same target size, for a fair
    # fixed-vs-grow comparison).
    max_rbf_eff = max_rbf if max_rbf is not None else rbf_base * (2 ** latent_dim)
    n_seed = int(min(max(8, 2 ** latent_dim), cover_window.shape[0])) if grow else \
        int(min(max_rbf_eff, cover_window.shape[0]))
    wscale = width_scale if grow else 1.0                  # narrower seed only when growing
    model = VJF.make_model(ydim=N, xdim=latent_dim, udim=0, n_rbf=n_seed, hidden_sizes=HIDDEN,
                           likelihood="poisson", transition_flow=flow, encoder="projection",
                           optimizer=optimizer, lr=lr)
    if grow:
        gap = (grow_min_gap if grow_min_gap is not None
               else max(1, steps_per_epoch // max(1, max_rbf_eff - n_seed)))
        model.transition.grow_rbf = True
        model.transition.grow_thresh = grow_thresh
        model.transition.max_rbf = max_rbf_eff
        model.transition.grow_min_gap = gap
        model.transition.grow_weight_init = grow_weight_init
    model.dyn_noise = dyn_noise                            # denoising stabilization (0 = off)
    model.dyn_noise_period = dyn_noise_period
    model.dyn_noise_decay = dyn_noise_decay
    model.dyn_noise_fit_ref = dyn_noise_fit_ref            # gate noise by flow fit (0 = off)
    ro = OnlineReadout(N, latent_dim, smooth_tau=8.0, refresh_K=refresh_k, link="log",
                       column_norm=column_norm)
    Cw, bw = ro.warm_start(cover_window)
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cw))
        model.decoder.decode.bias.copy_(torch.as_tensor(bw.reshape(-1)))
    seed = (lambda s: kmeans_centers(s, n_seed, width_scale=wscale)) if flow in ("srrls", "sgd") else None
    # res.step is 0-based, so the last sample of epoch e is e*steps_per_epoch - 1; schedule
    # there (else the final/E-th checkpoint is never reached and is silently dropped).
    snap_steps = sorted(int(e * steps_per_epoch) - 1 for e in snapshot_epochs)
    snaps, si, loss_trace = [], 0, []
    for res in online_filter_trials(model, rep, readout=ro, warmup_trials=WARMUP_TRIALS,
                                    seed_centers=seed):
        if res.step % 100 == 0 and not res.diverged:
            loss_trace.append(res.loss)
        if si < len(snap_steps) and res.step >= snap_steps[si]:    # snapshot on a deepcopy
            si += 1
            m2 = copy.deepcopy(model)                              # weights intact (eval is forward-only)
            pll_s, fc_s, paths_s, fr_s = _eval_model(
                m2, copy.deepcopy(ro), test_trials, test_counts, ybar)
            snap = dict(epoch=res.step / steps_per_epoch, pll=pll_s, fc=fc_s,
                        paths=paths_s, freerun=fr_s,
                        n_basis=int(model.transition.velocity.feature.n_basis))
            if probe_fn is not None:                              # extra mechanistic diagnostics
                snap.update(probe_fn(m2, paths_s, fr_s))
            snaps.append(snap)
    pll, fc, paths, _ = _eval_model(model, ro, test_trials, test_counts, ybar)
    centers = model.transition.velocity.feature.centroid.detach().cpu().numpy()
    widths = np.exp(model.transition.velocity.feature.logwidth.detach().cpu().numpy())
    out = dict(pll=float(pll), forecast_r2=float(fc), paths=paths, centers=centers,
               widths=widths, loss_trace=np.asarray(loss_trace), snapshots=snaps,
               n_basis=int(model.transition.velocity.feature.n_basis))
    if psth_counts is not None:                               # SELECTION METRIC (gold standard)
        fskill = _forecast_skill(model, ro, test_trials, psth_counts)
        pll_ceiling = predictive_ll_bits_per_spike(
            test_counts, np.broadcast_to(psth_counts, test_counts.shape), ybar)
        out["forecast_skill"] = fskill
        out["fc_weighted_persist_skill"] = float(fskill["weighted_persist_skill"])
        out["pll_psth_ceiling"] = float(pll_ceiling)
    if return_model:                                          # for downstream forecasting/probes
        out["model"], out["ro"] = model, ro
    return out


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
    ax.set_title(f"dir {d_star:.0f} deg, L={avg.shape[1]}, {epochs} epochs\n"
                 f"PLL={pll:.3f} (PSTH ceiling {pll_psth:.3f}), forecast R2={fc:.2f}")
    fig.savefig(os.path.join(FIGS, f"single_dir_latent_E{epochs}{suffix}.png"))
    fig.savefig(os.path.join(FIGS, f"single_dir_latent_E{epochs}{suffix}.pdf")); plt.close(fig)


def _plot_snapshots(snaps, d_star, pll_psth, latent_dim, flow, suffix=""):
    """Small multiples: inferred test cycle (viridis) + free-run forecast (red) at a series
    of training checkpoints from ONE run -- shows when/how the free-run flow destabilizes."""
    n = len(snaps); ncol = min(5, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow), squeeze=False)
    lim = 3.0                                              # fixed frame so divergence is visible
    for j, s in enumerate(snaps):
        ax = axes[j // ncol][j % ncol]
        for p in s["paths"]:
            ax.plot(p[:, 0], p[:, 1], color="0.8", lw=0.4, alpha=0.7)
        avg = np.mean(s["paths"], 0)
        ax.scatter(avg[:, 0], avg[:, 1], c=np.arange(avg.shape[0]), cmap="viridis", s=5, zorder=3)
        fr = s["freerun"]
        ax.plot(fr[:, 0], fr[:, 1], color="C3", lw=1.2, zorder=4)             # free run
        ax.plot(fr[0, 0], fr[0, 1], "o", color="C3", ms=4, zorder=5)
        ax.set_title(f"ep {s['epoch']:.0f}: R2={s['fc']:+.2f}, PLL={s['pll']:.2f}", fontsize=8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"single-dir sVJF latent + free-run snapshots over training "
                 f"(dir {d_star:.0f} deg, L={latent_dim}, flow={flow}; ceiling PLL {pll_psth:.2f})",
                 fontsize=10)
    fig.savefig(os.path.join(FIGS, f"single_dir_snapshots{suffix}.png"))
    fig.savefig(os.path.join(FIGS, f"single_dir_snapshots{suffix}.pdf")); plt.close(fig)


def main(epochs_list=(5, 20, 50), latent_dim=2, direction=None,
         grow=False, grow_thresh=0.5, max_rbf=None, grow_min_gap=None, refresh_k=500, tag="",
         column_norm="eig", rbf_base=25, width_scale=0.5, flow="srrls", snapshots=False,
         optimizer="sgd", lr=1e-4, grow_weight_init="zero"):
    os.makedirs(FIGS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    set_style()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    data = prepare_single_dir_data(direction)
    train_trials, test_trials = data["train_trials"], data["test_trials"]
    test_counts, ybar, N = data["test_counts"], data["ybar"], data["N"]
    d_star, pll_psth = data["d_star"], data["pll_psth"]

    summary = {"direction_deg": d_star, "n_neurons": N, "n_train": len(train_trials),
               "n_test": len(test_trials), "latent_dim": latent_dim, "grow": grow,
               "grow_thresh": grow_thresh, "max_rbf": max_rbf, "refresh_k": refresh_k,
               "column_norm": column_norm, "flow": flow, "optimizer": optimizer, "lr": lr,
               "grow_weight_init": grow_weight_init,
               "pll_psth_ceiling": float(pll_psth), "epochs": {}}
    sfx = ("_" + tag) if tag else ("_grow" if grow else "")

    if snapshots:                                          # one run, many checkpoints
        E = max(epochs_list)
        snap_eps = tuple(sorted({e for e in (1, 2, 3, 5, 8, 12, 20, 30, 50) if e <= E} | {E}))
        res = _train_eval(train_trials, test_trials, test_counts, ybar, E, latent_dim, N,
                          grow=grow, grow_thresh=grow_thresh, max_rbf=max_rbf,
                          grow_min_gap=grow_min_gap, refresh_k=refresh_k, column_norm=column_norm,
                          rbf_base=rbf_base, width_scale=width_scale, flow=flow,
                          snapshot_epochs=snap_eps, optimizer=optimizer, lr=lr,
                          grow_weight_init=grow_weight_init)
        _plot_snapshots(res["snapshots"], d_star, pll_psth, latent_dim, flow, suffix=sfx)
        for s in res["snapshots"]:
            print(f"  ep {s['epoch']:5.1f}: PLL={s['pll']:+.3f}  forecast R2={s['fc']:+.3f}  "
                  f"n_basis={s['n_basis']}")
        print(f"direction={d_star:.0f} deg, flow={flow}; snapshots -> "
              f"{FIGS}/single_dir_snapshots{sfx}.png")
        return

    for E in epochs_list:
        res = _train_eval(train_trials, test_trials, test_counts, ybar, E, latent_dim, N,
                          grow=grow, grow_thresh=grow_thresh, max_rbf=max_rbf,
                          grow_min_gap=grow_min_gap, refresh_k=refresh_k, column_norm=column_norm,
                          rbf_base=rbf_base, width_scale=width_scale, flow=flow,
                          optimizer=optimizer, lr=lr, grow_weight_init=grow_weight_init)
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
    fig.suptitle(f"single-dir sVJF (dir {d_star:.0f} deg, L={latent_dim}): metrics vs epochs")
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
    ap.add_argument("--max-rbf", type=int, default=None)       # None -> rbf_base * 2^latent_dim
    ap.add_argument("--grow-min-gap", type=int, default=None)  # None -> spread adds across epoch 1
    ap.add_argument("--refresh-k", type=int, default=500)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--column-norm", type=str, default="eig", choices=["eig", "unit"])
    ap.add_argument("--rbf-base", type=int, default=25)        # cap = rbf_base * 2^latent_dim
    ap.add_argument("--width-scale", type=float, default=0.5)
    ap.add_argument("--flow", type=str, default="srrls", choices=["srrls", "sgd", "rls"])
    ap.add_argument("--snapshots", action="store_true")        # one run, latent/forecast grid
    ap.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grow-weight-init", type=str, default="zero", choices=["zero", "residual"])
    args = ap.parse_args()
    main(epochs_list=tuple(args.epochs), latent_dim=args.latent_dim, direction=args.direction,
         grow=args.grow, grow_thresh=args.grow_thresh, max_rbf=args.max_rbf,
         grow_min_gap=args.grow_min_gap, refresh_k=args.refresh_k, tag=args.tag,
         column_norm=args.column_norm, rbf_base=args.rbf_base, width_scale=args.width_scale,
         flow=args.flow, snapshots=args.snapshots, optimizer=args.optimizer, lr=args.lr,
         grow_weight_init=args.grow_weight_init)
