"""M1 driver: sVJF end-to-end on the Graf V1 dataset (array_5 @ 10 ms).

Single streaming pass with an unknown readout, then the core M1 metrics:
leave-one-neuron-out predictive log-likelihood (PLL), single-trial orientation
decoding, the orientation torus embedding, a free-run forecast R^2, and per-bin
timing. This is integration glue over already-tested building blocks
(``graf_loader``, ``vjf.realtime.online_filter_trials``, ``vjf.readout``,
``eval``); no new math.

FULL mode: all 72 directions x 50 trials/direction, well-tuned neurons (r2 >= 0.75;
~74 for array_5, threshold-sensitive),
n_rbf=200. QUICK mode (the smoke test): a handful of directions/trials and a
small fixed neuron subset, to finish in well under a couple of minutes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess

import numpy as np
import torch

from experiments.v1_graf.graf_loader import (
    STIM_MS, bin_spikes, kmeans_centers, load_array, well_tuned_mask)
from experiments.v1_graf.eval import (
    forecast_r2, leave_one_neuron_rates, orientation_decode_acc,
    predictive_ll_bits_per_spike, torus_embedding)
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter_trials

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SEED = 20260609


def _git(args):
    try:
        return subprocess.check_output(["git", *args], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def _provenance(cfg: dict) -> dict:
    return {
        "array_num": cfg["array_num"],
        "seed": SEED,
        "config": cfg,
        "vjf_commit": _git(["rev-parse", "HEAD"]),
        "vjf_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "vjf_dirty": _git(["status", "--porcelain"]) != "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }


def _split_trials(dirs: np.ndarray, n_test: int, n_train: int, rng: np.random.Generator):
    """Per-direction train/test split. Returns dicts {direction -> trial-index list}.

    n_test test trials and up to n_train train trials are kept per direction;
    None means "all the rest go to train" (FULL mode keeps all 50/direction)."""
    train, test = {}, {}
    for d in np.unique(dirs):
        idx = np.where(dirs == d)[0]
        rng.shuffle(idx)
        test[d] = idx[:n_test].tolist()
        rest = idx[n_test:]
        train[d] = rest.tolist() if n_train is None else rest[:n_train].tolist()
    return train, test


@torch.no_grad()
def _infer_latent_paths(model, readout, trial_counts_list):
    """TRULY frozen inference: per trial, run the per-sample filter with NO learning
    (sgd=False, update=False) and the frozen readout projection (mirrors
    leave_one_neuron_rates but keeps every neuron). Returns a list of (n_bin, xdim)
    latent-mean paths, one per trial. Posterior is reset to the prior each trial.

    EMA discipline: snapshot readout.nu at entry so every trial starts from the SAME
    fixed EMA state (order-independent, side-effect-free) - consistent with the
    leave_one_neuron_rates path."""
    nu_snapshot = readout.nu.copy()                              # freeze EMA entry state
    paths = []
    for tc in trial_counts_list:
        readout.nu = nu_snapshot.copy()                          # each trial sees same EMA
        q = None
        means = []
        for t in range(tc.shape[0]):
            y = tc[t]
            g = readout.feature(y, update_mean=False)
            x_enc = torch.as_tensor(readout.project(g))
            qt, *_ = model.filter(y, None, q, sgd=False, update=False, verbose=False,
                                  y_enc=x_enc)
            q = qt
            means.append(qt.mean.detach().cpu().numpy()[0])
        paths.append(np.asarray(means))
    readout.nu = nu_snapshot                                     # restore caller's EMA state
    return paths


def main(*, array_num: int = 5, bin_ms: float = 10.0, latent_dim: int = 3,
         quick: bool = False, n_dir: int = None, flow: str = "srrls", grow: bool = False,
         grow_weight_init: str = "zero", dyn_noise: float = 0.0, dyn_noise_period: int = 1000,
         dyn_noise_decay: float = 1.0, column_norm: str = "eig", optimizer: str = "sgd",
         lr: float = 1e-4, rbf_base: int = 25, max_rbf: int = None, refresh_k: int = 1000,
         seed: int = SEED) -> dict:
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # 1. Load + bin.
    arr = load_array(array_num)
    counts = bin_spikes(arr["spk_times"], bin_ms=bin_ms)     # (3600, n_bin, N)
    dirs_all = arr["ori"]
    n_stim_bins = int(round(STIM_MS / bin_ms))

    # 2. Neuron selection.
    if quick:
        # The tuning fit is unreliable on a tiny direction subset, so just keep the
        # highest mean-rate neurons (a cheap, robust proxy) for the smoke test.
        mean_rate = counts.sum((0, 1))
        keep_idx = np.argsort(mean_rate)[::-1][:20]
        counts = counts[:, :, keep_idx]
        r2 = None
    else:
        mask, r2 = well_tuned_mask(counts, dirs_all)
        counts = counts[:, :, mask]
    n_kept = counts.shape[2]

    # 3. Subsample directions/trials + (quick) truncate the bins.
    if quick:
        keep_dirs = np.unique(dirs_all)[:6]
        n_test, n_train = 2, 6
        n_bin_use = 60
        n_stim_bins = min(n_stim_bins, n_bin_use)
        n_rbf = 30
        hidden = [32, 32]
        counts = counts[:, :n_bin_use, :]
    else:
        all_dirs = np.unique(dirs_all)
        keep_dirs = all_dirs if n_dir is None else all_dirs[:n_dir]
        n_test, n_train = 10, 40
        n_rbf = 200
        hidden = [100, 100]

    dir_mask = np.isin(dirs_all, keep_dirs)
    counts = counts[dir_mask]
    dirs = dirs_all[dir_mask]

    # 4. Train/test split per direction.
    train_by_dir, test_by_dir = _split_trials(dirs, n_test, n_train, rng)

    # 5. Coverage warm-up: the FIRST coverage_per_dir train trial(s) of EACH direction,
    #    placed FIRST in the trial list. warmup_trials = size of the coverage set.
    coverage_per_dir = 1
    coverage_idx, rest_train_idx = [], []
    for d in keep_dirs:
        tr = train_by_dir[d]
        coverage_idx += tr[:coverage_per_dir]
        rest_train_idx += tr[coverage_per_dir:]
    warmup_trials = len(coverage_idx)
    train_idx = coverage_idx + rest_train_idx
    test_idx = [i for d in keep_dirs for i in test_by_dir[d]]

    train_trials = [counts[i] for i in train_idx]
    test_trials = [counts[i] for i in test_idx]
    test_dirs = dirs[test_idx]

    # n_rbf must not exceed the coverage-buffer state count.
    n_cov_states = sum(counts[i].shape[0] for i in coverage_idx)
    n_bin = counts.shape[1]
    n_dir_eff = len(keep_dirs)
    # grow: seed a small basis, grow via novelty to an O(2^d) cap (optionally hard-capped
    # by max_rbf so it does not explode at high latent_dim); fixed: the n_rbf basis.
    rbf_budget = max_rbf if max_rbf is not None else rbf_base * (2 ** latent_dim)
    max_rbf_eff = min(rbf_budget, n_cov_states)
    n_seed = min(max(8, 2 ** latent_dim), n_cov_states) if grow else min(n_rbf, n_cov_states)

    # 6. Build the projection sVJF (flow learner + optimizer configurable).
    model = VJF.make_model(ydim=n_kept, xdim=latent_dim, udim=0, n_rbf=n_seed,
                           hidden_sizes=hidden, likelihood="poisson",
                           transition_flow=flow, encoder="projection",
                           optimizer=optimizer, lr=lr)
    if grow:
        gap = max(1, (n_dir_eff * n_bin) // max(1, max_rbf_eff - n_seed))  # spread over ~1 trial/dir
        model.transition.grow_rbf = True
        model.transition.grow_thresh = 0.5
        model.transition.max_rbf = max_rbf_eff
        model.transition.grow_min_gap = gap
        model.transition.grow_weight_init = grow_weight_init
    model.dyn_noise = dyn_noise                            # denoising stabilization (0 = off)
    model.dyn_noise_period = dyn_noise_period
    model.dyn_noise_decay = dyn_noise_decay

    # 7. Readout warm-start from the concatenated coverage window -> decoder (C, b).
    ro = OnlineReadout(n_kept, latent_dim, smooth_tau=8.0, refresh_K=refresh_k, link="log",
                       column_norm=column_norm)
    cover_window = np.concatenate([counts[i] for i in coverage_idx], 0)   # (W, N)
    C, b = ro.warm_start(cover_window)
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(C))
        model.decoder.decode.bias.copy_(torch.as_tensor(b.reshape(-1)))

    # 8. Stream the train trials (coverage first); collect online-phase timing + diverge.
    seed_fn = (lambda s: kmeans_centers(s, n_seed)) if flow in ("srrls", "sgd") else None
    elapsed_online, n_diverge = [], 0
    for res in online_filter_trials(model, train_trials, readout=ro,
                                    warmup_trials=warmup_trials, seed_centers=seed_fn):
        if res.trial >= warmup_trials:
            elapsed_online.append(res.elapsed_s)
        if res.diverged:
            n_diverge += 1

    # 9. Freeze (no more readout adaptation) and evaluate on the held-out test trials.
    #    9a. Leave-one-neuron-out PLL.
    lam_stacked = np.stack([leave_one_neuron_rates(model, ro, tc) for tc in test_trials], 0)
    test_counts = np.stack(test_trials, 0)
    ybar = float(test_counts.mean())
    pll = predictive_ll_bits_per_spike(test_counts, lam_stacked, ybar)

    #    9b. Per-test-trial latent paths (frozen) -> stimulus-window summary -> decode + torus.
    latent_paths = _infer_latent_paths(model, ro, test_trials)
    latent_per_trial = np.asarray([p[:n_stim_bins].mean(0) for p in latent_paths])
    # The split is balanced, so the smallest per-direction test count == n_test; cap
    # n_splits by it so cross_val_score never asks for more folds than samples/class.
    labels = np.round(test_dirs).astype(int)
    smallest_class = int(np.unique(labels, return_counts=True)[1].min())
    n_splits = max(2, min(5, smallest_class))
    decode_acc = orientation_decode_acc(latent_per_trial, test_dirs,
                                        n_splits=n_splits, seed=seed)
    torus, torus_axis = torus_embedding(latent_per_trial, test_dirs)

    #    9c. Forecast: free-run the flow over the first test trial's stimulus-window path.
    rep_means = latent_paths[0][:n_stim_bins]
    k = min(50, len(rep_means) - 1)
    fc_r2 = forecast_r2(model, rep_means[0], rep_means[1:k + 1], k) if k > 0 else float("nan")

    #    9d. Timing.
    elapsed = np.asarray(elapsed_online) if elapsed_online else np.array([float("nan")])
    median_ms = float(1000.0 * np.median(elapsed))
    p95_ms = float(1000.0 * np.percentile(elapsed, 95))

    cfg = {"array_num": array_num, "bin_ms": bin_ms, "latent_dim": latent_dim,
           "quick": quick, "n_seed_rbf": int(n_seed), "max_rbf_eff": int(max_rbf_eff),
           "hidden_sizes": hidden, "flow": flow, "grow": grow,
           "grow_weight_init": grow_weight_init, "dyn_noise": dyn_noise,
           "dyn_noise_period": dyn_noise_period, "dyn_noise_decay": dyn_noise_decay,
           "column_norm": column_norm, "optimizer": optimizer, "lr": lr, "refresh_k": refresh_k,
           "n_basis_final": int(model.transition.velocity.feature.n_basis), "seed": int(seed),
           "n_dir": int(len(keep_dirs)), "n_train_trials": len(train_trials),
           "n_test_trials": len(test_trials), "coverage_per_dir": coverage_per_dir,
           "warmup_trials": warmup_trials, "n_bin": int(counts.shape[1]),
           "n_stim_bins": int(n_stim_bins)}

    out = {
        "pll_bits_per_spike": float(pll),
        "decode_acc": float(decode_acc),
        "median_ms_per_bin": median_ms,
        "p95_ms_per_bin": p95_ms,
        "forecast_r2": float(fc_r2),
        "n_neurons": int(n_kept),
        "latent_dim": int(latent_dim),
        "bin_ms": float(bin_ms),
        "n_diverge": int(n_diverge),
        "provenance": _provenance(cfg),
    }
    if r2 is not None:
        out["n_well_tuned"] = int(np.sum(r2 >= 0.75))
    # The torus array travels under a private key so main() can stay a single dict;
    # __main__ pops it out into the saved JSON (it is not a scalar metric).
    out["_torus"] = np.asarray(torus).tolist()
    out["_torus_axis"] = np.asarray(torus_axis).tolist()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="sVJF M1 driver on the Graf V1 dataset")
    ap.add_argument("--array-num", type=int, default=5)
    ap.add_argument("--bin-ms", type=float, default=10.0)
    ap.add_argument("--latent-dim", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-dir", type=int, default=None)
    ap.add_argument("--flow", type=str, default="srrls", choices=["srrls", "sgd", "rls"])
    ap.add_argument("--grow", action="store_true")
    ap.add_argument("--grow-weight-init", type=str, default="zero", choices=["zero", "residual"])
    ap.add_argument("--dyn-noise", type=float, default=0.0)
    ap.add_argument("--dyn-noise-decay", type=float, default=1.0)
    ap.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--column-norm", type=str, default="eig", choices=["eig", "unit"])
    args = ap.parse_args()

    out = main(array_num=args.array_num, bin_ms=args.bin_ms,
               latent_dim=args.latent_dim, quick=args.quick, n_dir=args.n_dir,
               flow=args.flow, grow=args.grow, grow_weight_init=args.grow_weight_init,
               dyn_noise=args.dyn_noise, dyn_noise_decay=args.dyn_noise_decay,
               optimizer=args.optimizer, lr=args.lr, column_norm=args.column_norm)

    print("=== sVJF M1 ===")
    print(f"  array_{args.array_num} @ {args.bin_ms} ms, L={args.latent_dim}, "
          f"N={out['n_neurons']} neurons")
    print(f"  PLL          : {out['pll_bits_per_spike']:.4f} bits/spike")
    print(f"  decode acc   : {out['decode_acc']:.3f}")
    print(f"  forecast R^2 : {out['forecast_r2']:.3f}")
    print(f"  timing       : median {out['median_ms_per_bin']:.3f} ms/bin, "
          f"p95 {out['p95_ms_per_bin']:.3f} ms/bin")
    print(f"  diverged     : {out['n_diverge']}")

    if not args.quick:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(
            RESULTS_DIR,
            f"m1_array{args.array_num}_bin{args.bin_ms}_L{args.latent_dim}.json")
        out_save = dict(out)
        out_save["torus"] = out_save.pop("_torus")
        out_save["torus_axis"] = out_save.pop("_torus_axis")
        with open(path, "w") as fh:
            json.dump(out_save, fh, indent=2)
        print(f"  saved        : {path}")
