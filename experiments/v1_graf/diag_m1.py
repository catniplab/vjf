"""M1 diagnostic run: capture what the M1 metrics could not show.

Retrains sVJF on array_5 @ 10 ms (faithful to run_m1's FULL config) and dumps the
artifacts a report needs:
  - training convergence trace (ELBO components over the stream),
  - inferred test-trial latent paths (run A = dynamics ON, the real M1 model),
  - a NO-DYNAMICS control (run B: flow never turned on) inferred the same way,
  - per-trial latent summaries + single-trial-decode accuracy for A and B,
  - the torus embedding for A and B,
  - neuron orientation tuning curves + R^2.

The expensive leave-one-neuron-out PLL is skipped on purpose (already measured:
-3.50 in the FULL run). Saves a single .npz under results/ (gitignored).

Run: PYTHONPATH=. uv run python experiments/v1_graf/diag_m1.py [--n-train-per-dir 40]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from experiments.v1_graf.graf_loader import (
    STIM_MS, bin_spikes, kmeans_centers, load_array, tuning_curve, well_tuned_mask)
from experiments.v1_graf.eval import orientation_decode_acc, torus_embedding
from experiments.v1_graf.run_m1 import _infer_latent_paths, _split_trials
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter_trials

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SEED = 20260609


def _build(n_kept, latent_dim, n_rbf, hidden):
    return VJF.make_model(ydim=n_kept, xdim=latent_dim, udim=0, n_rbf=n_rbf,
                          hidden_sizes=hidden, likelihood="poisson",
                          transition_flow="srrls", encoder="projection")


def _readout_warmstart(n_kept, latent_dim, cover_window, model):
    ro = OnlineReadout(n_kept, latent_dim, smooth_tau=8.0, refresh_K=1000, link="log")
    C, b = ro.warm_start(cover_window)
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(C))
        model.decoder.decode.bias.copy_(torch.as_tensor(b.reshape(-1)))
    return ro


def main(n_train_per_dir=40, latent_dim=3):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    rng = np.random.default_rng(SEED)

    arr = load_array(5)
    counts = bin_spikes(arr["spk_times"], bin_ms=10.0)
    dirs_all = arr["ori"]
    n_stim_bins = int(round(STIM_MS / 10.0))

    mask, r2 = well_tuned_mask(counts, dirs_all)
    tc, tc_axis = tuning_curve(counts, dirs_all, stim_only=True)   # tuning of ALL neurons
    counts = counts[:, :, mask]
    n_kept = counts.shape[2]
    tc_kept = tc[mask]

    keep_dirs = np.unique(dirs_all)
    train_by_dir, test_by_dir = _split_trials(dirs_all, 10, n_train_per_dir, rng)
    coverage_idx, rest_train_idx = [], []
    for d in keep_dirs:
        coverage_idx += train_by_dir[d][:1]
        rest_train_idx += train_by_dir[d][1:]
    warmup_trials = len(coverage_idx)
    train_idx = coverage_idx + rest_train_idx
    test_idx = [i for d in keep_dirs for i in test_by_dir[d]]

    train_trials = [counts[i] for i in train_idx]
    test_trials = [counts[i] for i in test_idx]
    test_dirs = dirs_all[test_idx]
    n_cov_states = sum(counts[i].shape[0] for i in coverage_idx)
    n_rbf = min(200, n_cov_states)
    cover_window = np.concatenate([counts[i] for i in coverage_idx], 0)

    # ---- Run A: full sVJF (dynamics ON) -- the real M1 model ----
    torch.manual_seed(SEED)
    model_a = _build(n_kept, latent_dim, n_rbf, [100, 100])
    ro_a = _readout_warmstart(n_kept, latent_dim, cover_window, model_a)
    steps, loss, recon, dyn, ent = [], [], [], [], []
    for res in online_filter_trials(model_a, train_trials, readout=ro_a,
                                    warmup_trials=warmup_trials,
                                    seed_centers=lambda s: kmeans_centers(s, n_rbf)):
        if res.step % 50 == 0 and not res.diverged:
            steps.append(res.step); loss.append(res.loss)
            recon.append(res.recon); dyn.append(res.dynamics); ent.append(res.entropy)
    paths_a = _infer_latent_paths(model_a, ro_a, test_trials)
    summ_a = np.asarray([p[:n_stim_bins].mean(0) for p in paths_a])
    dec_a = orientation_decode_acc(summ_a, test_dirs, n_splits=5, seed=SEED)
    torus_a, torus_axis = torus_embedding(summ_a, test_dirs)

    # ---- Run B: NO-DYNAMICS control (flow never turns on) ----
    torch.manual_seed(SEED)
    model_b = _build(n_kept, latent_dim, n_rbf, [100, 100])
    ro_b = _readout_warmstart(n_kept, latent_dim, cover_window, model_b)
    # warmup_trials = ALL train trials -> the boundary init is never reached, dynamics stay off.
    for _ in online_filter_trials(model_b, train_trials, readout=ro_b,
                                  warmup_trials=len(train_trials)):
        pass
    paths_b = _infer_latent_paths(model_b, ro_b, test_trials)
    summ_b = np.asarray([p[:n_stim_bins].mean(0) for p in paths_b])
    dec_b = orientation_decode_acc(summ_b, test_dirs, n_splits=5, seed=SEED)
    torus_b, _ = torus_embedding(summ_b, test_dirs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "diag_m1_array5_L%d.npz" % latent_dim)
    # store a few representative single-trial paths per a handful of directions
    rep_dirs = keep_dirs[::12]                      # 0,60,120,180,240,300 deg
    rep = {}
    for d in rep_dirs:
        ti = [j for j, dd in enumerate(test_dirs) if dd == d][:5]
        rep["pathsA_%d" % int(d)] = np.asarray([paths_a[j][:n_stim_bins] for j in ti])
        rep["pathsB_%d" % int(d)] = np.asarray([paths_b[j][:n_stim_bins] for j in ti])
    np.savez(out,
             steps=np.asarray(steps), loss=np.asarray(loss), recon=np.asarray(recon),
             dynamics=np.asarray(dyn), entropy=np.asarray(ent),
             summ_a=summ_a, summ_b=summ_b, test_dirs=test_dirs,
             dec_a=dec_a, dec_b=dec_b, torus_a=torus_a, torus_b=torus_b,
             torus_axis=torus_axis, tc_kept=tc_kept, tc_axis=tc_axis, r2_kept=r2[mask],
             rep_dirs=np.asarray(rep_dirs), n_kept=n_kept,
             n_train_trials=len(train_trials), n_test_trials=len(test_trials),
             n_train_per_dir=n_train_per_dir, **rep)
    print("decode A (dynamics ON) =", round(float(dec_a), 4))
    print("decode B (no dynamics) =", round(float(dec_b), 4))
    print("PCA-3 feature baseline = 0.234 (from diag_pca_decode)")
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-per-dir", type=int, default=40)
    ap.add_argument("--latent-dim", type=int, default=3)
    args = ap.parse_args()
    main(n_train_per_dir=args.n_train_per_dir, latent_dim=args.latent_dim)
