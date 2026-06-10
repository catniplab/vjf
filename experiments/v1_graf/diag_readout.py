"""Localize the M1 failure inside the projection-encoder pipeline.

Both sVJF variants (dynamics on/off) decode orientation at chance, while a batch
PCA-3 of the feature gets 0.23. So the encoder pipeline loses orientation. This
splits it: decode straight from the readout's projection pi = C^+ (g~(y)-b) -- the
RECOGNITION INPUT -- for (a) the warm-start C (batch PCA on the 72-trial coverage
window) and (b) the online CCIPCA C after the full train stream. If pi decodes
well, the recognition net is the culprit; if pi decodes at chance, the online
readout C is.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from experiments.v1_graf.graf_loader import (
    STIM_MS, bin_spikes, load_array, well_tuned_mask)
from experiments.v1_graf.run_m1 import _split_trials
from vjf.readout import OnlineReadout

SEED = 20260609


def _decode(X, y):
    return float(cross_val_score(LogisticRegression(max_iter=3000), X,
                                 np.round(y).astype(int), cv=5).mean())


def _proj_summary(ro, trials, nstim):
    """Per-trial mean projection pi over the stimulus window, EMA reset per trial."""
    nu0 = ro.nu.copy()
    out = []
    for tc in trials:
        ro.nu = nu0.copy()
        ps = []
        for t in range(min(nstim, tc.shape[0])):
            g = ro.feature(tc[t], update_mean=False)
            ps.append(ro.project(g))
        out.append(np.mean(ps, 0))
    ro.nu = nu0
    return np.asarray(out)


def main():
    rng = np.random.default_rng(SEED)
    arr = load_array(5)
    counts = bin_spikes(arr["spk_times"], bin_ms=10.0)
    dirs_all = arr["ori"]
    nstim = int(round(STIM_MS / 10.0))
    mask, _ = well_tuned_mask(counts, dirs_all)
    counts = counts[:, :, mask]
    n_kept = counts.shape[2]

    keep = np.unique(dirs_all)
    train_by_dir, test_by_dir = _split_trials(dirs_all, 10, 40, rng)
    cov_idx = [train_by_dir[d][0] for d in keep]
    rest_idx = [i for d in keep for i in train_by_dir[d][1:]]
    test_idx = [i for d in keep for i in test_by_dir[d]]
    test_dirs = dirs_all[test_idx]
    cover = np.concatenate([counts[i] for i in cov_idx], 0)

    ro = OnlineReadout(n_kept, 3, smooth_tau=8.0, refresh_K=1000, link="log")
    ro.warm_start(cover)                                    # warm-start C (PCA on coverage)
    pi_ws = _proj_summary(ro, [counts[i] for i in test_idx], nstim)
    acc_ws = _decode(pi_ws, test_dirs)

    # stream the full train set through the online CCIPCA update (no refresh write needed:
    # rebuild C from the converged vectors), then re-project the test set.
    for i in cov_idx + rest_idx:
        tc = counts[i]
        ro.nu = np.zeros(n_kept)
        for t in range(tc.shape[0]):
            g = ro.feature(tc[t], update_mean=True)
            ro.update(g)
    ro.C = ro._scaled_C()
    ro.C_pinv = np.linalg.pinv(ro.C).astype(np.float32)
    pi_on = _proj_summary(ro, [counts[i] for i in test_idx], nstim)
    acc_on = _decode(pi_on, test_dirs)

    # reference: principal angle between the online C subspace and a batch PCA-3 of the
    # per-trial-mean features over ALL trials (the 0.234 subspace).
    feat_all = np.log(counts[:, :nstim, :].mean(1) + 1e-2)
    fc = feat_all - feat_all.mean(0)
    _, _, vt = np.linalg.svd(fc, full_matrices=False)
    pca3 = vt[:3]
    qa = np.linalg.qr(ro.C)[0]
    s = np.linalg.svd(qa.T @ pca3.T, compute_uv=False)
    max_angle_deg = float(np.degrees(np.arccos(np.clip(s.min(), -1, 1))))

    print(f"decode from readout projection pi (warm-start C)  = {acc_ws:.3f}")
    print(f"decode from readout projection pi (online CCIPCA C)= {acc_on:.3f}")
    print(f"PCA-3 batch feature baseline                       = 0.234")
    print(f"max principal angle, online-C vs batch-PCA-3       = {max_angle_deg:.1f} deg")
    print(f"(chance = {1/72:.4f})")


if __name__ == "__main__":
    main()
