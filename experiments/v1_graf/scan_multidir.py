"""Multi-direction sVJF scan (GCP overnight): does the fixed stack rescue multi-condition?

The original M1 (all directions, one shared latent/flow, per-trial reset) was a NO-GO from
encoder collapse -- the readout could not separate conditions. This scan compares, across
direction counts and latent dims:

  - srrls_base     : the original srrls flow, now with the CCIPCA scale fix (isolates it)
  - sgd_noise      : full stack, fixed basis  (sgd + dyn_noise=0.2, adam)
  - sgd_grow_noise : full stack + growing basis (residual init)

Key metric: orientation decode accuracy (the encoder-collapse failure mode), plus
leave-one-neuron PLL and free-run forecast R2. Writes results/scan_multidir.json
incrementally; figures are made locally from the JSON.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.v1_graf.run_m1 import main as run_m1_main, RESULTS_DIR, SEED

ARMS = {
    "srrls_base":     dict(flow="srrls", grow=False, dyn_noise=0.0),                  # scale-fix only (decode ref)
    "sgd_grow_noise": dict(flow="sgd", grow=True, grow_weight_init="residual", dyn_noise=0.2,
                           optimizer="adam", lr=1e-4, max_rbf=400),                   # previous full stack
    # refined: more RBFs (bigger cap) + smaller widths (more growth) + harder-annealed,
    # fit-gated noise (suppressed while the flow underfits).
    "sgd_grow_refined": dict(flow="sgd", grow=True, grow_weight_init="residual", optimizer="adam",
                             lr=1e-4, max_rbf=600, width_scale=0.4, dyn_noise=0.3,
                             dyn_noise_decay=0.9, dyn_noise_fit_ref=0.3),
}
DIM_DIRS = [(3, 8), (3, 24), (3, 72)]                # L=3 across direction counts
SEEDS = [SEED, SEED + 1]
OUT_NAME = "scan_multidir_refined.json"


def _run(cfg):
    import torch
    torch.set_num_threads(1)
    out = run_m1_main(array_num=5, bin_ms=10.0, latent_dim=cfg["latent_dim"],
                      n_dir=cfg["n_dir"], seed=cfg["seed"], **ARMS[cfg["arm"]])
    pc = out["provenance"]["config"]
    return {k: cfg[k] for k in ("arm", "latent_dim", "n_dir", "seed")} | {
        "pll": out["pll_bits_per_spike"], "decode_acc": out["decode_acc"],
        "forecast_r2": out["forecast_r2"], "n_diverge": out["n_diverge"],
        "n_basis": pc["n_basis_final"], "n_neurons": out["n_neurons"]}


def main(workers=6, quick=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dim_dirs = [(3, 8)] if quick else DIM_DIRS
    seeds = [SEED] if quick else SEEDS
    arms = ["sgd_grow_refined"] if quick else list(ARMS)
    configs = [dict(arm=a, latent_dim=L, n_dir=nd, seed=s)
               for a, (L, nd), s in itertools.product(arms, dim_dirs, seeds)]
    print(f"scan_multidir: {len(configs)} configs, workers={workers}", flush=True)

    results, out_path = [], os.path.join(RESULTS_DIR, OUT_NAME)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"[{i}/{len(configs)}] {r['arm']:15} L={r['latent_dim']} dir={r['n_dir']:>2} "
                      f"sd={r['seed']}: decode={r['decode_acc']:.3f} PLL={r['pll']:.4f} "
                      f"fc={r['forecast_r2']:+.3f} nb={r['n_basis']} div={r['n_diverge']}", flush=True)
            except Exception as err:
                print(f"[{i}/{len(configs)}] FAILED {c['arm']} L={c['latent_dim']} "
                      f"dir={c['n_dir']}: {err!r}", flush=True)
            with open(out_path, "w") as fh:
                json.dump(results, fh, indent=2)
    print(f"done: {len(results)}/{len(configs)} -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(workers=a.workers, quick=a.quick)
