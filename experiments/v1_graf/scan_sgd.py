"""SGD-flow hyperparameter scan for the single-direction sVJF (run on GCP via /gcp_run).

Grid over optimizer {sgd, adam} x arm {fixed O(2^d) basis, grow + residual weight-init}
x lr x latent_dim x seed. Each config trains to E epochs and records the leave-one-neuron
PLL and the free-run forecast R2 at epoch checkpoints (to expose stability over training,
the failure mode that killed srrls). Writes results/scan_sgd.json incrementally so a
partial run is still usable; figures are made locally afterwards from the JSON.

The data (single direction, 40 train / 10 test) is fixed across configs; only the model
seed varies. Configs run in a process pool, one torch thread each.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from experiments.v1_graf.single_dir import (
    prepare_single_dir_data, _train_eval, RESULTS, SEED)

E = 50
SNAP_EPS = (10, 25, 50)                      # forecast/PLL checkpoints (stability over training)
OPTIMIZERS = ["sgd", "adam"]
ARMS = [("fixed", dict(grow=False)),                                  # full O(2^d) data-driven basis
        ("grow_res", dict(grow=True, grow_weight_init="residual"))]   # seed 8 + RAN-grow to O(2^d)
LRS = [1e-4, 5e-4, 2e-3]
LATENT_DIMS = [2, 3, 4]
SEEDS = [SEED, SEED + 1]

_DATA = None


def _data():
    global _DATA                              # load once per worker process
    if _DATA is None:
        _DATA = prepare_single_dir_data(None)
    return _DATA


def _run(cfg):
    torch.set_num_threads(1)
    d = _data()
    res = _train_eval(d["train_trials"], d["test_trials"], d["test_counts"], d["ybar"],
                      E, cfg["latent_dim"], d["N"], flow="sgd", optimizer=cfg["optimizer"],
                      lr=cfg["lr"], seed=cfg["seed"], snapshot_epochs=SNAP_EPS,
                      **cfg["arm_kw"])
    snaps = [{"epoch": s["epoch"], "pll": s["pll"], "fc": s["fc"], "n_basis": s["n_basis"]}
             for s in res["snapshots"]]
    return {k: cfg[k] for k in ("optimizer", "arm", "lr", "latent_dim", "seed")} | {
        "pll_psth": d["pll_psth"], "final_pll": res["pll"], "final_fc": res["forecast_r2"],
        "n_basis": res["n_basis"], "snapshots": snaps}


def main(workers=None, quick=False):
    os.makedirs(RESULTS, exist_ok=True)
    lrs = [5e-4] if quick else LRS
    lds = [3] if quick else LATENT_DIMS
    seeds = [SEED] if quick else SEEDS
    configs = [dict(optimizer=opt, arm=name, arm_kw=kw, lr=lr, latent_dim=ld, seed=sd)
               for opt, (name, kw), lr, ld, sd
               in itertools.product(OPTIMIZERS, ARMS, lrs, lds, seeds)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"scan_sgd: {len(configs)} configs, E={E}, workers={workers}", flush=True)

    results, out = [], os.path.join(RESULTS, "scan_sgd.json")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                traj = "/".join(f"{s['fc']:+.2f}" for s in r["snapshots"])    # ep10/25/50
                print(f"[{i}/{len(configs)}] {r['optimizer']}/{r['arm']} lr={r['lr']:.0e} "
                      f"L={r['latent_dim']} sd={r['seed']}: PLL={r['final_pll']:.3f} "
                      f"fc={r['final_fc']:+.3f} (ep10/25/50 {traj})", flush=True)
            except Exception as err:                                 # keep going on a bad config
                print(f"[{i}/{len(configs)}] FAILED {c['optimizer']}/{c['arm']} "
                      f"lr={c['lr']:.0e} L={c['latent_dim']}: {err!r}", flush=True)
            with open(out, "w") as fh:
                json.dump(results, fh, indent=2)
    print(f"done: {len(results)}/{len(configs)} -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true")             # tiny subset for a local smoke
    a = ap.parse_args()
    main(workers=a.workers, quick=a.quick)
