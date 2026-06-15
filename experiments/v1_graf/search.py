"""Phase-1 hyperparameter search for sVJF on Graf V1.

Selection metric (LOCKED, see PLAN_CLEAN.md): future forecasted reconstruction --
the weighted Poisson-deviance skill of a free-run, decoded forecast against the
held-out FUTURE spikes (k = 8/16/32 bins = half/one/two grating cycles), vs a
persistence baseline, **gated** by a decent reconstruction (leave-one-neuron PLL
within a margin of the PSTH ceiling). Decode accuracy is recorded but does NOT drive
selection. Selection is on a VALIDATION split; the test trials are reserved for the
clean experiments.

This driver is *sharded* for parallel `/gcp_run`: each VM runs one shard
(``configs[shard::n_shards]``) and writes ``results/search_<space>_shard<i>.json``
incrementally (so a killed VM keeps its finished configs). A ``--merge`` pass reads
all shard files, applies the reconstruction gate, ranks by the weighted skill, and
writes ``results/best_<space>.json`` + a human-readable ``search_<space>_summary.md``.

Usage (one VM per shard):
    python -m experiments.v1_graf.search --space single --shard 0 --n-shards 8
    ...
    python -m experiments.v1_graf.search --space single --shard 7 --n-shards 8
Then locally:
    python -m experiments.v1_graf.search --space single --merge

Shard/space may also come from env (SEARCH_SPACE / SEARCH_SHARD / SEARCH_NSHARDS),
so a fixed `/gcp_run` entrypoint can select its shard without CLI args.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Single direction for the deep-detail section (strongest-response direction in array_5).
SINGLE_DIR_DEG = 225.0
# Reconstruction gate: a config is eligible only if its val PLL clears this fraction of
# the PSTH-ceiling PLL ("if reconstruction is bad, everything is bad"). Stored, not baked
# in -- applied at merge so the threshold can be retuned without re-running.
GATE_FRAC = 0.6

# dyn_noise as (sigma0, decay, fit_ref) triples (not a full factorial -- a few sensible
# regimes: off, decaying, decaying+fit-gated).
DYN_NOISE = [(0.0, 1.0, 0.0), (0.2, 0.97, 0.0), (0.3, 0.97, 0.3)]


def _grid(axes: dict) -> list:
    """Cartesian product of a dict of name -> list-of-values, as a list of dicts,
    in a deterministic order (so shard i is reproducible)."""
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]


def single_grid() -> list:
    """Single-direction (deep-detail) search grid. Fixed: scale-fixed CCIPCA, Adam flow,
    growing basis + RAN-residual init, randomized replay order."""
    cfgs = []
    for g in _grid({"latent_dim": [3, 4], "rbf_base": [50, 100],
                    "epochs": [20, 50, 100], "lr": [3e-4, 1e-3]}):
        for sig0, decay, fit_ref in DYN_NOISE:
            cfgs.append({**g, "dyn_noise": sig0, "dyn_noise_decay": decay,
                         "dyn_noise_fit_ref": fit_ref})
    return cfgs


def single_grid_xl() -> list:
    """Extended capacity-push single-dir grid (run after the base search). Higher latent
    dim and a larger growing RBF basis (explicit ``max_rbf`` cap so it stays tractable --
    the latent trajectory is a low-D cycle, so center count saturates), plus longer
    training, all in the winning lr/denoising region from the base search. Tests whether
    more capacity closes the PSTH-forecast gap or confirms a ceiling."""
    cfgs = []
    for g in _grid({"latent_dim": [4, 5, 6], "max_rbf": [1600, 2400],
                    "epochs": [100, 200], "lr": [3e-4, 1e-3]}):
        for sig0, decay, fit_ref in [(0.3, 0.97, 0.3), (0.2, 0.97, 0.0)]:
            cfgs.append({**g, "rbf_base": 100, "dyn_noise": sig0,
                         "dyn_noise_decay": decay, "dyn_noise_fit_ref": fit_ref})
    return cfgs


def single_grid_reg() -> list:
    """Regularized search: the R2 curvature penalty (lambda_smooth) + backed-off denoising,
    in the xl winner capacity region (L in {4,5}, 1600 centers, E=100, lr=1e-3). Tests whether
    penalizing field curvature turns the jagged shrinkage-flow into a smooth limit cycle. The
    lambda sweep brackets a wide log range; dyn_noise is off or gentle (R2 replaces its
    smoothing role without the contraction)."""
    # lambda range from the local bracket (lambda_bracket.py, E=30): the sweet spot is
    # ~1e-3..1e-2 (curvature 112->1.4, jaggedness 0.098->0.012, skill peaks); 1e-1
    # over-regularizes (field too flat -> persistence) and lambda>=1 destabilizes the SGD
    # (curvature explodes). So sweep the usable range, refined near the peak; drop 1,10.
    cfgs = []
    for L in [4, 5]:
        for lam in [0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
            for dn in [0.0, 0.1]:
                cfgs.append({"latent_dim": L, "max_rbf": 1600, "epochs": 100, "lr": 1e-3,
                             "rbf_base": 100, "dyn_noise": dn, "dyn_noise_decay": 1.0,
                             "dyn_noise_fit_ref": 0.0, "smooth_lambda": lam})
    return cfgs


def multi_grid() -> list:
    """Multi-direction search grid (run at a fixed n_dir; finalists confirmed at the full
    direction set in Phase 2). Pruned after the single-dir search localizes lr/noise."""
    cfgs = []
    for g in _grid({"latent_dim": [3, 4, 6], "rbf_base": [50, 100, 200],
                    "epochs": [1, 2, 4], "lr": [3e-4, 1e-3]}):
        for sig0, decay, fit_ref in DYN_NOISE:
            cfgs.append({**g, "dyn_noise": sig0, "dyn_noise_decay": decay,
                         "dyn_noise_fit_ref": fit_ref})
    return cfgs


def _run_single(cfg: dict, data: dict, quick: bool) -> dict:
    """Train + evaluate one single-direction config on the VALIDATION split."""
    from experiments.v1_graf.single_dir import _train_eval
    epochs = 2 if quick else cfg["epochs"]                      # quick = smoke only
    res = _train_eval(
        data["train_trials"], data["val_trials"], data["val_counts"], data["val_ybar"],
        epochs=epochs, latent_dim=cfg["latent_dim"], N=data["N"],
        grow=True, grow_weight_init="residual", flow="sgd", optimizer="adam",
        lr=cfg["lr"], rbf_base=cfg.get("rbf_base", 50), max_rbf=cfg.get("max_rbf"),
        dyn_noise=cfg["dyn_noise"], dyn_noise_decay=cfg["dyn_noise_decay"],
        dyn_noise_fit_ref=cfg["dyn_noise_fit_ref"], smooth_lambda=cfg.get("smooth_lambda", 0.0),
        psth_counts=data["psth_counts"], return_model=False)
    return {"pll": res["pll"], "pll_psth_ceiling": res["pll_psth_ceiling"],
            "weighted_persist_skill": res["fc_weighted_persist_skill"],
            "skill": res["forecast_skill"]["skill"], "forecast_r2_diag": res["forecast_r2"],
            "n_basis": res["n_basis"]}


def _run_multi(cfg: dict, n_dir: int, quick: bool) -> dict:
    """Train + evaluate one multi-direction config on the VALIDATION split."""
    from experiments.v1_graf.run_m1 import main
    out = main(latent_dim=cfg["latent_dim"], n_dir=n_dir, flow="sgd", grow=True,
               grow_weight_init="residual", optimizer="adam", lr=cfg["lr"],
               rbf_base=cfg["rbf_base"], epochs=cfg["epochs"], dyn_noise=cfg["dyn_noise"],
               dyn_noise_decay=cfg["dyn_noise_decay"], dyn_noise_fit_ref=cfg["dyn_noise_fit_ref"],
               smooth_lambda=cfg.get("smooth_lambda", 0.0),
               n_val=10, eval_split="val", quick=quick)
    return {"pll": out["pll_bits_per_spike"], "pll_psth_ceiling": out["pll_psth_ceiling"],
            "weighted_persist_skill": out["fc_weighted_persist_skill"],
            "skill": out["forecast_skill"]["skill"], "forecast_r2_diag": out["forecast_r2"],
            "decode_acc": out["decode_acc"], "n_basis": out["provenance"]["config"]["n_basis_final"],
            "n_dir": n_dir}


def _flatten_skill(rec: dict) -> None:
    """Promote the nested skill dict to flat scalar fields (skill8_persist, ...) so
    best_<space>.json stays usable after a JSON round-trip turns int keys into strings."""
    sk = rec.get("skill")
    if not isinstance(sk, dict):
        return
    for k in (8, 16, 32):
        d = sk.get(k, sk.get(str(k), {}))
        rec[f"skill{k}_persist"] = d.get("vs_persist", float("nan"))
        rec[f"skill{k}_psth"] = d.get("vs_psth", float("nan"))


def _dump_atomic(path: str, obj) -> None:
    """Write JSON to a temp file then os.replace -- a killed VM can never leave a
    half-written (un-parseable) shard file behind."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def run_shard(space: str, shard: int, n_shards: int, quick: bool, n_dir: int,
              grid: str = "base") -> str:
    """Run this shard's slice of the grid; dump results atomically + incrementally."""
    if not (n_shards > 0 and 0 <= shard < n_shards):
        raise SystemExit(f"invalid shard params: shard={shard}, n_shards={n_shards} "
                         f"(need n_shards>0 and 0<=shard<n_shards)")
    if space == "single":
        cfgs = {"xl": single_grid_xl, "reg": single_grid_reg, "base": single_grid}[grid]()
    else:
        cfgs = multi_grid()
    mine = cfgs[shard::n_shards]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"search_{space}_shard{shard}.json")
    meta = {"space": space, "shard": shard, "n_shards": n_shards, "grid": grid,
            "grid_size": len(cfgs), "n_assigned": len(mine),
            "n_dir": n_dir if space == "multi" else None, "quick": quick}
    print(f"[search] space={space} shard={shard}/{n_shards}: {len(mine)}/{len(cfgs)} configs"
          f"{' (QUICK)' if quick else ''}", flush=True)

    data = None
    if space == "single":
        from experiments.v1_graf.single_dir import prepare_single_dir_data
        data = prepare_single_dir_data(direction=SINGLE_DIR_DEG, n_val=10)
        data["val_ybar"] = float(data["val_counts"].mean())
        print(f"[search] single dir={data['d_star']:.0f} deg, N={data['N']}, "
              f"PSTH ceiling PLL={data['pll_psth']:.3f}, "
              f"{len(data['train_trials'])} train / {len(data['val_trials'])} val", flush=True)

    records = []
    for j, cfg in enumerate(mine):
        t0 = time.time()
        rec = {"config": cfg, "space": space}
        try:
            metrics = _run_single(cfg, data, quick) if space == "single" \
                else _run_multi(cfg, n_dir, quick)
            rec.update(metrics)
            _flatten_skill(rec)
        except Exception as e:                                  # one bad config must not kill the shard
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
        rec["elapsed_s"] = time.time() - t0
        records.append(rec)
        if "error" in rec:
            print(f"[search] {j + 1}/{len(mine)} {cfg}  ->  ERROR {rec['error']}", flush=True)
        else:
            print(f"[search] {j + 1}/{len(mine)} {cfg}  ->  S={rec['weighted_persist_skill']:+.4f} "
                  f"PLL={rec.get('pll', float('nan')):.3f}/"
                  f"{rec.get('pll_psth_ceiling', float('nan')):.3f}  "
                  f"({rec['elapsed_s']:.0f}s)", flush=True)
        _dump_atomic(out_path, {"meta": meta, "records": records})   # survive a killed VM
    print(f"[search] shard done -> {out_path}", flush=True)
    return out_path


def _finite(x) -> bool:
    try:
        return np.isfinite(x)
    except TypeError:
        return False


def merge(space: str) -> str:
    """Read all shard files for `space`, validate coverage, apply the reconstruction gate,
    rank by the weighted forecasted-reconstruction skill, and write best_<space>.json + a
    summary. A diverged config (non-finite S) is never eligible; if NOTHING passes the gate
    `best` is null (no silent promotion of a reconstruction-failing config)."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"search_{space}_shard*.json")))
    if not files:
        raise SystemExit(f"no shard files matching search_{space}_shard*.json in {RESULTS_DIR}")
    recs, metas, bad_files = [], [], []
    for f in files:
        try:
            with open(f) as fh:
                blob = json.load(fh)
        except json.JSONDecodeError as e:                       # partial/corrupt shard
            bad_files.append((os.path.basename(f), str(e)))
            continue
        if isinstance(blob, dict):                              # new format {meta, records}
            metas.append(blob.get("meta", {}))
            recs += blob.get("records", [])
        else:                                                   # legacy bare-list format
            recs += blob
    # Coverage check: shards should agree on n_shards/grid_size and cover all shard IDs.
    coverage_warn = []
    n_shards_set = {m.get("n_shards") for m in metas if m.get("n_shards")}
    if len(n_shards_set) > 1:
        coverage_warn.append(f"inconsistent n_shards across files: {sorted(n_shards_set)}")
    elif n_shards_set:
        n_sh = n_shards_set.pop()
        present = {m.get("shard") for m in metas}
        missing = sorted(set(range(n_sh)) - present)
        if missing:
            coverage_warn.append(f"missing shards {missing} of {n_sh} -- merge is PARTIAL")

    ok = [r for r in recs if "error" not in r]
    failed = [r for r in recs if "error" in r]
    wts = {8: 0.5, 16: 0.3, 32: 0.2}                            # same weights as the persistence S
    for r in ok:
        ceil = r.get("pll_psth_ceiling", 0.0)
        S = r.get("weighted_persist_skill", float("nan"))
        r["gate_pass"] = bool(ceil > 0 and r.get("pll", -1) >= GATE_FRAC * ceil and _finite(S))
        # Reference (NOT the selection key): weighted skill vs the near-oracle PSTH, from the
        # per-k psth skills already stored in each record.
        pk = [r.get(f"skill{k}_psth") for k in (8, 16, 32)]
        r["weighted_psth_skill"] = (sum(w * v for w, v in zip(wts.values(), pk))
                                    if all(_finite(v) for v in pk) else float("nan"))
    # Rank only finite-S configs (NaN = diverged/empty -> not comparable, pushed out).
    key = lambda r: r.get("weighted_persist_skill", float("nan"))
    eligible = sorted((r for r in ok if r["gate_pass"]), key=key, reverse=True)
    ranked_all = sorted((r for r in ok if _finite(key(r))), key=key, reverse=True)
    best = eligible[0] if eligible else None                    # no gate bypass
    top_ungated = ranked_all[0] if ranked_all else None         # for inspection only

    best_path = os.path.join(RESULTS_DIR, f"best_{space}.json")
    with open(best_path, "w") as fh:
        json.dump({"best": best, "top_ungated": top_ungated, "n_configs": len(recs),
                   "n_eligible": len(eligible), "n_failed": len(failed),
                   "gate_frac": GATE_FRAC, "coverage_warnings": coverage_warn,
                   "bad_files": bad_files, "shard_metas": metas}, fh, indent=2)

    lines = [f"# Search summary: {space} ({len(recs)} configs, {len(eligible)} pass gate, "
             f"{len(failed)} failed)\n",
             f"Gate: val PLL >= {GATE_FRAC} x PSTH-ceiling PLL (and finite S). **Ranked by the "
             f"weighted persistence-skill S_persist** (the achievable forecast target). "
             f"**S_psth** = weighted skill vs the stimulus-locked PSTH, a phase-aware near-oracle "
             f"baseline -- shown for REFERENCE, it does not drive selection. Per-k skills are over "
             f"the first k forecast bins (k=8/16/32 = half/one/two grating cycles).\n"]
    for w in coverage_warn:
        lines.append(f"> WARNING: {w}\n")
    for fn, err in bad_files:
        lines.append(f"> WARNING: unreadable shard {fn}: {err}\n")
    lines += ["| rank | S_persist | S_psth(ref) | persist k8/k16/k32 | psth k8/k16/k32 | PLL/ceil | gate | L | E | lr | dyn_noise | n_basis |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked_all[:25]):
        c = r["config"]
        dn = f"{c['dyn_noise']}/{c['dyn_noise_decay']}/{c['dyn_noise_fit_ref']}"
        sp = [r.get(f"skill{k}_persist", float("nan")) for k in (8, 16, 32)]
        sq = [r.get(f"skill{k}_psth", float("nan")) for k in (8, 16, 32)]
        lines.append(
            f"| {i + 1} | {r['weighted_persist_skill']:+.4f} | {r.get('weighted_psth_skill', float('nan')):+.4f} | "
            f"{sp[0]:+.3f}/{sp[1]:+.3f}/{sp[2]:+.3f} | {sq[0]:+.3f}/{sq[1]:+.3f}/{sq[2]:+.3f} | "
            f"{r.get('pll', float('nan')):.3f}/{r.get('pll_psth_ceiling', float('nan')):.3f} | "
            f"{'Y' if r['gate_pass'] else 'n'} | {c['latent_dim']} | "
            f"{c['epochs']} | {c['lr']:g} | {dn} | {r.get('n_basis', '?')} |")
    if failed:
        lines.append(f"\n{len(failed)} failed configs:")
        for r in failed:
            lines.append(f"- {r['config']}: {r['error']}")
    summary_path = os.path.join(RESULTS_DIR, f"search_{space}_summary.md")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[merge] {len(recs)} configs, {len(eligible)} eligible, {len(failed)} failed -> {best_path}")
    for w in coverage_warn:
        print(f"[merge] WARNING: {w}")
    print(f"[merge] summary -> {summary_path}")
    if best is not None:
        print(f"[merge] BEST (gate pass): S={best['weighted_persist_skill']:+.4f} {best['config']}")
    elif top_ungated is not None:
        print(f"[merge] NO config passed the gate. top ungated (NOT selected): "
              f"S={top_ungated['weighted_persist_skill']:+.4f} {top_ungated['config']}")
    return best_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="sVJF Phase-1 hyperparameter search (sharded)")
    ap.add_argument("--space", type=str, default=os.environ.get("SEARCH_SPACE", "single"),
                    choices=["single", "multi"])
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SEARCH_SHARD", 0)))
    ap.add_argument("--n-shards", type=int, default=int(os.environ.get("SEARCH_NSHARDS", 1)))
    ap.add_argument("--n-dir", type=int, default=int(os.environ.get("SEARCH_NDIR", 8)),
                    help="multi-dir: directions used for the search grid")
    ap.add_argument("--grid", type=str, default=os.environ.get("SEARCH_GRID", "base"),
                    choices=["base", "xl", "reg"],
                    help="single-dir grid: base, xl capacity-push, or reg (curvature-regularized)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--results-dir", type=str, default=None,
                    help="override the results dir (for merging shard files pulled off VMs)")
    args = ap.parse_args()

    if args.results_dir:                                        # functions read RESULTS_DIR at call time
        RESULTS_DIR = args.results_dir
    if args.merge:
        merge(args.space)
    else:
        run_shard(args.space, args.shard, args.n_shards, args.quick, args.n_dir, args.grid)
