"""Is the per-bin latency tail caused by Python's cyclic garbage collector?

Mirror the online setup, time every bin, and log when gc fires (gc.callbacks).
Then re-run with gc.disable() and compare the tail. If disabling gc removes the
slow bins -> it's GC; if not -> it's allocator/scheduler/VM contention.

Run: uv run --with numpy --with torch python experiments/lc_poisson_stream/gc_probe.py
"""
import gc
import time

import numpy as np
import torch

from vjf import synthetic as syn
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter

SEED = 20260605
T_EFF = 22000
WARMUP = 1500
REFRESH_K = 1000


def build(N):
    z = syn.limit_cycle(T_EFF, dt=5e-3, angular_velocity=30.0, seed=SEED)
    C, b = syn.poisson_readout(z, N, mean_rate=0.1, peak_rate=0.5, seed=SEED)
    counts = np.array(list(syn.stream(z, C, b, seed=SEED + 1)))
    init_w = min(10 * N, T_EFF // 2)
    torch.manual_seed(SEED)
    model = VJF.make_model(N, 2, 0, 100, hidden_sizes=[100, 100],
                           likelihood="poisson", transition_flow="srrls", encoder="projection")
    ro = OnlineReadout(N, 2, smooth_tau=8.0, refresh_K=REFRESH_K, link="log")
    Cp, bp = ro.warm_start(counts[:init_w])
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cp))
        model.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1)))
    model.decoder.requires_grad_(False)
    return model, ro, counts[init_w:]


def timed_pass(N, gc_on):
    model, ro, live = build(N)
    bin_idx = [0]
    gc_bins = []                       # bins during which a gc collection finished
    def cb(phase, info):
        if phase == "stop":
            gc_bins.append((bin_idx[0], info.get("generation"), info.get("collected")))
    gc.collect()
    if gc_on:
        gc.enable(); gc.callbacks.append(cb)
    else:
        gc.disable()
    per_ms = np.zeros(len(live))
    for r in online_filter(model, live, readout=ro, warmup_steps=WARMUP, rbf_width_scale=0.5):
        bin_idx[0] = r.step
        t = time.perf_counter()
        # the loop already filtered; r.elapsed_s is the loop's own measurement
        per_ms[r.step] = r.elapsed_s * 1e3
    if gc_on and cb in gc.callbacks:
        gc.callbacks.remove(cb)
    gc.enable()
    s = per_ms[WARMUP:]                 # steady state only
    return s, gc_bins


def summarize(tag, s, gc_bins):
    med = np.median(s)
    slow = s > 2 * med
    idx = set(np.where(slow)[0].tolist())
    gc_at = set(b for b, _, _ in gc_bins if b >= WARMUP)
    # a slow bin "explained" by gc if a collection finished within +/-1 bin
    expl = sum(any((i + d) in {b - WARMUP for b in gc_at} for d in (-1, 0, 1)) for i in idx)
    gens = {}
    for _, g, _ in gc_bins:
        gens[g] = gens.get(g, 0) + 1
    print(f"\n[{tag}]  N={len(s)}  median={med:.2f}ms  p95={np.percentile(s,95):.2f}  "
          f"p99={np.percentile(s,99):.2f}  max={s.max():.2f}")
    print(f"   bins >2x median: {slow.sum()} ({100*slow.mean():.2f}%);  bins >5ms: {(s>5).sum()}")
    print(f"   gc collections in steady state: {sum(1 for b,_,_ in gc_bins if b>=WARMUP)} "
          f"(by generation {gens});  slow bins coinciding with a gc stop (+/-1): {expl}/{len(idx)}")


if __name__ == "__main__":
    # match experiment.py: torch default thread count (all cores), not forced to 1
    print(f"torch threads={torch.get_num_threads()}, T_EFF={T_EFF}, refresh_K={REFRESH_K}")
    for N in (50, 250):                 # n=50 was the anomalous timing condition; n=250 a control
        print(f"\n================ n={N} (steady bins={T_EFF - min(10*N,T_EFF//2) - WARMUP}) ================")
        s_on, gcb = timed_pass(N, gc_on=True)
        summarize("gc ENABLED", s_on, gcb)
        s_off, _ = timed_pass(N, gc_on=False)
        summarize("gc DISABLED", s_off, [])
        print(f"verdict n={N}: p99 {np.percentile(s_on,99):.2f} -> {np.percentile(s_off,99):.2f} ms, "
              f"max {s_on.max():.2f} -> {s_off.max():.2f} ms with gc disabled")
