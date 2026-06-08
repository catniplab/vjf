"""Self-contained, single-core per-bin timing across the five SNR conditions.

Uses only vjf.synthetic + vjf.realtime (no neurofisherSNR, no experiment.py), so it
runs on a clean VM with just numpy+torch. torch threads pinned to 1 so "single CPU
core" is literally true and thread-pool jitter is removed. Emits the same JSON shape
plot_results.fig_timing reads (conditions[].timing_sample_ms / snr_target / n_neurons).

Run: uv run --with numpy --with torch python experiments/lc_poisson_stream/timing_probe.py
"""
import json
import time

import numpy as np
import torch

from vjf import synthetic as syn
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter

SEED = 20260605
T_EFF = 54000
WARMUP = 1500
REFRESH_K = 1000
N_SNR = {15: -3, 30: 0, 50: 3, 150: 6, 250: 8}   # population size -> nominal SNR (dB)
SAMPLE = 8000                                       # downsampled points for the violin


def measure(n):
    z = syn.limit_cycle(T_EFF, dt=5e-3, angular_velocity=30.0, seed=SEED)
    C, b = syn.poisson_readout(z, n, mean_rate=0.1, peak_rate=0.5, seed=SEED)
    counts = np.array(list(syn.stream(z, C, b, seed=SEED + 1)))
    init_w = min(10 * n, T_EFF // 2)
    torch.manual_seed(SEED)
    model = VJF.make_model(n, 2, 0, 100, hidden_sizes=[100, 100],
                           likelihood="poisson", transition_flow="srrls", encoder="projection")
    ro = OnlineReadout(n, 2, smooth_tau=8.0, refresh_K=REFRESH_K, link="log")
    Cp, bp = ro.warm_start(counts[:init_w])
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cp))
        model.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1)))
    model.decoder.requires_grad_(False)
    live = counts[init_w:]
    per_ms = np.zeros(len(live))
    for r in online_filter(model, live, readout=ro, warmup_steps=WARMUP, rbf_width_scale=0.5):
        per_ms[r.step] = r.elapsed_s * 1e3
    s = per_ms[WARMUP:]                              # steady state only
    idx = np.linspace(0, len(s) - 1, min(SAMPLE, len(s))).astype(int)
    return {
        "n_neurons": n, "snr_target": N_SNR[n], "n_bins": int(len(s)),
        "per_bin_p50_ms": float(np.percentile(s, 50)),
        "per_bin_p95_ms": float(np.percentile(s, 95)),
        "per_bin_p99_ms": float(np.percentile(s, 99)),
        "per_bin_max_ms": float(s.max()),
        "timing_sample_ms": s[idx].tolist(),
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(f"single-core timing probe; torch threads={torch.get_num_threads()}, T_EFF={T_EFF}")
    conds = []
    for n in N_SNR:
        t0 = time.perf_counter()
        c = measure(n)
        conds.append(c)
        print(f"n={n:>3} ({c['snr_target']:>2} dB): p50={c['per_bin_p50_ms']:.2f} "
              f"p95={c['per_bin_p95_ms']:.2f} p99={c['per_bin_p99_ms']:.2f} "
              f"max={c['per_bin_max_ms']:.2f} ms  [{time.perf_counter()-t0:.0f}s]")
    out = {"conditions": conds, "torch_num_threads": 1, "machine": "e2-standard-4 (single core)"}
    with open("x_timing.json", "w") as f:
        json.dump(out, f)
    print("wrote x_timing.json")
