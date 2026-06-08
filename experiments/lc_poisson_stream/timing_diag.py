"""Why is n=50 (3 dB) the only condition with an elevated latency tail, and only on
long streams? Test the leading hypothesis: occasional filter near-divergences at the
marginal SNR trigger the (slower) recovery path. Log per-bin time + r.diverged, then
report whether slow bins ARE the diverged bins and WHERE in the stream they fall.

Run: uv run --with numpy --with torch python experiments/lc_poisson_stream/timing_diag.py
"""
import numpy as np
import torch

from vjf import synthetic as syn
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter

SEED = 20260605
WARMUP = 1500
REFRESH_K = 1000


def run(n, threads, T_EFF, flush_denormal=False):
    torch.set_num_threads(threads)
    torch.set_flush_denormal(flush_denormal)
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
    per_ms = np.zeros(len(live)); div = np.zeros(len(live), bool)
    for r in online_filter(model, live, readout=ro, warmup_steps=WARMUP, rbf_width_scale=0.5):
        per_ms[r.step] = r.elapsed_s * 1e3
        div[r.step] = bool(r.diverged)
    s = per_ms[WARMUP:]; d = div[WARMUP:]
    med = np.median(s); slow = s > 2 * med
    half = len(s) // 2
    refresh = np.zeros(len(s), bool)
    rk = np.arange(len(s)) + WARMUP + init_w           # absolute stream index
    refresh[(rk % REFRESH_K) == 0] = True
    print(f"\n[n={n}, threads={threads}, flush_denormal={flush_denormal}, steady={len(s)}]  median={med:.2f} p95={np.percentile(s,95):.2f} "
          f"p99={np.percentile(s,99):.2f} max={s.max():.2f}")
    print(f"   diverged bins: {d.sum()} ({100*d.mean():.2f}%);  slow bins (>2x med): {slow.sum()} ({100*slow.mean():.2f}%)")
    print(f"   slow & diverged: {(slow & d).sum()};  slow & refresh-bin: {(slow & refresh).sum()};  "
          f"slow & neither: {(slow & ~d & ~refresh).sum()}")
    print(f"   slow bins in 1st half: {slow[:half].sum()}  vs 2nd half: {slow[half:].sum()}")
    if slow.sum():
        first = np.where(slow)[0][0]
        print(f"   first slow bin at steady-index {first} ({100*first/len(s):.0f}% through stream); "
              f"median slow-bin time={np.median(s[slow]):.2f} ms")


if __name__ == "__main__":
    # threads pinned to 1 throughout; toggle flush-denormal to test the subnormal-float hypothesis
    run(50, 1, 54000, flush_denormal=False)   # the anomaly
    run(50, 1, 54000, flush_denormal=True)    # same, denormals flushed -> does the tail vanish?
    run(30, 1, 54000, flush_denormal=False)   # control (clean condition)
