"""E6 (expanded): does ORIGINAL VJF converge online? Four original variants --
C initialization {random, oracle} x optimizer {Adam, SGD} -- plus sVJF, on the same
streams, 5 seeds. Original VJF = raw-spike recognition, readout C and flow W learned by
the ELBO gradient (flow_learner='sgd'), no readout warm-start. Logs filtered-latent R^2
AND one-step prediction R^2 vs stream position. Self-contained (vjf.synthetic; no neurofisherSNR).

Run: uv run --with numpy --with torch python experiments/lc_poisson_stream/motivation_compare.py
"""
import json

import numpy as np
import torch

from vjf import synthetic as syn
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter

SEEDS = [20260605, 20260606, 20260607, 20260608, 20260609]   # 5 seeds
N, SNR = 50, 3
T_EFF, EVERY, WIN = 60000, 1000, 4000
VARIANTS = ["orig_randC_adam", "orig_randC_sgd", "orig_oracleC_adam", "orig_oracleC_sgd", "svjf"]


def affine_r2(mu, z):
    A = np.concatenate([mu, np.ones((len(mu), 1))], 1)
    W, *_ = np.linalg.lstsq(A, z, rcond=None)
    sse = ((z - A @ W) ** 2).sum(); tss = ((z - z.mean(0)) ** 2).sum() + 1e-12
    return float(1 - sse / tss)


def onestep_r2(mu, z, model):
    with torch.no_grad():
        o = model.transition(torch.as_tensor(mu), None, sampling=False)
        nxt = (o.mean if isinstance(o, tuple) else o).cpu().numpy()
    A = np.concatenate([mu[:-1], np.ones((len(mu) - 1, 1))], 1)
    W, *_ = np.linalg.lstsq(A, z[1:], rcond=None)
    pred = np.concatenate([nxt[:-1], np.ones((len(nxt) - 1, 1))], 1) @ W
    sse = ((z[1:] - pred) ** 2).sum(); tss = ((z[1:] - z[1:].mean(0)) ** 2).sum() + 1e-12
    return float(1 - sse / tss)


def run(variant, seed):
    z = syn.limit_cycle(T_EFF, dt=5e-3, angular_velocity=30.0, seed=seed)
    C, b = syn.poisson_readout(z, N, mean_rate=0.1, peak_rate=0.5, seed=seed)
    counts = np.array(list(syn.stream(z, C, b, seed=seed + 1)))
    torch.manual_seed(seed)
    if variant == "svjf":
        model = VJF.make_model(N, 2, 0, 100, hidden_sizes=[100, 100], likelihood="poisson",
                               transition_flow="srrls", encoder="projection")
        ro = OnlineReadout(N, 2, smooth_tau=8.0, refresh_K=1000, link="log")
        iw = min(10 * N, T_EFF // 2)
        Cp, bp = ro.warm_start(counts[:iw])
        with torch.no_grad():
            model.decoder.decode.weight.copy_(torch.as_tensor(Cp))
            model.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1)))
        model.decoder.requires_grad_(False)
        live, z_live = counts[iw:], z[iw:]
        gen = online_filter(model, live, readout=ro, warmup_steps=1500, rbf_width_scale=0.5)
    else:                                                    # original VJF variants
        init, opt = variant.split("_")[1], variant.split("_")[2]   # randC|oracleC , adam|sgd
        model = VJF.make_model(N, 2, 0, 100, hidden_sizes=[100, 100], likelihood="poisson",
                               transition_flow="sgd", encoder="spikes", optimizer=opt)
        if init == "oracleC":                                # initialize readout at the true C, b
            with torch.no_grad():
                model.decoder.decode.weight.copy_(torch.as_tensor(C.astype(np.float32)))
                model.decoder.decode.bias.copy_(torch.as_tensor(b.reshape(-1).astype(np.float32)))
        model.decoder.requires_grad_(True)                   # C learned by the ELBO gradient
        live, z_live = counts, z
        gen = online_filter(model, live, readout=None, warmup_steps=200, rbf_width_scale=1.0)
    mu = np.zeros((len(live), 2), dtype=np.float32)
    steps, rf, ro_ = [], [], []
    for r in gen:
        mu[r.step] = r.mean
        s = r.step + 1
        if s >= WIN and s % EVERY == 0:
            w = slice(s - WIN, s)
            steps.append(int(s)); rf.append(affine_r2(mu[w], z_live[w]))
            ro_.append(onestep_r2(mu[w], z_live[w], model))
    return steps, rf, ro_


if __name__ == "__main__":
    out = {}
    for v in VARIANTS:
        steps0, filt, one = None, [], []
        for sd in SEEDS:
            st, rf, ro_ = run(v, sd)
            steps0 = st; filt.append(rf); one.append(ro_)
        out[v] = {"step": steps0, "r2_filt": filt, "r2_onestep": one}     # [seed][t]
        fm = np.mean([c[-1] for c in filt]); om = np.mean([c[-1] for c in one])
        print(f"{v:18s} final filtered R2={fm:.2f}  one-step R2={om:.2f}  (5 seeds)")
    with open("motivation_compare.json", "w") as f:
        json.dump(out, f)
    print("wrote motivation_compare.json")
