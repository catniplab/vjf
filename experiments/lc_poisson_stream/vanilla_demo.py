"""E6: original VJF does not converge in a single real-time low-SNR pass (motivating figure).

Original VJF = raw-spike recognition, readout C learned by the SGD/Adam ELBO gradient, flow by
SGD, NO readout warm-start. We stream it once and log the filtered-latent R^2 and one-step
prediction R^2 vs stream position; both stay low because the readout collapses to the mean rate
under sparse spikes. For contrast we also stream sVJF (projection encoder + decoupled online
readout + warm-start + srrls) on the SAME stream. Self-contained (vjf.synthetic; no neurofisherSNR).

Run: uv run --with numpy --with torch python experiments/lc_poisson_stream/vanilla_demo.py
"""
import json

import numpy as np
import torch

from vjf import synthetic as syn
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter

SEED = 20260605
N, SNR = 50, 3          # representative mid-SNR neural regime
T_EFF = 60000
EVERY = 1000            # log cadence (bins)
WIN = 4000             # trailing window for the R^2 estimates


def affine_r2(mu, z):
    A = np.concatenate([mu, np.ones((len(mu), 1))], 1)
    W, *_ = np.linalg.lstsq(A, z, rcond=None)
    sse = ((z - A @ W) ** 2).sum(); tss = ((z - z.mean(0)) ** 2).sum() + 1e-12
    return 1.0 - sse / tss, W


def onestep_r2(mu, z, model, device):
    with torch.no_grad():
        o = model.transition(torch.as_tensor(mu, device=device), None, sampling=False)
        nxt = (o.mean if isinstance(o, tuple) else o).cpu().numpy()  # Gaussian namedtuple vs Tensor
    A = np.concatenate([mu[:-1], np.ones((len(mu) - 1, 1))], 1)
    W, *_ = np.linalg.lstsq(A, z[1:], rcond=None)         # align mu->z on the window
    pred = np.concatenate([nxt[:-1], np.ones((len(nxt) - 1, 1))], 1) @ W
    sse = ((z[1:] - pred) ** 2).sum(); tss = ((z[1:] - z[1:].mean(0)) ** 2).sum() + 1e-12
    return 1.0 - sse / tss


def stream_log(kind):
    device = "cpu"
    z = syn.limit_cycle(T_EFF, dt=5e-3, angular_velocity=30.0, seed=SEED)
    C, b = syn.poisson_readout(z, N, mean_rate=0.1, peak_rate=0.5, seed=SEED)
    counts = np.array(list(syn.stream(z, C, b, seed=SEED + 1)))
    torch.manual_seed(SEED)
    if kind == "vanilla":                                  # original VJF
        model = VJF.make_model(N, 2, 0, 100, hidden_sizes=[100, 100],
                               likelihood="poisson", transition_flow="sgd", encoder="spikes")
        model.decoder.requires_grad_(True)                 # learn C by the ELBO gradient
        live, z_live = counts, z
        gen = online_filter(model, live, readout=None, warmup_steps=200, rbf_width_scale=1.0)
    else:                                                  # sVJF (the fix)
        model = VJF.make_model(N, 2, 0, 100, hidden_sizes=[100, 100],
                               likelihood="poisson", transition_flow="srrls", encoder="projection")
        ro = OnlineReadout(N, 2, smooth_tau=8.0, refresh_K=1000, link="log")
        iw = min(10 * N, T_EFF // 2)
        Cp, bp = ro.warm_start(counts[:iw])
        with torch.no_grad():
            model.decoder.decode.weight.copy_(torch.as_tensor(Cp))
            model.decoder.decode.bias.copy_(torch.as_tensor(bp.reshape(-1)))
        model.decoder.requires_grad_(False)
        live, z_live = counts[iw:], z[iw:]
        gen = online_filter(model, live, readout=ro, warmup_steps=1500, rbf_width_scale=0.5)
    mu = np.zeros((len(live), 2), dtype=np.float32)
    steps, r2f, r2o = [], [], []
    for r in gen:
        mu[r.step] = r.mean
        s = r.step + 1
        if s >= WIN and s % EVERY == 0:
            w = slice(s - WIN, s)
            rf, _ = affine_r2(mu[w], z_live[w])
            steps.append(int(s)); r2f.append(float(rf))
            r2o.append(float(onestep_r2(mu[w], z_live[w], model, device)))
    return {"kind": kind, "n": N, "snr": SNR, "step": steps, "r2_filt": r2f, "r2_onestep": r2o}


if __name__ == "__main__":
    out = {"vanilla": stream_log("vanilla"), "svjf": stream_log("svjf")}
    for k, d in out.items():
        print(f"{k:8s} final filtered R2={d['r2_filt'][-1]:.2f}  one-step R2={d['r2_onestep'][-1]:.2f}")
    with open("vanilla_demo.json", "w") as f:
        json.dump(out, f)
    print("wrote vanilla_demo.json")
