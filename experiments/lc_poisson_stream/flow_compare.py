"""E7: flow-learner numerical comparison -- Adam/SGD gradient vs square-root RLS (and plain RLS).

The readout is held FIXED at the oracle (projection encoder + frozen oracle C,b) so the only thing
that differs across arms is how the RBF velocity weights W are learned. Recognition is trained by
the same optimizer (Adam) in the RLS arms so the comparison is the flow learner alone:
  sqrt-RLS (sVJF): flow_learner='srrls', optimizer='adam'
  plain RLS:       flow_learner='rls',   optimizer='adam'
  Adam flow:       flow_learner='sgd',   optimizer='adam'   (W trained by Adam with recognition)
  SGD flow:        flow_learner='sgd',   optimizer='sgd'    (the all-SGD original)
Logs filtered-latent R^2 and one-step prediction R^2 vs stream position; 5 seeds.
Self-contained (vjf.synthetic; no neurofisherSNR).

Run: uv run --with numpy --with torch python experiments/lc_poisson_stream/flow_compare.py
"""
import json

import numpy as np
import torch

from vjf import synthetic as syn
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter

SEEDS = [20260605, 20260606, 20260607, 20260608, 20260609]
N, SNR = 50, 3
T_EFF, EVERY, WIN = 60000, 1000, 4000
#        arm        flow_learner  optimizer
ARMS = {"srrls":   ("srrls", "adam"),
        "rls":     ("rls",   "adam"),
        "adam":    ("sgd",   "adam"),
        "sgd":     ("sgd",   "sgd")}


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


def run(arm, seed):
    flow, opt = ARMS[arm]
    z = syn.limit_cycle(T_EFF, dt=5e-3, angular_velocity=30.0, seed=seed)
    C, b = syn.poisson_readout(z, N, mean_rate=0.1, peak_rate=0.5, seed=seed)
    counts = np.array(list(syn.stream(z, C, b, seed=seed + 1)))
    torch.manual_seed(seed)
    model = VJF.make_model(N, 2, 0, 100, hidden_sizes=[100, 100], likelihood="poisson",
                           transition_flow=flow, encoder="projection", optimizer=opt)
    with torch.no_grad():                                    # freeze the readout at the oracle
        model.decoder.decode.weight.copy_(torch.as_tensor(C.astype(np.float32)))
        model.decoder.decode.bias.copy_(torch.as_tensor(b.reshape(-1).astype(np.float32)))
    model.decoder.requires_grad_(False)
    ro = OnlineReadout(N, 2, smooth_tau=8.0, refresh_K=10**9, link="log")
    ro.set_fixed(C.astype(np.float32), b.reshape(-1))        # oracle projection, no adaptation
    gen = online_filter(model, counts, readout=ro, adapt_readout=False,
                        warmup_steps=1500, rbf_width_scale=0.5)
    mu = np.zeros((len(counts), 2), dtype=np.float32)
    steps, rf, ro_ = [], [], []
    for r in gen:
        mu[r.step] = r.mean
        s = r.step + 1
        if s >= WIN and s % EVERY == 0:
            w = slice(s - WIN, s)
            steps.append(int(s)); rf.append(affine_r2(mu[w], z[w]))
            ro_.append(onestep_r2(mu[w], z[w], model))
    return steps, rf, ro_


if __name__ == "__main__":
    out = {}
    for arm in ARMS:
        steps0, filt, one = None, [], []
        for sd in SEEDS:
            st, rf, ro_ = run(arm, sd)
            steps0 = st; filt.append(rf); one.append(ro_)
        out[arm] = {"step": steps0, "r2_filt": filt, "r2_onestep": one}
        print(f"{arm:6s} final filtered R2={np.mean([c[-1] for c in filt]):.2f}  "
              f"one-step R2={np.mean([c[-1] for c in one]):.2f}  (5 seeds)")
    with open("flow_compare.json", "w") as f:
        json.dump(out, f)
    print("wrote flow_compare.json")
