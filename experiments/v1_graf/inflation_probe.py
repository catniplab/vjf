"""What drives the single-direction latent-scale inflation? Measurement only.

Trains the single-direction sVJF (dir 225, L=2, E=20) under three arms and logs the
running latent spread over training:
  - baseline            (readout refresh on, dynamics on)
  - frozen readout      (refresh_K huge -> C never rewritten; tests the CCIPCA
                         eigenvalue-rescaling feedback as the inflation driver)
  - dynamics off        (warmup_trials = all -> flow never turns on; tests the flow)
The arm that does NOT inflate identifies the cause. No core changes; no PLL eval.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask, kmeans_centers
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter_trials

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
BIN_MS, T_MAX_MS, SEED, EPOCHS, WARMUP, NRBF = 10.0, 1400.0, 20260609, 20, 8, 100
matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 130, "savefig.bbox": "tight"})


def _run(train_trials, N, refresh_K, dyn):
    torch.manual_seed(SEED)
    rep = train_trials * EPOCHS
    cover = np.concatenate(train_trials[:WARMUP], 0)
    n_rbf = min(NRBF, cover.shape[0])
    model = VJF.make_model(ydim=N, xdim=2, udim=0, n_rbf=n_rbf, hidden_sizes=[64, 64],
                           likelihood="poisson", transition_flow="srrls", encoder="projection")
    ro = OnlineReadout(N, 2, smooth_tau=8.0, refresh_K=refresh_K, link="log")
    Cw, bw = ro.warm_start(cover)
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cw))
        model.decoder.decode.bias.copy_(torch.as_tensor(bw.reshape(-1)))
    warmup_trials = len(rep) if not dyn else WARMUP        # dyn off => never leave warm-up
    seed_c = None if not dyn else (lambda s: kmeans_centers(s, n_rbf))
    mu, steps = [], []
    for res in online_filter_trials(model, rep, readout=ro, warmup_trials=warmup_trials,
                                    seed_centers=seed_c):
        if res.step % 50 == 0 and not res.diverged:
            mu.append(res.mean.copy()); steps.append(res.step)
    mu = np.asarray(mu); steps = np.asarray(steps)
    w = 200
    spread = np.array([mu[max(0, i - w):i + 1].std(0).mean() for i in range(len(mu))])
    return steps, spread


def main():
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    rng = np.random.default_rng(SEED)
    arr = load_array(5)
    counts = bin_spikes(arr["spk_times"], bin_ms=BIN_MS)
    dirs = arr["ori"]
    mask, _ = well_tuned_mask(counts, dirs)
    counts = counts[:, :, mask]
    N = counts.shape[2]
    nt = int(round(T_MAX_MS / BIN_MS))
    keep = np.unique(dirs)
    d_star = float(max(keep, key=lambda d: counts[dirs == d][:, :nt, :].mean()))
    idx = np.where(dirs == d_star)[0]; rng.shuffle(idx)
    train_trials = [counts[i][:nt] for i in idx[10:]]
    spe = len(train_trials) * nt

    arms = [("baseline (refresh on, dyn on)", dict(refresh_K=500, dyn=True), "C0"),
            ("frozen readout (no refresh)", dict(refresh_K=10 ** 9, dyn=True), "C1"),
            ("dynamics off", dict(refresh_K=500, dyn=False), "C2")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    out = {}
    for label, kw, col in arms:
        s, spread = _run(train_trials, N, **kw)
        ax.plot(s / spe, spread, color=col, lw=1.2, label=label)
        out[label] = float(spread[-1])
        print(f"{label:34s} final latent spread = {spread[-1]:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("latent spread (running std)")
    ax.set_title(f"What drives the latent inflation? (dir {d_star:.0f} deg, L=2, {EPOCHS} ep)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGS, "inflation_probe.png"))
    fig.savefig(os.path.join(FIGS, "inflation_probe.pdf")); plt.close(fig)
    print("fig ->", os.path.join(FIGS, "inflation_probe.png"))


if __name__ == "__main__":
    main()
