"""Visualize the sVJF model's own latent space.

Trains the M1 sVJF (faithful config, moderate train size), infers latent paths on
held-out trials, condition-averages them per direction (0-175 deg), and plots in the
3-D latent space together with the RBF flow centers (native latent coords) sized by
their width. This is the model's native space -- no decoder/PCA mapping.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch

from experiments.v1_graf.graf_loader import load_array, bin_spikes, well_tuned_mask, kmeans_centers
from experiments.v1_graf.run_m1 import _split_trials, _infer_latent_paths
from vjf.model import VJF
from vjf.readout import OnlineReadout
from vjf.realtime import online_filter_trials

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
BIN_MS, T_MAX_MS, SEED = 10.0, 1400.0, 20260609
PER_DIR = 15                      # train trials/dir and (held) infer trials/dir
matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 130, "savefig.bbox": "tight"})


def _color(d):
    return cm.hsv((float(d) % 180.0) / 180.0)


def main():
    os.makedirs(FIGS, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    arr = load_array(5)
    counts = bin_spikes(arr["spk_times"], bin_ms=BIN_MS)
    dirs = arr["ori"]
    mask, _ = well_tuned_mask(counts, dirs)
    counts = counts[:, :, mask]
    N = counts.shape[2]
    nt = int(round(T_MAX_MS / BIN_MS))

    train_by_dir, test_by_dir = _split_trials(dirs, PER_DIR, PER_DIR, rng)
    keep = np.unique(dirs)
    cov = [train_by_dir[d][0] for d in keep]
    rest = [i for d in keep for i in train_by_dir[d][1:]]
    cover_window = np.concatenate([counts[i] for i in cov], 0)
    n_rbf = min(200, cover_window.shape[0])

    model = VJF.make_model(ydim=N, xdim=3, udim=0, n_rbf=n_rbf, hidden_sizes=[100, 100],
                           likelihood="poisson", transition_flow="srrls", encoder="projection")
    ro = OnlineReadout(N, 3, smooth_tau=8.0, refresh_K=1000, link="log")
    Cw, bw = ro.warm_start(cover_window)
    with torch.no_grad():
        model.decoder.decode.weight.copy_(torch.as_tensor(Cw))
        model.decoder.decode.bias.copy_(torch.as_tensor(bw.reshape(-1)))
    train_trials = [counts[i] for i in cov + rest]
    for _ in online_filter_trials(model, train_trials, readout=ro, warmup_trials=len(cov),
                                  seed_centers=lambda s: kmeans_centers(s, n_rbf)):
        pass

    centers = model.transition.velocity.feature.centroid.detach().cpu().numpy()   # (n_rbf, 3)
    widths = np.exp(model.transition.velocity.feature.logwidth.detach().cpu().numpy())

    # condition-averaged inferred latent per direction (0-175), over held-out test trials
    sel = keep[(keep >= 0) & (keep < 180)]
    cond_avg = []
    for d in sel:
        tr = [counts[i] for i in test_by_dir[d]]
        paths = _infer_latent_paths(model, ro, tr)               # list of (n_bin, 3)
        stk = np.stack([p[:nt] for p in paths], 0)               # (n_trial, nt, 3)
        cond_avg.append(stk.mean(0))
    cond_avg = np.asarray(cond_avg)                              # (D, nt, 3)

    # save the computed arrays before plotting (so a plot hiccup can't waste the train)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    np.savez(os.path.join(HERE, "results", "latent_space_L3.npz"),
             cond_avg=cond_avg, centers=centers, widths=widths, sel=sel)

    # marker size ~ RBF width (native latent radius)
    sizes = 20 + 360 * (widths - widths.min()) / (np.ptp(widths) + 1e-12)

    fig = plt.figure(figsize=(7.8, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    for i, d in enumerate(sel):
        t = cond_avg[i]
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=_color(d), lw=1.5, alpha=0.30)
        ax.scatter(*t[0], color=_color(d), s=12, alpha=0.6)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=sizes, c="k",
               alpha=0.75, edgecolors="none", label="RBF centers (size ~ width)")
    ax.set_xlabel("latent 1"); ax.set_ylabel("latent 2"); ax.set_zlabel("latent 3")
    ax.view_init(elev=22, azim=-60)
    ax.set_title("sVJF latent space: condition-averaged inferred trajectories (transparent)\n"
                 f"+ RBF flow centers/widths; dir 0-175 deg, {N} neurons")
    ax.legend(loc="upper left", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(0, 180))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.10); cb.set_label("direction (deg)")
    fig.savefig(os.path.join(FIGS, "latent_space_3d.png"))
    fig.savefig(os.path.join(FIGS, "latent_space_3d.pdf")); plt.close(fig)

    print(f"n_rbf={len(centers)}  latent dims std (cond-avg) = {np.round(cond_avg.reshape(-1,3).std(0),4)}")
    print(f"RBF center std per latent dim = {np.round(centers.std(0),4)}")
    print(f"RBF width (min,med,max) = {widths.min():.3g},{np.median(widths):.3g},{widths.max():.3g}")
    print("fig ->", os.path.join(FIGS, "latent_space_3d.png"))


if __name__ == "__main__":
    main()
