"""Render M1 diagnostic figures from diag_m1.npz into report_m1/figs/.

Figures: (1) decode contrast bar, (2) inferred single-trial latent trajectories
(dynamics ON vs no-dynamics control), (3) orientation tuning (neurons + latent),
(4) training convergence (ELBO components), (5) trial-averaged latent per direction
(the torus attempt), ON vs control.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight",
})
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "results", "diag_m1_array5_L3.npz")
FIGS = os.path.join(HERE, "report_m1", "figs")
CHANCE = 1.0 / 72


def _circ_colors(dirs):
    return plt.cm.hsv((np.asarray(dirs) % 360) / 360.0)


def fig_decode(d):
    # information cascade through the encoder pipeline (readout pi numbers from diag_readout.py)
    labels = ["raw\ncounts", "log-rate\nfeature", "PCA-3\nfeature\n(ideal)",
              "readout pi\n(warm-start)", "readout pi\n(online)",
              "sVJF latent\n(dyn ON)", "sVJF latent\n(no dyn)"]
    vals = [0.338, 0.297, 0.234, 0.126, 0.103, float(d["dec_a"]), float(d["dec_b"])]
    cols = ["0.7", "0.7", "0.45", "#e8a", "#e8a", "#c44", "#48c"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(range(len(vals)), vals, color=cols)
    ax.axhline(CHANCE, ls="--", lw=0.8, color="k")
    ax.text(len(vals) - 0.6, CHANCE + 0.006, "chance (1/72)", ha="right", va="bottom", fontsize=7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5)
    # annotate the two loss stages
    ax.annotate("", xy=(3, 0.16), xytext=(2, 0.245), arrowprops=dict(arrowstyle="->", color="#a33"))
    ax.text(2.5, 0.255, "readout subspace\n(46.5 deg off): -0.11", fontsize=6.5, color="#a33", ha="center")
    ax.annotate("", xy=(5, 0.03), xytext=(4, 0.10), arrowprops=dict(arrowstyle="->", color="#a33"))
    ax.text(4.7, 0.115, "recognition\ncollapse: -0.09", fontsize=6.5, color="#a33", ha="center")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("single-trial direction decode acc")
    ax.set_ylim(0, 0.40)
    ax.set_title("Where orientation is lost: readout subspace, then recognition collapse")
    fig.savefig(os.path.join(FIGS, "fig1_decode_cascade.pdf"))
    fig.savefig(os.path.join(FIGS, "fig1_decode_cascade.png")); plt.close(fig)


def _proj2(summary, paths_list):
    mu = summary.mean(0)
    _, _, vt = np.linalg.svd(summary - mu, full_matrices=False)
    W = vt[:2].T
    return [(p - mu) @ W for p in paths_list]


def fig_traj(d):
    rep = [int(x) for x in d["rep_dirs"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    for ax, tag, summ, title in ((axes[0], "A", d["summ_a"], "dynamics ON (sVJF, M1)"),
                                  (axes[1], "B", d["summ_b"], "no-dynamics control")):
        allp = []
        for dd in rep:
            allp += list(d[f"paths{tag}_{dd}"])
        proj = _proj2(summ, allp)
        k = 0
        for dd in rep:
            n = len(d[f"paths{tag}_{dd}"])
            col = plt.cm.hsv((dd % 360) / 360.0)
            for _ in range(n):
                p = proj[k]; k += 1
                ax.plot(p[:, 0], p[:, 1], lw=0.8, color=col, alpha=0.8)
                ax.plot(p[0, 0], p[0, 1], "o", ms=3, color=col)
        ax.set_title(title); ax.set_xlabel("latent PC1"); ax.set_ylabel("latent PC2")
        ax.set_aspect("equal", "datalim")
    sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(0, 360))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02); cb.set_label("grating direction (deg)")
    fig.suptitle("Inferred single-trial latent trajectories (stimulus window), dot = start")
    fig.savefig(os.path.join(FIGS, "fig2_trajectories.pdf"))
    fig.savefig(os.path.join(FIGS, "fig2_trajectories.png")); plt.close(fig)


def fig_tuning(d):
    tc, axis, r2 = d["tc_kept"], d["tc_axis"], d["r2_kept"]
    order = np.argsort(r2)[::-1][:6]                 # six best-tuned neurons
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 5.2))
    for i in order:
        axes[0].plot(axis, tc[i], lw=1.0)
    axes[0].set_title("Example V1 neuron tuning (6 best-tuned)")
    axes[0].set_xlabel("grating direction (deg)"); axes[0].set_ylabel("rate (Hz)")
    axes[0].set_xlim(0, 360)
    # latent "tuning": each latent coord vs direction (trial-averaged), dynamics ON
    axis2 = d["torus_axis"]
    avg = np.stack([d["summ_a"][d["test_dirs"] == dd].mean(0) for dd in axis2], 0)
    for j in range(avg.shape[1]):
        axes[1].plot(axis2, avg[:, j], lw=1.0, label=f"latent {j+1}")
    axes[1].set_title("sVJF latent vs direction (dynamics ON) -- nearly flat")
    axes[1].set_xlabel("grating direction (deg)"); axes[1].set_ylabel("trial-avg latent")
    axes[1].set_xlim(0, 360); axes[1].legend(fontsize=7, ncol=3, loc="upper right")
    fig.savefig(os.path.join(FIGS, "fig3_tuning.pdf"))
    fig.savefig(os.path.join(FIGS, "fig3_tuning.png")); plt.close(fig)


def _smooth(x, w=21):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def fig_convergence(d):
    s = d["steps"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for key, lab in (("recon", "recon (-loglik)"), ("dynamics", "dynamics"),
                     ("entropy", "entropy"), ("loss", "total (-ELBO)")):
        y = d[key]
        ys = _smooth(y)
        xs = s[len(s) - len(ys):]
        ax.plot(xs, ys, lw=1.0, label=lab)
    ax.set_xlabel("online step (training stream)")
    ax.set_ylabel("ELBO component (smoothed)")
    ax.set_title("Training convergence over the single online pass")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIGS, "fig4_convergence.pdf"))
    fig.savefig(os.path.join(FIGS, "fig4_convergence.png")); plt.close(fig)


def fig_torus(d):
    axis = d["torus_axis"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    for ax, t, title in ((axes[0], d["torus_a"], "dynamics ON (sVJF, M1)"),
                         (axes[1], d["torus_b"], "no-dynamics control")):
        t = np.asarray(t)
        ax.scatter(t[:, 0], t[:, 1], c=_circ_colors(axis), s=18)
        ax.plot(t[:, 0], t[:, 1], lw=0.5, color="0.7", zorder=0)
        ax.set_title(title); ax.set_xlabel("SV1"); ax.set_ylabel("SV2")
        ax.set_aspect("equal", "datalim")
    sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(0, 360))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02); cb.set_label("grating direction (deg)")
    fig.suptitle("Trial-averaged latent per direction (top 2 singular vectors)")
    fig.savefig(os.path.join(FIGS, "fig5_torus.pdf"))
    fig.savefig(os.path.join(FIGS, "fig5_torus.png")); plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    d = dict(np.load(NPZ, allow_pickle=True))
    fig_decode(d); fig_traj(d); fig_tuning(d); fig_convergence(d); fig_torus(d)
    print("decode A (dynamics ON):", round(float(d["dec_a"]), 4))
    print("decode B (no dynamics):", round(float(d["dec_b"]), 4))
    print("figures ->", FIGS)


if __name__ == "__main__":
    main()
