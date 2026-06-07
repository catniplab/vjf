"""Regenerate publication figures (vector PDF) from the staged result JSONs in paper/data/.

Run:  uv run --with matplotlib --with numpy python paper/figs_src/plot_results.py
Writes paper/figs/*.pdf. Fonts embedded (Type 42), sized for the 11pt body at ~scale 1.
"""
from __future__ import annotations
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

# --- shared muted / information-designer palette (no saturated red/magenta/pure-green) ---
INK = "#3b3b3b"
PALETTE = {
    "blue":  "#6a8ec9",   # muted slate blue
    "amber": "#e0a458",   # muted amber
    "teal":  "#6fb3a8",   # muted teal
    "mauve": "#b08fb0",   # muted lavender
    "slate": "#9aa7b0",   # muted gray-blue
    "olive": "#a6b06a",   # soft olive
}
# canonical method colors (reused across all figures)
METHOD_COLORS = {
    "proj_oracle": PALETTE["slate"], "oracle": PALETTE["slate"],
    "frozen_pca": PALETTE["amber"], "frozenpca": PALETTE["amber"],
    "freeze_after": PALETTE["teal"],
    "online": PALETTE["blue"], "online_base": PALETTE["blue"],
    "spike_oracle": PALETTE["mauve"], "spikeoracle": PALETTE["mauve"],
}
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "lines.linewidth": 1.8, "savefig.bbox": "tight", "legend.frameon": False,
})
SNRS = [-3, 0, 3, 6, 8]
# muted sequential for ordered SNR (cividis: colorblind-safe, blue->gold, no green/red)
SNR_COL = {s: plt.cm.cividis(v) for s, v in zip(SNRS, np.linspace(0.12, 0.9, len(SNRS)))}


def _faint_ygrid(ax):
    ax.yaxis.grid(True, color="0.85", lw=0.6, zorder=0); ax.set_axisbelow(True)


def _bar_labels(ax, bars, fmt="%.2f"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.012, fmt % h, ha="center", va="bottom",
                fontsize=7.5, color=INK)


def auc(kr):
    return float(np.clip(np.asarray(kr), 0, None).mean())


# ---------- Figure: E1 readout stability (C3-revised), high SNR ----------
def fig_readout_stability():
    by = defaultdict(list)
    for f in glob.glob(os.path.join(DATA, "e1", "*.json")):
        d = json.load(open(f)); by[d["arm"]].append(d)
    arms = [("proj_oracle", "oracle\n(ceiling)"), ("frozen_pca", "frozen PCA\n(good init)"),
            ("freeze_after", "freeze-after\n(our recipe)"), ("online_base", "online\n(keep refreshing)")]
    one = [(np.mean([r["onestep_r2"] for r in by[a]]), np.std([r["onestep_r2"] for r in by[a]]) / np.sqrt(len(by[a]))) for a, _ in arms]
    kau = [(np.mean([auc(r["kpred_r2"]) for r in by[a]]), np.std([auc(r["kpred_r2"]) for r in by[a]]) / np.sqrt(len(by[a]))) for a, _ in arms]
    x = np.arange(len(arms)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _faint_ygrid(ax)
    b1 = ax.bar(x - w/2, [m for m, _ in one], w, yerr=[e for _, e in one], capsize=2.5,
                color=PALETTE["blue"], edgecolor="none", label="one-step $R^2$")
    b2 = ax.bar(x + w/2, [m for m, _ in kau], w, yerr=[e for _, e in kau], capsize=2.5,
                color=PALETTE["amber"], edgecolor="none", label="$k$-step forecast $R^2$ (AUC)")
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in arms])
    ax.set_ylim(0, 1.05); ax.set_ylabel("$R^2$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.set_title("Readout stability at high SNR (8 dB, 8 seeds)")
    fig.savefig(os.path.join(FIGS, "readout_stability.pdf")); plt.close(fig)


# ---------- Figure: E3 smoothing tau trend (C4) ----------
def fig_tau():
    taus = [1, 2, 4, 8, 16, 32]
    rate = {s: [] for s in SNRS}; r2 = {s: [] for s in SNRS}
    for t in taus:
        d = json.load(open(os.path.join(DATA, "e3", f"e3_online_tau{t}.json")))["conditions"]
        by = {round(c["snr_target"]): c for c in d}
        for s in SNRS:
            rate[s].append(by[s]["rate_corr"] if s in by else np.nan)
            r2[s].append(by[s]["r2_final"] if s in by else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for s in SNRS:
        axes[0].plot(taus, rate[s], "o-", color=SNR_COL[s], ms=4)
        axes[1].plot(taus, r2[s], "o-", color=SNR_COL[s], ms=4, label=f"{s} dB")
    for ax, ttl, yl in zip(axes, ["rate correlation", "filtered latent $R^2$"], ["rate corr.", "$R^2$"]):
        _faint_ygrid(ax)
        ax.set_xscale("log", base=2); ax.set_xticks(taus); ax.set_xticklabels(taus)
        ax.set_xlabel(r"smoothing $\tau$ (bins)"); ax.set_ylabel(yl); ax.set_ylim(0, 1.0); ax.set_title(ttl)
    axes[1].legend(title="SNR", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(r"Smoothing helps most at low SNR; over-smoothing ($\tau\gtrsim16$) hurts (lag)", y=1.02)
    fig.savefig(os.path.join(FIGS, "tau_sweep.pdf")); plt.close(fig)


# ---------- Figure: E5 per-bin timing (C5) ----------
def fig_timing():
    d = json.load(open(os.path.join(DATA, "e2", "e2_K1000.json")))["conditions"]
    d = sorted(d, key=lambda c: c["snr_target"])
    labels = [f"{round(c['snr_target'])} dB\n(n={c['n_neurons']})" for c in d]
    p50 = [c.get("per_bin_p50_ms") for c in d]; p95 = [c.get("per_bin_p95_ms") for c in d]
    pmax = [c.get("per_bin_max_ms") for c in d]
    x = np.arange(len(d)); w = 0.27
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    _faint_ygrid(ax)
    ax.bar(x - w, p50, w, label="p50", color=PALETTE["slate"], edgecolor="none")
    ax.bar(x, p95, w, label="p95", color=PALETTE["blue"], edgecolor="none")
    ax.bar(x + w, pmax, w, label="max", color=PALETTE["amber"], edgecolor="none")
    ax.axhline(5.0, ls="--", c=INK, lw=1)
    ax.text(len(d) - 0.5, 5.15, "5 ms bin budget", ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("per-bin wall time (ms)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="percentile")
    ax.set_title("Real-time per-bin cost (online readout, e2-standard-4)")
    fig.savefig(os.path.join(FIGS, "timing.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_readout_stability(); fig_tau(); fig_timing()
    print("wrote:", ", ".join(sorted(os.listdir(FIGS))))
