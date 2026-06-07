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


def _bar_labels(ax, bars, errs=None, fmt="%.2f"):
    for i, b in enumerate(bars):
        h = b.get_height()
        off = (errs[i] if errs is not None else 0.0) + 0.025   # clear the error bar
        ax.text(b.get_x() + b.get_width() / 2, h + off, fmt % h, ha="center", va="bottom",
                fontsize=7.5, color=INK)


def auc(kr):
    return float(np.clip(np.asarray(kr), 0, None).mean())


# ---------- Figure: E1 readout stability (C3-revised), high + low SNR ----------
def _readout_arms(data_dir, n):
    by = defaultdict(list)
    for f in glob.glob(os.path.join(data_dir, "*.json")):
        if f"_n{n}_" in os.path.basename(f):
            d = json.load(open(f)); by[d["arm"]].append(d)
    return by


def _plot_readout(ax, by, title):
    arms = [("proj_oracle", "oracle\n(ceiling)"), ("frozen_pca", "frozen PCA\n(good init)"),
            ("freeze_after", "slow/locked\nreadout"), ("online_base", "fast online\nreadout")]
    arms = [a for a in arms if by.get(a[0])]
    def agg(a, fn):
        v = [fn(r) for r in by[a]]; return np.mean(v), np.std(v) / np.sqrt(len(v))
    one = [agg(a, lambda r: r["onestep_r2"]) for a, _ in arms]
    kau = [agg(a, lambda r: auc(r["kpred_r2"])) for a, _ in arms]
    x = np.arange(len(arms)); w = 0.38; _faint_ygrid(ax)
    b1 = ax.bar(x - w/2, [m for m, _ in one], w, yerr=[e for _, e in one], capsize=2.5,
                color=PALETTE["blue"], edgecolor="none", label="one-step $R^2$")
    b2 = ax.bar(x + w/2, [m for m, _ in kau], w, yerr=[e for _, e in kau], capsize=2.5,
                color=PALETTE["amber"], edgecolor="none", label="$k$-step forecast $R^2$ (AUC)")
    _bar_labels(ax, b1, [e for _, e in one]); _bar_labels(ax, b2, [e for _, e in kau])
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in arms]); ax.set_ylim(0, 1.1)
    ax.set_ylabel("$R^2$"); ax.set_title(title)
    return b1, b2


def fig_readout_stability():
    hi = _readout_arms(os.path.join(DATA, "e1"), 250)
    lo = _readout_arms(os.path.join(DATA, "e1_low"), 30)
    if not hi:
        print("skip readout_stability (no e1 data)"); return
    if lo:
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
        _plot_readout(axes[0], hi, "high SNR (8 dB)")
        _plot_readout(axes[1], lo, "low SNR (0 dB)")
        axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.16), ncol=2)
        fig.suptitle("Readout stability: lock once converged at high SNR, keep adapting at low SNR", y=1.0)
    else:
        fig, ax = plt.subplots(figsize=(5.6, 3.7))
        _plot_readout(ax, hi, "Readout stability at high SNR (8 dB, 8 seeds)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
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


# ---------- Figure: E5 per-bin timing (C5) -- violin of the latency distribution ----------
def fig_timing():
    f = os.path.join(DATA, "extra", "x_timing.json")
    if not os.path.exists(f):
        print("skip timing (no raw sample yet)"); return
    d = sorted(json.load(open(f))["conditions"], key=lambda c: c["snr_target"])
    samples = [np.asarray(c.get("timing_sample_ms") or []) for c in d]
    labels = [f"{round(c['snr_target'])} dB\n(n={c['n_neurons']})" for c in d]
    fig, ax = plt.subplots(figsize=(6.0, 3.4)); _faint_ygrid(ax)
    pos = np.arange(1, len(d) + 1)
    parts = ax.violinplot(samples, positions=pos, showextrema=False, widths=0.85)
    for pc in parts["bodies"]:
        pc.set_facecolor(PALETTE["blue"]); pc.set_alpha(0.55)
        pc.set_edgecolor(INK); pc.set_linewidth(0.5)
    for i, s in zip(pos, samples):       # median dot + 5-95% line (honest weight on the bulk)
        if len(s):
            ax.plot([i, i], [np.percentile(s, 5), np.percentile(s, 95)], color=INK, lw=1)
            ax.plot(i, np.percentile(s, 50), "o", color=INK, ms=3)
    ax.axhline(5.0, ls="--", c=INK, lw=1)
    ax.text(pos[-1] + 0.45, 5.15, "5 ms bin budget", ha="right", va="bottom", fontsize=8, color=INK)
    ymax = max(6.0, max((np.percentile(s, 99.5) for s in samples if len(s)), default=6) * 1.15)
    ax.set_ylim(0, ymax)
    ax.set_xticks(pos); ax.set_xticklabels(labels); ax.set_ylabel("per-bin wall time (ms)")
    ax.set_title("Real-time per-bin latency (online readout, e2-standard-4)\n"
                 "violin = distribution; dot = median, bar = 5-95%")
    fig.savefig(os.path.join(FIGS, "timing.pdf")); plt.close(fig)


# ---------- Figure A: 5-SNR method comparison (C1, C2) ----------
def _main_runs():
    by = defaultdict(list)
    for f in glob.glob(os.path.join(DATA, "main", "mo_*.json")):
        mode = os.path.basename(f).split("_")[1]   # mo_<mode>_s<seed>.json
        by[mode].append(json.load(open(f)))
    return by


def fig_summary():
    by = _main_runs()
    if not by:
        print("skip summary (no main data yet)"); return
    modes = [("projoracle", "proj+oracle (ceiling)", METHOD_COLORS["proj_oracle"]),
             ("online", "projection + online readout", METHOD_COLORS["online"]),
             ("frozenpca", "spike + frozen PCA", METHOD_COLORS["frozen_pca"]),
             ("spikeoracle", "spike + oracle", METHOD_COLORS["spike_oracle"])]
    modes = [m for m in modes if by.get(m[0])]
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); _faint_ygrid(ax)
    x = np.arange(len(SNRS)); w = 0.8 / len(modes)
    for j, (mk, lab, col) in enumerate(modes):
        means, ses = [], []
        for s in SNRS:
            vals = [c["r2_final"] for r in by[mk] for c in r["conditions"] if round(c["snr_target"]) == s]
            means.append(np.mean(vals) if vals else np.nan)
            ses.append((np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        ax.bar(x + (j - (len(modes)-1)/2) * w, means, w, yerr=ses, capsize=2,
               color=col, edgecolor="none", label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"{s} dB" for s in SNRS]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("filtered latent $R^2$"); ax.set_xlabel("SNR")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title("Latent recovery across SNR (the gain is largest at low SNR)")
    fig.savefig(os.path.join(FIGS, "summary.pdf")); plt.close(fig)


# ---------- Figure B: online convergence over the stream, per SNR (C2) ----------
def fig_curves():
    by = _main_runs()
    if not by.get("online"):
        print("skip curves (no main online data yet)"); return
    fig, ax = plt.subplots(figsize=(6.0, 3.6)); _faint_ygrid(ax)
    for s in SNRS:
        curves = []
        for r in by["online"]:
            for c in r["conditions"]:
                if round(c["snr_target"]) == s and c.get("log"):
                    curves.append((np.array(c["log"]["step"]), np.array(c["log"]["r2"])))
        if not curves:
            continue
        steps = curves[0][0]
        m = np.nanmean([c[1] for c in curves], axis=0)
        ax.plot(steps, m, color=SNR_COL[s], label=f"{s} dB")
    ax.set_xlabel("stream position (bins)"); ax.set_ylabel("aligned latent $R^2$")
    ax.set_ylim(-0.05, 1.0); ax.legend(title="SNR", loc="lower right")
    ax.set_title("Online readout converges over the stream (slower / lower at low SNR)")
    fig.savefig(os.path.join(FIGS, "curves.pdf")); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_readout_stability, fig_tau, fig_timing, fig_summary, fig_curves):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__}: {e}")
    print("wrote:", ", ".join(sorted(p for p in os.listdir(FIGS) if p.endswith(".pdf"))))
