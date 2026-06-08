"""Regenerate publication figures (vector PDF) from the staged result JSONs in paper/data/.

Run:  uv run --with matplotlib --with numpy python paper/figs_src/plot_results.py
Writes paper/figs/*.pdf. Fonts embedded (Type 42), sized for the 11pt body at ~scale 1.
"""
from __future__ import annotations
import glob
import json
import os
import re
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
DT = 5e-3              # s per bin -> forecast horizon is a general predictability TIME
PERIOD = 42.0          # limit-cycle period (bins); 1 period ~ 0.21 s for THIS benchmark
KCAP = 200             # k-step evaluation cap
CAP_S = KCAP * DT      # eval horizon in seconds (1.0 s)
CYC_S = PERIOD * DT    # seconds per limit-cycle period (~0.21 s)
RHO = 0.5              # forecast retains this fraction of the FILTERING accuracy

ARMS = [("proj_oracle", "oracle\n$C$"), ("frozen_pca", "frozen\nPCA"),
        ("freeze_after", "slow/\nlocked"), ("online_base", "fast\nonline")]


def _readout_arms(data_dir, n):
    by = defaultdict(list)
    for f in glob.glob(os.path.join(data_dir, "*.json")):
        if f"_n{n}_" in os.path.basename(f):
            d = json.load(open(f)); by[d["arm"]].append(d)
    return by


def _fc_horizon_s(d):
    """Predictive horizon as a TIME (s): first lead k with R2_forecast(k) < RHO*R2_filter,
    times the bin width. Threshold is relative to the achievable filtering accuracy (so it
    stays meaningful when that accuracy is low); censored at the evaluation cap."""
    r2 = np.asarray(d["kpred_r2"]); thr = RHO * d["r2_final"]
    below = np.where(r2 < thr)[0]
    k = int(below[0]) if len(below) else KCAP
    return k * DT


def _agg(by, arm, fn):
    v = [fn(r) for r in by[arm]]
    return np.mean(v), np.std(v) / np.sqrt(len(v))


def _bars(ax, by, arms, fn, ylabel, ylim, labfmt, cap=None, title=None):
    _faint_ygrid(ax)
    x = np.arange(len(arms)); mh = ylim[1]
    for i, (a, _) in enumerate(arms):
        m, e = _agg(by, a, fn)
        censored = cap is not None and m >= cap - 1e-6
        ax.bar(i, m, 0.66, yerr=(None if censored else e), capsize=2.5,
               color=METHOD_COLORS[a], edgecolor="none", hatch=("////" if censored else None))
        top = m + (0.0 if censored else e) + mh * 0.03   # clear the error bar (per prior feedback)
        lab = (">%.1f" % cap) if censored else (labfmt % m)
        ax.text(i, top, lab, ha="center", va="bottom", fontsize=7.5, color=INK)
    if cap is not None:
        ax.axhline(cap, ls="--", lw=0.8, color="0.55")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in arms]); ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def fig_readout_stability():
    hi = _readout_arms(os.path.join(DATA, "e1"), 250)
    lo = _readout_arms(os.path.join(DATA, "e1_low"), 30)
    if not hi:
        print("skip readout_stability (no e1 data)"); return
    cols = [("high SNR (8 dB)", hi)] + ([("low SNR (0 dB)", lo)] if lo else [])
    fig, axes = plt.subplots(2, len(cols), figsize=(4.9 * len(cols), 5.7), squeeze=False)
    for j, (ttl, by) in enumerate(cols):
        arms = [a for a in ARMS if by.get(a[0])]
        _bars(axes[0][j], by, arms, lambda r: r["onestep_r2"],
              "one-step $R^2$", (0, 1.14), "%.2f", title=ttl)
        _bars(axes[1][j], by, arms, _fc_horizon_s,
              "forecast horizon (s)", (0, CAP_S * 1.22), "%.2f", cap=CAP_S)
        axes[1][j].text(len(arms) - 0.5, CAP_S, "eval cap", ha="right", va="bottom",
                        fontsize=6.5, color="0.45")
    # right-edge secondary axis: the same horizon in limit-cycle periods (this benchmark)
    sec = axes[1][-1].secondary_yaxis("right", functions=(lambda s: s / CYC_S, lambda c: c * CYC_S))
    sec.set_ylabel("periods (this example)")
    fig.suptitle("Lock the readout once converged at high SNR (forecast horizon "
                 r"$\to$ oracle); keep adapting at low SNR", y=0.995, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIGS, "readout_stability.pdf")); plt.close(fig)


# ---------- Supplementary: k-step forecast curves underlying the horizon ----------
def fig_kstep_curves():
    hi = _readout_arms(os.path.join(DATA, "e1"), 250)
    lo = _readout_arms(os.path.join(DATA, "e1_low"), 30)
    if not hi:
        print("skip kstep_curves (no e1 data)"); return
    cols = [("high SNR (8 dB)", hi)] + ([("low SNR (0 dB)", lo)] if lo else [])
    fig, axes = plt.subplots(1, len(cols), figsize=(4.9 * len(cols), 3.6),
                             sharey=True, squeeze=False)
    x = np.arange(1, KCAP + 1) * DT
    for j, (ttl, by) in enumerate(cols):
        ax = axes[0][j]; _faint_ygrid(ax)
        ax.axhline(0, lw=0.6, color="0.6")
        for a, lab in ARMS:
            if not by.get(a):
                continue
            curve = np.mean([d["kpred_r2"] for d in by[a]], 0)
            rf = float(np.mean([d["r2_final"] for d in by[a]]))
            ax.plot(x, curve, color=METHOD_COLORS[a], label=lab.replace("\n", " "))
            below = np.where(curve < RHO * rf)[0]          # mark the horizon crossing
            if len(below):
                ax.plot(x[below[0]], curve[below[0]], "o", color=METHOD_COLORS[a], ms=4, zorder=5)
        ax.set_xlim(0, CAP_S); ax.set_ylim(-0.25, 1.05)
        ax.set_xlabel("forecast lead (s)"); ax.set_title(ttl)
        sec = ax.secondary_xaxis("top", functions=(lambda s: s / CYC_S, lambda c: c * CYC_S))
        if j == len(cols) - 1:
            sec.set_xlabel("periods (this example)")
        if j == 0:
            ax.set_ylabel("free-run forecast $R^2$")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="readout")
    fig.suptitle(r"$k$-step free-run forecast skill ($\bullet$ = horizon, where $R^2$ drops "
                 r"below half the filtering accuracy)", y=1.04, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "kstep_curves.pdf")); plt.close(fig)


# ---------- Figure: the problem -- original VJF does not converge online (E6, motivator) ----------
def fig_motivation():
    f = os.path.join(DATA, "extra", "motivation_compare.json")
    if not os.path.exists(f):
        print("skip motivation (no motivation_compare data)"); return
    d = json.load(open(f))
    methods = [("orig_randC_adam", "orig.\\ VJF: random $C$, Adam", PALETTE["mauve"]),
               ("orig_randC_sgd", "orig.\\ VJF: random $C$, SGD", PALETTE["slate"]),
               ("orig_oracleC_adam", "orig.\\ VJF: oracle $C$, Adam", PALETTE["amber"]),
               ("orig_oracleC_sgd", "orig.\\ VJF: oracle $C$, SGD", PALETTE["olive"]),
               ("svjf", "sVJF (ours)", PALETTE["blue"])]
    methods = [m for m in methods if m[0] in d]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7), sharex=True, sharey=True)
    for key, lab, col in methods:
        e = d[key]; x = np.asarray(e["step"])
        for ax, fld in ((axes[0], "r2_filt"), (axes[1], "r2_onestep")):
            arr = np.asarray(e[fld]); m = arr.mean(0); se = arr.std(0) / np.sqrt(len(arr))
            lw = 2.3 if key == "svjf" else 1.5
            ax.plot(x, m, color=col, lw=lw, label=lab)
            ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, lw=0)
    for ax, ttl in ((axes[0], "filtered-latent $R^2$"), (axes[1], "one-step prediction $R^2$")):
        _faint_ygrid(ax); ax.axhline(0, lw=0.6, color="0.6")
        ax.set_xlabel("stream position (time steps, 5\\,ms bins)")
        ax.set_ylim(-0.18, 1.0); ax.set_title(ttl)
    axes[0].set_ylabel("$R^2$ (best affine)")
    axes[1].legend(loc="lower right", fontsize=8, ncol=1)
    fig.suptitle("Original VJF needs the right readout to converge online; sVJF reaches it from spikes "
                 "(mean $\\pm$ s.e., 5 seeds)", y=1.02, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "motivation.pdf")); plt.close(fig)


# ---------- Figure (E7): flow-learner convergence (readout fixed at oracle) ----------
def fig_flow():
    f = os.path.join(DATA, "extra", "flow_compare.json")
    if not os.path.exists(f):
        print("skip flow (no flow_compare data)"); return
    d = json.load(open(f))
    arms = [("srrls", "square-root RLS (sVJF)", PALETTE["blue"]),
            ("sgd", "SGD flow", PALETTE["teal"]),
            ("adam", "Adam flow", PALETTE["amber"]),
            ("rls", "plain RLS", PALETTE["mauve"])]
    arms = [a for a in arms if a[0] in d]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True, sharey=True)
    x = np.asarray(d["srrls"]["step"])
    for key, lab, col in arms:
        for ax, fld in ((axes[0], "r2_filt"), (axes[1], "r2_onestep")):
            a = np.asarray(d[key][fld], dtype=float)
            with np.errstate(invalid="ignore"):
                m = np.nanmean(a, 0); se = np.nanstd(a, 0) / np.sqrt(max(len(a), 1))
            mc = np.clip(m, -0.3, None)                   # clip diverging plain RLS for display
            lw = 2.3 if key == "srrls" else 1.6
            ax.plot(x, mc, color=col, lw=lw, label=(lab if ax is axes[1] else None))
            if key != "rls":
                ax.fill_between(x, np.clip(m - se, -0.3, None), m + se, color=col, alpha=0.15, lw=0)
    for ax, ttl in ((axes[0], "filtered-latent $R^2$"), (axes[1], "one-step prediction $R^2$")):
        _faint_ygrid(ax); ax.axhline(0, lw=0.6, color="0.6")
        ax.set_xlabel("stream position (time steps, 5\\,ms bins)"); ax.set_ylim(-0.3, 1.0)
        ax.set_xlim(x[0], min(9000, x[-1])); ax.set_title(ttl)   # zoom on the convergence/crossover
    axes[0].set_ylabel("$R^2$ (best affine)")
    axes[0].text(x[0], -0.26, "plain RLS diverges (off scale)", fontsize=7, color=PALETTE["mauve"])
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Flow learner, readout fixed at oracle (early stream; stable arms stay flat to 60k): "
                 "square-root RLS leads early, plain RLS diverges", y=1.02, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "flow.pdf")); plt.close(fig)


# ---------- Figure (dormant until the SNR sweep lands): readout schedule vs SNR ----------
N_SNR_BY_N = {15: -3, 30: 0, 50: 3, 150: 6, 250: 8}


def _readout_all_snr():
    """Gather readout-schedule runs across all SNR (n) from the staged e1 dirs -> {snr: {arm: [d]}}."""
    by = defaultdict(lambda: defaultdict(list))
    for sub in ("e1", "e1_low", "e1_mid"):
        for f in glob.glob(os.path.join(DATA, sub, "*.json")):
            m = re.search(r"_n(\d+)_", os.path.basename(f))
            if not m or int(m.group(1)) not in N_SNR_BY_N:
                continue
            d = json.load(open(f)); by[N_SNR_BY_N[int(m.group(1))]][d["arm"]].append(d)
    return by


def fig_readout_snr():
    by = _readout_all_snr()
    snrs = sorted(by)
    if len(snrs) < 4:
        print(f"skip readout_snr (only {len(snrs)} SNR present)"); return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))   # match fig_tau aesthetics/size
    for a, lab in ARMS:
        lab = lab.replace("\n", " ")
        m1, e1, mh, eh, cens = [], [], [], [], []
        for s in snrs:
            rs = by[s].get(a, [])
            if rs:
                o = [r["onestep_r2"] for r in rs]; m1.append(np.mean(o)); e1.append(np.std(o) / np.sqrt(len(o)))
                h = [_fc_horizon_s(r) for r in rs]; mh.append(np.mean(h)); eh.append(np.std(h) / np.sqrt(len(h)))
                cens.append(np.mean(h) >= CAP_S - 1e-6)
            else:
                m1.append(np.nan); e1.append(0); mh.append(np.nan); eh.append(0); cens.append(False)
        axes[0].plot(snrs, m1, "o-", color=METHOD_COLORS[a], ms=4, label=lab)
        axes[1].plot(snrs, mh, "o-", color=METHOD_COLORS[a], ms=4)
        for s, h, c in zip(snrs, mh, cens):     # up-tick = forecast never decays within the window
            if c:
                axes[1].annotate("", xy=(s, CAP_S * 1.04), xytext=(s, CAP_S),
                                 arrowprops=dict(arrowstyle="-|>", color=METHOD_COLORS[a], lw=1.2))
    for ax, ttl, yl, ylim in zip(axes, ["one-step prediction", "free-run forecast"],
                                 ["$R^2$", "horizon (s)"], [(0, 1.0), (0, CAP_S * 1.12)]):
        _faint_ygrid(ax); ax.set_xlabel("SNR (dB)"); ax.set_xticks(snrs)
        ax.set_ylabel(yl); ax.set_ylim(*ylim); ax.set_title(ttl)
    axes[1].axhline(CAP_S, ls="--", lw=0.8, color="0.55")
    axes[1].legend(title="readout", loc="upper left", fontsize=8, title_fontsize=8)
    fig.suptitle("Readout schedule across SNR: adapt at low SNR, lock once converged at high SNR", y=1.02)
    fig.savefig(os.path.join(FIGS, "readout_snr.pdf")); plt.close(fig)


# ---------- Figure: input rasters across SNR (shared latent) ----------
def fig_raster():
    try:
        from vjf import synthetic as syn
    except Exception as e:
        print(f"skip raster (vjf import: {e})"); return
    dt, T, seed = 5e-3, 600, 20260605                       # 3 s window
    z = syn.limit_cycle(T, dt=dt, angular_velocity=30.0, seed=seed)  # SAME latent for all panels
    t = np.arange(T) * dt
    conds = [(15, -3), (30, 0), (50, 3), (150, 6), (250, 8)]
    fig = plt.figure(figsize=(7.0, 9.0))
    gs = fig.add_gridspec(len(conds) + 1, 1, height_ratios=[55] + [n for n, _ in conds], hspace=0.15)
    # top: the single shared latent trajectory every population observes
    az = fig.add_subplot(gs[0])
    az.plot(t, z[:, 0], color=PALETTE["blue"], lw=1.3)
    az.plot(t, z[:, 1], color=PALETTE["amber"], lw=1.3)
    az.set_xlim(0, t[-1]); az.set_xticks([]); az.set_yticks([])
    az.set_ylabel("latent\nstate", rotation=0, ha="right", va="center", fontsize=9)
    az.set_title("One latent trajectory, observed by five populations (SNR set by neuron count)",
                 fontsize=10.5)
    for sp in ("top", "right", "bottom"):
        az.spines[sp].set_visible(False)
    for i, (n, snr) in enumerate(conds):
        ax = fig.add_subplot(gs[i + 1], sharex=az)
        C, b = syn.poisson_readout(z, n, mean_rate=0.1, peak_rate=0.5, seed=seed)
        counts = np.array(list(syn.stream(z, C, b, seed=seed + 1)))      # T x n
        order = np.argsort(np.arctan2(C[:, 1], C[:, 0]))                 # sort by preferred phase
        spikes = [t[counts[:, j] > 0] for j in order]
        ax.eventplot(spikes, lineoffsets=np.arange(n), linelengths=0.9,
                     linewidths=0.4, colors=INK)
        ax.set_ylim(-0.5, n - 0.5); ax.set_yticks([])
        ax.set_ylabel(f"$n{{=}}{n}$\n{snr:+d} dB", rotation=0, ha="right", va="center", fontsize=9)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        if i < len(conds) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time (s)", fontsize=10)
    fig.savefig(os.path.join(FIGS, "raster.pdf")); plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); _faint_ygrid(ax)
    pos = np.arange(1, len(d) + 1)
    # box = IQR, line = median, whiskers = 5-95% -- the bulk dominates; rare host-jitter
    # excursions are not given equal visual weight (they live in the upper whisker only).
    for i, s in zip(pos, samples):
        if not len(s):
            continue
        q1, q2, q3 = np.percentile(s, [25, 50, 75]); p5, p95 = np.percentile(s, [5, 95])
        ax.add_patch(plt.Rectangle((i - 0.22, q1), 0.44, q3 - q1, facecolor=PALETTE["blue"],
                                   alpha=0.55, edgecolor=INK, lw=0.6, zorder=3))
        ax.plot([i - 0.22, i + 0.22], [q2, q2], color=INK, lw=1.5, zorder=4)
        ax.plot([i, i], [p5, q1], color=INK, lw=0.8); ax.plot([i, i], [q3, p95], color=INK, lw=0.8)
    ax.axhline(5.0, ls="--", c="0.5", lw=1)
    ax.text(pos[-1] + 0.5, 5.08, "5 ms bin budget", ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, 6.0); ax.set_xlim(0.5, len(d) + 0.5)
    ax.set_xticks(pos); ax.set_xticklabels(labels); ax.set_ylabel("per-bin wall time (ms)")
    ax.set_title("Real-time per-bin latency (single core, e2-standard-4)\n"
                 "box = IQR, line = median, whiskers = 5-95%")
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


# ---------- Figure: conceptual timescales (W, K, tau) ----------
def fig_timescales():
    from matplotlib.patches import ConnectionPatch, FancyArrowPatch
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 4.8))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # --- (1) macro: never-ending real-time stream, warm-up W then ongoing online learning ---
    a = axes[0]
    xend, Wf = 0.90, 0.12
    a.add_patch(plt.Rectangle((0, 0.35), xend, 0.3, fc="#eef1f4", ec="none"))
    a.add_patch(plt.Rectangle((0, 0.35), Wf, 0.3, fc=PALETTE["amber"], alpha=0.5, ec="none"))
    a.plot([0, 0, xend], [0.35, 0.65, 0.65], color=INK, lw=0.8)   # left + top edges
    a.plot([0, xend], [0.35, 0.35], color=INK, lw=0.8)            # bottom edge (right stays open)
    a.axvline(Wf, ymin=0.35, ymax=0.65, color=INK, lw=1, ls="--")
    a.text(Wf/2, 0.5, "warm-up\n$W$", ha="center", va="center", fontsize=8)
    a.text((Wf+xend)/2, 0.5, "online learning  (per-bin filter + dynamics; readout every $K$)",
           ha="center", va="center", fontsize=8)
    a.annotate("", xy=(1.0, 0.5), xytext=(xend, 0.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    a.text((xend + 1) / 2, 0.7, r"$\cdots$ ongoing (real time)", ha="center", va="center", fontsize=8)
    a.set_title("Three timescales of sVJF", fontsize=11, loc="left")

    # --- (2) meso: readout refresh every K (slow outer loop) ---
    b = axes[1]
    b.add_patch(plt.Rectangle((0, 0.4), 1, 0.22, fc="#eef1f4", ec=INK, lw=0.8))
    for xk in np.linspace(0.1, 0.9, 5):
        b.add_patch(FancyArrowPatch((xk, 0.62), (xk, 0.78), arrowstyle="-|>", mutation_scale=8, color=PALETTE["teal"], lw=1.4))
    b.annotate("", xy=(0.3, 0.5), xytext=(0.1, 0.5), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    b.text(0.2, 0.33, "$K$", ha="center", fontsize=9)
    b.text(0.5, 0.9, "readout refresh: slow outer loop, every $K$ bins (CCIPCA $\\to$ Procrustes $\\to$ write $C$)",
           ha="center", fontsize=8, color=PALETTE["teal"])

    # --- (3) micro: per-bin loop + EMA memory tau ---
    c = axes[2]
    c.add_patch(plt.Rectangle((0, 0.45), 1, 0.18, fc="#eef1f4", ec=INK, lw=0.8))
    bins = np.linspace(0.05, 0.95, 19)
    x_now = bins[12]
    for xb in bins:                                   # per-bin steps; the current bin highlighted
        cur = abs(xb - x_now) < 1e-6
        c.add_patch(FancyArrowPatch((xb, 0.63), (xb, 0.71), arrowstyle="-|>",
                    mutation_scale=7 if cur else 5, color=INK if cur else PALETTE["blue"], lw=1.6 if cur else 0.9))
    c.text(x_now, 0.80, "now (bin $t$)", ha="center", fontsize=7.5, color=INK)
    tau_w = 0.09                                       # EMA weight decays into the PAST, peaks at now
    xs = np.linspace(x_now - 0.30, x_now, 120)
    c.plot(xs, 0.50 + 0.10 * np.exp(-(x_now - xs) / tau_w), color=PALETTE["mauve"], lw=1.6)
    c.annotate("", xy=(x_now, 0.30), xytext=(x_now - 0.12, 0.30), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    c.text(x_now - 0.06, 0.15, r"$\tau$ (EMA memory:" "\n" r"past $\sim\tau$ bins)", ha="center", fontsize=7.5, color=PALETTE["mauve"])
    c.text(0.5, 0.93, "fast inner loop: filter + dynamics every bin; feature EMA memory $\\sim\\tau$ bins",
           ha="center", fontsize=8, color=PALETTE["blue"])

    # zoom connectors macro->meso (around mid) and meso->micro (around a refresh)
    for (ax_top, x0, x1), ax_bot in [((axes[0], 0.45, 0.6), axes[1]), ((axes[1], 0.5, 0.62), axes[2])]:
        for xt in (x0, x1):
            con = ConnectionPatch(xyA=(xt, 0.35 if ax_top is axes[0] else 0.4), coordsA=ax_top.transData,
                                  xyB=(0 if xt == x0 else 1, 0.62 if ax_bot is axes[1] else 0.63),
                                  coordsB=ax_bot.transData, color="0.7", lw=0.6, ls=":")
            fig.add_artist(con)
    fig.savefig(os.path.join(FIGS, "timescales.pdf")); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_motivation, fig_flow, fig_timescales, fig_raster, fig_readout_stability,
               fig_readout_snr, fig_kstep_curves, fig_tau, fig_timing, fig_summary, fig_curves):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__}: {e}")
    print("wrote:", ", ".join(sorted(p for p in os.listdir(FIGS) if p.endswith(".pdf"))))
