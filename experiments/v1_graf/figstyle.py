"""Shared figure style for the v1_graf diagnostics.

Tufte data-ink (despined axes, no grid, layered emphasis: secondary data recedes,
primary dominates) + paper-figures scale-1.0 mechanics (one rcParams block so every
panel agrees on font/weights; constrained_layout, not bbox='tight'; embedded fonts).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Tech-report text width. MEASURE the real document with \showthe\textwidth before
# using FW() to place a single-panel figure in the paper at scale 1.0; this is a
# stand-in for the multi-panel diagnostics, which are not placed at a fixed fraction.
TEXTWIDTH_PT = 469.755
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27


def FW(frac: float) -> float:
    """Figure width (in) for an \\includegraphics[width=frac\\textwidth] at scale 1.0."""
    return frac * TEXTWIDTH_IN


RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,                  # embed Type-42 fonts
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,   # despine
    "axes.grid": False, "lines.linewidth": 1.3,
    "legend.frameon": False, "figure.constrained_layout.use": True,
    "savefig.dpi": 200,                                     # NOTE: no savefig.bbox='tight'
}


def set_style() -> None:
    plt.rcParams.update(RC)


DIR_CMAP = "hsv"                 # cyclic colormap for drifting-grating direction (0..360 deg)


def dir_color(deg):
    """Consistent circular color for a drifting-grating direction (degrees). Use this
    everywhere a figure encodes direction by color so the scheme is shared across figures."""
    return plt.get_cmap(DIR_CMAP)((np.asarray(deg, dtype=float) % 360.0) / 360.0)


def ema(x, tau: float) -> np.ndarray:
    """Causal exponential moving average, timescale ``tau`` samples (alpha = 1/tau)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    a = 1.0 / max(tau, 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = (1.0 - a) * out[i - 1] + a * x[i]
    return out
