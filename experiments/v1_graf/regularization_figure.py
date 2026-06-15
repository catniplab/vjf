"""Regularization + forecast-ceiling figure for the single-direction report.

Panel A: the curvature-penalty bracket (lambda_bracket.py, dir 225, L=4, E=30) -- as
lambda rises, the velocity-field curvature and the free-run jaggedness collapse while the
forecast skill holds/peaks, then large lambda over-regularizes. Shows the penalty smooths
the one-step map.
Panel B: the regularized best config's forecast skill vs horizon (half/one/two cycles)
against BOTH baselines -- it beats persistence but loses to the stimulus-locked PSTH at
every horizon (the ceiling that capacity and smoothing do not close).
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.v1_graf.figstyle import set_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
# Panel B uses the reserved-TEST evaluation of the selected config (test_eval.py), NOT the
# validation/selection record -- the held-out numbers reported in the paper.
TESTEVAL = os.path.join(HERE, "report_m1", "videos", "test_eval.json")

# lambda_bracket.py result (dir 225, L=4, 1600 ctr, E=30, denoising off).
LAM = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
FIELD_R2 = [112.09, 76.44, 10.18, 1.40, 0.46, 1151.88, 1831.99]   # mean field curvature
JAG = [0.0978, 0.0396, 0.0176, 0.0119, 0.0127, 0.9371, 0.2158]    # free-run 2nd-diff
SKILL = [0.0093, 0.0143, 0.0166, 0.0183, 0.0021, -2.1854, -0.5977]  # S_persist (test)


def main():
    set_style()
    os.makedirs(FIGS, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    # Panel A: bracket (stable regime lambda <= 1e-1; lambda>=1 destabilizes the SGD).
    keep = [i for i, l in enumerate(LAM) if l <= 0.1]
    x = [max(l, 3e-5) for l in (LAM[i] for i in keep)]            # 0 plotted at the left edge
    axA.plot(x, [FIELD_R2[i] for i in keep], "o-", color="C3", label="field curvature $R_2$")
    axA.plot(x, [JAG[i] * 1e3 for i in keep], "s-", color="C0", label=r"free-run jaggedness ($\times10^{3}$)")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel(r"curvature penalty $\lambda$ (0 at left)")
    axA.set_ylabel("field curvature / jaggedness")
    axA.legend(fontsize=7, loc="upper right")
    axt = axA.twinx()
    axt.plot(x, [SKILL[i] for i in keep], "^--", color="C2", label="forecast skill vs persist")
    axt.axhline(0, ls=":", color="0.5", lw=0.7)
    axt.set_ylabel("forecast skill vs persistence", color="C2")
    axt.tick_params(axis="y", labelcolor="C2")
    axt.legend(fontsize=7, loc="lower right")
    axA.set_title("Curvature penalty smooths the flow (skill flat for $\\lambda\\!\\approx\\!10^{-3}$--$10^{-2}$)")

    # Panel B: selected config on the reserved TEST set, skill vs horizon, both baselines.
    te = json.load(open(os.path.abspath(TESTEVAL)))["0.001"]
    ks = [8, 16, 32]
    vp = [te["skill_persist"][str(k)] for k in ks]
    vq = [te["skill_psth"][str(k)] for k in ks]
    xb = np.arange(len(ks))
    axB.bar(xb - 0.18, vp, 0.36, color="C2", label="vs persistence")
    axB.bar(xb + 0.18, vq, 0.36, color="C3", label="vs PSTH (near-oracle)")
    axB.axhline(0, color="k", lw=0.8)
    axB.set_xticks(xb); axB.set_xticklabels(["8 (½ cyc)", "16 (1 cyc)", "32 (2 cyc)"])
    axB.set_xlabel("forecast horizon $k$ (bins)")
    axB.set_ylabel("forecasted-reconstruction skill")
    axB.legend(fontsize=7)
    axB.set_title(f"Selected model on test (L=4, $\\lambda$=$10^{{-3}}$): "
                  f"beats persistence, loses to PSTH")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"regularization.{ext}"), dpi=150)
    plt.close(fig)
    print(f"saved -> {FIGS}/regularization.png")
    print(f"  selected on TEST: PLL {te['test_pll']:.3f} (ceil {te['pll_ceiling']:.3f}) "
          f"S_persist {te['S_persist']:+.4f}")
    print(f"  vs persist k8/16/32: {vp[0]:+.3f}/{vp[1]:+.3f}/{vp[2]:+.3f}")
    print(f"  vs psth    k8/16/32: {vq[0]:+.3f}/{vq[1]:+.3f}/{vq[2]:+.3f}")


if __name__ == "__main__":
    main()
