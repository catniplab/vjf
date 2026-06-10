"""Grow vs fixed RBF: PLL and forecast vs epochs (from the two single_dir summaries)."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "report_m1", "figs")
matplotlib.rcParams.update({"font.size": 9, "figure.dpi": 130, "savefig.bbox": "tight"})

fixed = json.load(open(os.path.join(RES, "single_dir_summary.json")))
grow = json.load(open(os.path.join(RES, "single_dir_summary_grow.json")))
ceil = fixed["pll_psth_ceiling"]
Es = sorted(int(k) for k in fixed["epochs"])


def series(d, key):
    return [d["epochs"][str(E)][key] for E in Es]


fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
ax[0].plot(Es, series(fixed, "pll"), "o-", label="fixed (100)")
ax[0].plot(Es, series(grow, "pll"), "s--", label="grow (->400)")
ax[0].axhline(ceil, ls=":", color="g", label="PSTH ceiling")
ax[0].axhline(0, ls=":", color="k", lw=0.6)
ax[0].set_xlabel("epochs"); ax[0].set_ylabel("leave-1-neuron-out PLL (bits/spk)")
ax[0].set_title("PLL: growth gives no gain"); ax[0].legend(fontsize=7); ax[0].set_ylim(0, 0.8)
ax[1].plot(Es, series(fixed, "forecast_r2"), "o-", label="fixed (100)")
ax[1].plot(Es, series(grow, "forecast_r2"), "s--", label="grow (->400)")
ax[1].axhline(0, ls=":", color="k", lw=0.6)
ax[1].set_xlabel("epochs"); ax[1].set_ylabel("free-run forecast R2")
ax[1].set_title("Forecast: growth hurts"); ax[1].legend(fontsize=7)
fig.suptitle("Single-direction (dir 225 deg, L=2): growing RBF vs fixed -- coverage helped, metrics did not")
fig.savefig(os.path.join(FIGS, "compare_grow.png"))
fig.savefig(os.path.join(FIGS, "compare_grow.pdf")); plt.close(fig)
print("PLL  fixed", [round(x, 3) for x in series(fixed, "pll")],
      " grow", [round(x, 3) for x in series(grow, "pll")], " ceiling", round(ceil, 3))
print("fcst fixed", [round(x, 3) for x in series(fixed, "forecast_r2")],
      " grow", [round(x, 3) for x in series(grow, "forecast_r2")])
print("fig ->", os.path.join(FIGS, "compare_grow.png"))
