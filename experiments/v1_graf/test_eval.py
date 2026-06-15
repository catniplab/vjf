"""Honest TEST-set evaluation of the selected single-direction config, for the report.

Trains on the 30 train trials (same as the search) and evaluates the forecasted-reconstruction
skill on the RESERVED 10 TEST trials -- the search selected on validation, so the reported
headline numbers must come from test, not the (selection-biased) validation record. Runs the
regularized best (L=4, lambda=1e-3) and the lambda=0 control (for the generalization claim),
writes test skill per horizon vs both baselines + PLL to a JSON the report/figure consume.

Run: uv run python -m experiments.v1_graf.test_eval
"""
from __future__ import annotations
import json
import os

import numpy as np
import torch

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_m1", "videos", "test_eval.json")


def main():
    torch.set_default_dtype(torch.float32)
    data = prepare_single_dir_data(direction=225.0, n_val=10)        # 30 train / 10 val / 10 test
    test_ybar = float(data["test_counts"].mean())
    rows = {}
    for lam in [0.0, 1e-3]:
        res = _train_eval(
            data["train_trials"], data["test_trials"], data["test_counts"], test_ybar,
            epochs=100, latent_dim=4, N=data["N"], grow=True, grow_weight_init="residual",
            flow="sgd", optimizer="adam", lr=1e-3, rbf_base=100, max_rbf=1600,
            dyn_noise=0.0, smooth_lambda=lam, psth_counts=data["psth_counts"], return_model=False)
        sk = res["forecast_skill"]["skill"]
        rows[f"{lam:g}"] = {
            "smooth_lambda": lam, "test_pll": res["pll"],
            "pll_ceiling": data["pll_psth"], "n_basis": res["n_basis"],
            "S_persist": res["fc_weighted_persist_skill"],
            "skill_persist": {str(k): sk[k]["vs_persist"] for k in (8, 16, 32)},
            "skill_psth": {str(k): sk[k]["vs_psth"] for k in (8, 16, 32)},
        }
        r = rows[f"{lam:g}"]
        print(f"lambda={lam:g}: test PLL {r['test_pll']:.3f} (ceil {r['pll_ceiling']:.3f}) | "
              f"S_persist {r['S_persist']:+.4f} | "
              f"persist k8/16/32 {r['skill_persist']['8']:+.3f}/{r['skill_persist']['16']:+.3f}/{r['skill_persist']['32']:+.3f} | "
              f"psth {r['skill_psth']['8']:+.3f}/{r['skill_psth']['16']:+.3f}/{r['skill_psth']['32']:+.3f}", flush=True)
    with open(OUT, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
