"""Smoke test for the real-time tutorial script (script/realtime_tutorial.py).

Loads the script as a module and runs a short pass with plotting off, asserting the
online pipeline runs end-to-end, stays stable, and recovers the latent.
"""
import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "realtime_tutorial.py"


def _load():
    spec = importlib.util.spec_from_file_location("realtime_tutorial", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tutorial_runs_and_recovers():
    mod = _load()
    n_neurons, t_eff = 40, 2500
    out = mod.run(t_eff=t_eff, n_neurons=n_neurons, warmup_steps=500, refresh_K=200,
                  align_window=800, log_every=100, seed=20260605, plot=False)
    # exercises the claimed path: projection encoder, warm-start excluded from the live stream,
    # online readout refreshed via Procrustes.
    assert out["encoder"] == "projection"
    assert out["stream_start"] == min(10 * n_neurons, t_eff // 2)   # warm-start window excluded
    assert out["n_refresh"] > 0                                     # online (C,b) refresh fired
    assert out["n_diverge"] == 0
    assert np.isfinite(out["per_bin_ms"]) and out["per_bin_ms"] > 0
    assert len(out["r2_vals"]) >= 1 and np.isfinite(out["r2_vals"]).all()
    assert out["final_r2"] > 0.5            # online recovery of an unknown readout
