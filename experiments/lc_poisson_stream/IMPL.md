# VJF online learning of a Poisson limit-cycle stream

Tests how quickly VJF learns a 2-D limit cycle from a long single-trial,
biologically-realistic log-linear Poisson spike stream, across 3 SNR levels,
**with an oracle (true) readout** that sets the performance upper bound.

## Design

- **Latent dynamics**: 2-D limit cycle (radius_scale=2, omega=3 rad/s), 5 ms
  Euler step. One shared trajectory across all conditions.
- **Stream**: single trial, `T = 200000` bins = **1000 s at 5 ms bins**.
- **Observations**: log-linear Poisson `y ~ Poisson(exp(C z + b))`, calibrated
  to a target Fisher-information SNR by `neurofisherSNR`. Biological rates:
  mean ~20 Hz (0.1/bin), peak <=~100 Hz (0.5/bin). Counts are streamed in chunks
  (never materialized as a full T x N array).
- **SNR axis = population size** (the natural lever at fixed biological rate):
  `{50, 150, 250}` neurons -> realized SNR ~ **3 / 6 / 8 dB**. Capped at 250:
  under oracle readout VJF's online filter is numerically stable up to ~300 and
  diverges around 800 (recognition latent blow-up poisons the dynamics term).
- **Oracle readout**: the decoder is pinned to the generator's true `(C, b)` and
  frozen. This anchors the latent frame and isolates whether VJF learns the
  **dynamics** given a correct readout -- the upper bound. (Learned-readout
  online single-pass collapses to R^2~0, matching the meta-SSM Poisson LESSONS.)
- **Training**: single online pass. Warm-up (dynamics frozen) for the first 15%
  (capped at 20000 steps) while the recognition settles, then
  `transition.initialize` and dynamics learning on for the rest.

## Diagnostics (results/)

- `learning_curves.png` -- aligned latent R^2 and ELBO components vs stream pos.
- `phase_portraits.png` -- true orbit vs aligned VJF latent (late window).
- `vector_field_evolution.png` -- learned velocity field at snapshot times
  (rows=conditions, cols=stream fractions) -- watch the rotation emerge.
- `kstep_prediction.png` -- k-step (1..100) free-run forecast R^2 and skill vs a
  zero-flow (constant) baseline, final model.
- `kstep_evolution.png` -- k-step forecast skill at each snapshot time.
- `forecast.png` -- long free-run rollout vs true orbit.
- `rate_reconstruction.png` -- predicted vs true rate (example neuron + scatter).
- `spike_raster.png` -- 3 s raster, neurons sorted by preferred phase.
- `convergence_summary.png` -- final R^2, steps-to-threshold, rate corr, k-horizon.
- `summary.json` -- metrics + full config + provenance (commits, versions, seed).

## Reproducibility

- Deterministic: latent (CPU `torch.Generator`) + calibration + Poisson sampling
  (`np.random.default_rng`) under master seed `20260602`.
- Pinned deps: `vjf` via `-e .` (commit recorded), `neurofisherSNR` pinned by
  commit in `requirements-exp.txt`; versions captured in `summary.json`.

## Run

```bash
# from the vjf repo root
uv venv .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r experiments/lc_poisson_stream/requirements-exp.txt
.venv/bin/python experiments/lc_poisson_stream/experiment.py            # full T=200000
.venv/bin/python experiments/lc_poisson_stream/experiment.py --quick    # tiny smoke test
.venv/bin/python experiments/lc_poisson_stream/experiment.py --t-eff 20000   # medium
```
