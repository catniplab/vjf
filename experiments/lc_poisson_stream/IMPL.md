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

## Dynamics-estimator notes & TODO

VJF's default flow learner is **RLS** (`RBFDS(bayes=True)`). Over very long
streams (T > ~80-100k) its precision matrix grows unbounded (`shrink=1`) and
becomes ill-conditioned; combined with **non-identifiability** of RBF weights in
regions the trajectory never visits (a null-space / free-direction problem, not
a latent-scale invariance -- the oracle decoder pins the latent frame), the
weights drift and blow up (wmax ~1e8-1e17), killing the forecast while the
encoder/filter stays fine (R^2 ~0.84). Forgetting + ridge stops the *crash* but
not the weight explosion.

**SOLUTION (used here): square-root RLS** (`flow_learner='srrls'`). Potter
rank-1 square-root RLS propagates the weight-covariance Cholesky factor directly
(PD by construction, no precision accumulation/inversion), so it is numerically
stable over 200k+ steps AND keeps RLS-speed convergence. Unexcited RBF directions
keep their prior (weights stay at init), so the non-identifiability no longer
causes drift. References: Potter square-root filter / Bierman, *Factorization
Methods for Discrete Sequential Estimation* (1977); Haykin, *Adaptive Filter
Theory*; Sayed, *Adaptive Filters*.

Two other `flow_learner` options remain available:
- `'rls'` -- original; fast but precision blows up past T~100k (do not use long).
- `'sgd'` -- nn.Parameter trained by the dynamics ELBO (gradient-clipped); stable
  but converges too slowly to forecast well (kHor 1-3 at T=200k).

**Remaining TODO (future, lower priority).** Directional forgetting (Kulhavy) for
non-stationary dynamics; data-driven RBF placement (k-means on visited states)
vs the uniform grid; sweep n_rbf > 100 / narrower widths for an even sharper field.

## Results

Full 1000 s (T=200000, 5 ms bins) GCP runs, oracle readout, stable (diverge=0):

| flow (commit) | n_rbf | 50n ~3dB | 150n ~6dB | 250n ~8dB |
|---|---|---|---|---|
| sgd (3f27ab0)   |  50 | R^2 0.73, kHor 1  | R^2 0.84, kHor 2   | R^2 0.88, kHor 3   |
| **srrls (e317bdb)** | 100 | R^2 0.73, kHor 32 | R^2 0.86, kHor 100 | R^2 0.91, kHor 100 |

srrls recovers the latent (R^2 rising with SNR), learns a rotational velocity
field, and forecasts the full k=100 horizon at mid/high SNR -- the 50n case is
SNR-limited (kHor 32), not estimator-limited.
