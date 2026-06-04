# VJF online learning of a Poisson limit-cycle stream — findings

How fast and how well can **VJF** (variational joint filtering) learn a 2-D
rotating limit cycle from a long, biologically-realistic Poisson spike stream,
online, one bin at a time — first with a known readout (upper bound), then with
the readout learned from data (the realistic hard case)?

Branch: `exp/lc-poisson-stream-20260602`. Code: `experiment.py`, `lc_data.py`.
Results archived under `gcp_runs/<vm>/results/` (one dir per GCP run).

---

## 1. Headline results

Full runs: T = 200000 bins = **1000 s at 5 ms bins**, single trial, omega = 30
rad/s (rotating limit cycle), 3 SNR levels via population size {50, 150, 250}
neurons (realized ~3/6/8 dB), dynamics flow = square-root RLS (`srrls`).
Latent scored by affine-aligned R^2 (latent identifiable up to affine).

| readout | 50n ~3 dB | 150n ~6 dB | 250n ~8 dB | meaning |
|---|---|---|---|---|
| learned (random init) | ~0.01 | ~0.01 | 0.008 | **Poisson collapse** (trivial mean-rate solution) |
| **oracle** (true C,b, frozen) | 0.69 | 0.84 | 0.91 | **upper bound** |
| **PCA warm-start + freeze** | 0.61 | 0.86 | 0.92 | **data-driven, no oracle** — matches bound at mid/high SNR, worse at low SNR |

- Forecast horizon (k-step free-run R^2 >= 0): **k=100** (= full tested range,
  ~2.4 cycles ahead) at mid/high SNR; SNR-limited (k~15-33) at 50n.
- Stable over the whole 1000 s (diverge = 0) with `srrls`.
- **Per-bin online-filtering compute: ~3.0-3.15 ms/bin** (50->250 neurons, ~flat
  in N) on a GCP `e2-standard-4` (Intel Xeon @ 2.20 GHz, 4 vCPU, 2 torch threads,
  CPU only). At 5 ms bins this is **real-time / closed-loop feasible**; headroom
  shrinks below ~2 ms bins.

**Bottom line:** VJF learns the rotating limit cycle from realistic Poisson
spikes *without knowing the observation model* (causal PCA warm-start), with
SNR-graded quality, stable over 1000 s, at real-time per-bin cost — once the
dynamics estimator and the readout init are done right (sections 3-4).

---

## 2. Experimental design

- **Dynamics**: 2-D limit cycle (radius relaxes to sqrt(radius_scale=2),
  constant omega), 5 ms Euler step. One shared latent trajectory across all
  conditions; only the observation population differs.
- **Latent** `z`: per-dim zero-mean unit-var (the scored ground truth).
- **Observations**: log-linear Poisson `y ~ Poisson(exp(C z + b))`, calibrated to
  a target Fisher-information SNR by `neurofisherSNR` (Jeon & Park, EUSIPCO 2024;
  pinned by commit in `requirements-exp.txt`).
- **Biological rates** (a hard constraint that drives the whole design): at 5 ms
  bins, mean ~20 Hz = **0.1 spk/bin**, peak ~100 Hz = **0.5 spk/bin**.
- **SNR axis = population size.** At fixed biological rate the per-bin
  (instantaneous) SNR is low and the natural, biologically-meaningful lever is
  the number of neurons (more neurons = more aggregate Fisher info). Rate cannot
  be the SNR knob (it's pinned by biology). Capped at ~250-300 neurons: that's
  the numerically stable ceiling for VJF's online filter at xdim=2 (see 3).
- **Streaming**: Poisson counts are generated in chunks on the fly and never
  materialized as a full T x N array (would be tens of GB at large N*T). The
  whole experiment regenerates from the seed; only `z`, `(C,b)`, and the plots/
  summary are saved/retrieved.
- **Training protocol**: single online pass. Warm-up (dynamics frozen) for the
  first 15% (capped at 20000 steps) while the recognition settles, then
  `transition.initialize` + dynamics learning on for the rest.

### Calibration gotchas (neurofisherSNR)

- `priority='max'` makes the final `C` per-neuron-downscaled to cap the max rate
  but matches the bias `b` against the *un*-downscaled C -> realized mean rate
  collapses (0.02 when you asked for 1.0). **Use `priority='mean'`** (re-matches
  the bias at the end), which holds mean rate to target.
- At biological rate, **mean and peak can't both be hit** for high SNR: high SNR
  needs high gain -> peaks blow to 500+ Hz. Keeping peak <=100 Hz pins you to
  ~0-8 dB. We accept the realized SNR and report it (target is nominal).
- The library's 2-D coherence optimizer (`initialize_C`) doesn't converge for
  d_latent=2 (in 2-D, max coherence over many vectors is 1); pre-seed a sparse
  Gaussian `C` and pass it in to skip that step.

### omega and aliasing (rotation signal)

At omega=3 (419 steps/cycle) the per-step rotation `omega*dt = 0.015` rad is tiny
vs the radial restoring force + noise, so VJF learns a **radial ring attractor**
with weak rotation. **omega=30** (42 steps/cycle, `omega*dt=0.15`) gives a strong
rotational signal and a clearly *rotating* limit-cycle field, with comfortable
aliasing margin (steps/cycle = `2*pi/(omega*dt)` = `1257/omega` at 5 ms; keep
well above a few/cycle, i.e. omega well under ~120).

---

## 3. Dynamics estimator: RLS -> square-root RLS (`flow_learner`)

VJF's flow is an RBF velocity field whose weights were learned online by RLS.
Findings, in order:

1. **The flow is under-parameterized and the RLS conditioning is fragile.** Only
   `n_rbf * xdim = 50*2 = 100` trainable weights on 50 *fixed* RBF bumps. The
   bump width is set to the full state radius regardless of count, so adding RBFs
   makes them collinear -> singular precision -> blow-up. **Narrowing the width
   (x0.5)** sharpens the field and ~doubles the forecast horizon (k 54 -> >=100).
2. **Plain RLS blows up over long streams.** With `shrink=1` (no forgetting) the
   precision matrix grows unbounded (`trP` -> ~1e6) and becomes ill-conditioned;
   combined with **non-identifiability** of RBF weights in regions the trajectory
   never visits (a null-space / free-direction problem — *not* a latent-scale
   invariance; the oracle decoder pins the latent frame), the weights drift and
   explode (`wmax ~1e8-1e17`) past T ~ 80-120k, killing the forecast while the
   encoder/filter stays fine (R^2 ~0.84). Exponential forgetting alone *collapses*
   the precision in unexcited directions (windup); ridge + forgetting stops the
   crash but the weights still explode; capping `trP` desyncs `w_mean` -> NaN.
3. **`flow_learner='sgd'`** (weights as an `nn.Parameter` trained by the dynamics
   ELBO, gradient-clipped) is *stable* over long streams (unexcited RBFs get ~0
   gradient, stay at init) but **converges too slowly** to forecast (kHor 1-3 at
   T=200k; more n_rbf doesn't help under SGD).
4. **`flow_learner='srrls'` (Potter square-root RLS) is the solution.** It
   propagates the weight-covariance Cholesky factor directly via rank-1 Potter
   updates (PD by construction, no precision accumulation/inversion), so it is
   numerically **stable over 200k+ steps** AND keeps **RLS-speed convergence**.
   At T=120k, 250n: `srrls` -> diverge=0, `wmax~5`, R^2=0.92, kHor=100 (vs `sgd`
   R^2=0.84 kHor=0; `rls` explodes). Handles n_rbf=200 fine. We reuse `w_chol`
   as the covariance sqrt so the Gaussian forward/sampling path is unchanged.
   References: Potter square-root filter; Bierman, *Factorization Methods for
   Discrete Sequential Estimation* (1977); Haykin, *Adaptive Filter Theory*.

**VJF API added (backward-compatible; default `'rls'` reproduces original):**
- `RBFDS(..., flow_learner='rls'|'srrls'|'sgd')`; `VJF.make_model(...,
  transition_flow=...)`.
- `LinearRegression.srls()` (Potter update) + `init_srls()` (lstsq warm-start of
  `w_mean`, `w_chol = sqrt(p0)*I`).
- `LinearRegression.rls()` gained optional `ridge` (ridge-regularized forgetting).
- `RBFDS.rls_shrink`, `RBFDS.rls_ridge` (RLS flow only).
- Also fixed: `torch.eig` (removed in torch 2.0) -> `torch.linalg.eigvalsh` in the
  RLS Cholesky-recovery path.

Tests `test_module.py` / `test_model.py` pass (7); `test_sgp.py` is a pre-existing
orphan (imports a non-existent `vjf.gp`).

---

## 4. Readout learning: the hard case (unknown C,b)

With the decoder **learned from random init**, VJF hits the meta-SSM
`LESSONS_LEARNED` Poisson wall: it collapses to the trivial "predict mean rate via
bias `b`" solution (R^2 ~ 0). Cause: sparse low-rate spikes -> `exp()` scales
`dNLL/dC` by the rate (~0.1/bin) -> the gradient on `C` is tiny, and there's a
trivial-solution basin. The Fisher SNR bounds *state*-SNR (inferring x given C),
not the *identifiability of C* (~10^4 params, under-determined per step).

**Fix — causal PCA warm-start of the readout, then freeze:**
- PLDS/GPFA-style: Gaussian-smooth the spikes in time, log, PCA across neurons ->
  `(C, b)`; rescale the latent to unit variance per dim (fold scale into C).
  References: Macke et al. NeurIPS 2011 (PLDS init), Yu et al. 2009 (GPFA).
- **Causal**: uses only the FIRST `~10*N` bins (online algorithm, no look-ahead).
  `N=50/150/250 -> 500/1500/2500` bins.
- **Freeze it.** Frozen PCA readout recovers the oracle bound (0.93 vs 0.91 @
  250n). **Fine-tuning hurts**: unfreezing at full lr collapses it (R^2 0.04);
  decoder lr 0.1x degrades (0.71); 0.01x is safe but identical to freeze (no
  gain). Why: a frozen "good-enough" readout + a flexible recognition net already
  reaches the bound (the encoder compensates), so there is *no headroom*, only
  the downside of the gauge (rotation/scale) freedom + low-rate gradient pulling
  it back to the trivial solution.
- **SNR-graded** (as expected): PCA matches oracle at 150/250n but is worse at
  50n (0.61 vs 0.69) — the smoothed-spike estimate of `(C,b)` is noisier when
  spikes are sparse.

Experiment knob: `cfg["readout"] = 'oracle' | 'learned' | 'pca'` (default `pca`),
with `pca_init_mult` (=10) and `pca_smooth_sigma` (=8 bins).

---

## 5. Numerical / engineering gotchas (this codebase)

- **O(T^2) OOM**: `model.transition(sampling=False)` computes `FL @ FL.T` (an NxN
  matrix) just to read its diagonal, so calling it on the full 200k-step array
  allocated ~160 GB and was OOM-killed. **Chunk** any transition call over the
  full stream (`transition_mean(..., chunk=2000)`).
- **dtype**: `np.linalg.lstsq` returns float64; feeding it back into VJF (float32)
  raises a cdist dtype error. Cast to float32.
- **Latent / NaN guards** (experiment-side, no VJF edits): floor `transition.
  logvar` (state-noise) to avoid `exp(-0.5*logvar)` overflow -> NaN in the
  dynamics `gaussian_loss`; mask non-finite predictions in steady-state metrics;
  the run records `n_diverge`.
- **Equation note**: `gaussian_loss` two-Gaussian variance term was a real bug
  (product-in-exponent vs sum of two traces); fixed after confirming with
  Memming. (Separate from this experiment; landed on master via PR #6.)

---

## 6. Reproducibility & how to run

- Deterministic: latent (CPU `torch.Generator`) + calibration + Poisson sampling
  (`np.random.default_rng`) under master seed `20260602`; data generated on CPU
  so it's identical regardless of training device.
- Pinned deps: `vjf` via `-e .` (commit recorded in `summary.json`),
  `neurofisherSNR` pinned by commit in `requirements-exp.txt`. Provenance in
  `summary.json`: vjf commit/branch, torch/numpy/scipy/neurofisherSNR versions,
  full config, seed, CPU model / cores / torch threads.

```bash
# from the vjf repo root
uv venv .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r experiments/lc_poisson_stream/requirements-exp.txt
.venv/bin/python experiments/lc_poisson_stream/experiment.py            # full T=200000
.venv/bin/python experiments/lc_poisson_stream/experiment.py --quick    # tiny smoke test
.venv/bin/python experiments/lc_poisson_stream/experiment.py --t-eff 20000   # medium
```

### Plots (results/)
`learning_curves`, `phase_portraits`, `vector_field_evolution`,
`kstep_prediction` (k-step R^2 + RMSE vs zero-flow baseline),
`kstep_evolution`, `forecast` (long free-run), `forecast_examples`
(single-trial free-runs vs truth), `rate_reconstruction`, `spike_raster`
(3 s, phase-sorted), `convergence_summary`, `summary.json`.

### Running on GCP
Use `/gcp_run` (git mode: VM clones the pushed branch). Per-VM helper:
`gcp_runs/vm.sh` (SA impersonation + zone baked in: `vm.sh ssh '<cmd>'`,
`vm.sh pull '<glob>' <dir>`, `vm.sh log`). Hard-won infra lessons are in the
`gcp_run` skill (SA impersonation for external-org OS Login; set project
globally; `gcloud ssh` exits 255 cosmetically on backgrounding/kill — verify
separately; `pkill -f experiment.py` self-kills the ssh shell — match
`venv/bin/python`; don't pipe the run through `grep`, it masks exit codes; a
per-timestep online loop is CPU-bound and a GPU doesn't help). Each VM was
`e2-standard-4`, ~30 min for the full 3-condition run (~3-4x slower per core than
an M-series Mac for this sequential loop).

---

## 7. Open TODO

- **Directional forgetting** (Kulhavy) for `srrls` to track *non-stationary*
  dynamics without windup (current `shrink=1` is correct for this stationary
  system).
- **Data-driven RBF placement** (k-means on visited states) vs the uniform grid;
  sweep `n_rbf` > 100 / narrower widths for an even sharper field.
- A learned-readout *baseline* full run (random init) to plot the collapse
  alongside oracle/pca (the collapse is established at R^2 ~ 0).
- Closing the low-SNR PCA gap (better smoothing / FA init at sparse rates).
