# sVJF on Graf V1 -- experiment design (follow-up paper)

Date: 2026-06-09. Status: design approved (brainstorming). Next: writing-plans -> M1 implementation.
Scope: a SEPARATE follow-up paper to the synthetic sVJF tech report.

## 0. Goal

Validate sVJF on real macaque V1 (Graf et al. 2011): recover single-trial latent dynamics
ONLINE, in a single streaming pass, with an UNKNOWN readout. Establish four claims:

- C1 (parity): match vLGP / GPFA / PLDS on leave-one-neuron-out predictive log-likelihood
  (PLL, bits/spike).
- C2 (dynamics): a learned AUTONOMOUS oscillator flow -- vector field + free-run forecast --
  a dynamical-systems object the GP/LDS baselines cannot produce.
- C3 (real-time): one streaming single-pass fit; per-bin inference latency under budget on real data.
- C4 (structure): the filtered latent reproduces the orientation x phase torus and supports
  single-trial orientation decoding.

Continuity with the synthetic study: that benchmark is a rotating limit cycle; the V1
drifting-grating latent is a rotational oscillation, so sVJF "discovering" a limit-cycle flow in
real V1 mirrors the synthetic story exactly.

## 1. Data

- Source: Graf, Kohn, Jazayeri, Movshon (2011), "Decoding the activity of neuronal populations in
  macaque primary visual cortex," Nat Neurosci 14(2):239-245. Anesthetized monkey, Utah array.
- Structure (vLGP's array-5): 72 equally-spaced drifting-grating directions x 50 trials; 148 single
  units; trial ~2.56 s (stimulus 150-1150 ms, off-stimulus 1400-2400 ms).
- SEVERAL datasets (arrays/sessions) are available (Dropbox link held by Memming; data stored
  locally under `experiments/v1_graf/data/`, gitignored -- large + access-controlled). The exact set
  and identifiers are enumerated by the loader at first run.
- Neuron selection: 63 well-tuned neurons (bimodal circular-Gaussian tuning, R^2 >= 0.75) for the
  vLGP head-to-head; also an all-148 run for the unknown-readout-payoff angle.
- Trials are independent stimulus repeats: the latent state is reset to the prior (mean = 0) at each
  trial onset (the "reset to zero").

## 2. Reference method (vLGP)

vLGP (Zhao & Park 2017, Neural Computation 29(5); arXiv:1604.03053): a GP smoothness prior on the
latent + point-process (exp-link) likelihood with per-neuron autoregressive spike-history filters;
NO dynamics model. Evaluated by leave-one-neuron-out PLL (bits/spike, normalized to a
homogeneous-Poisson baseline) + R^2; baselines GPFA, PLDS. It recovered an orientation x
temporal-phase torus and the population noise-correlation structure. sVJF's distinction: a learned
nonlinear DYNAMICAL flow, fit ONLINE in one pass with an UNKNOWN readout.

## 3. Model (sVJF)

- Latent dim L in {3, 4} (sweep). 3D = minimal torus embedding (matches vLGP's 3D view);
  4D = two clean 2-planes (phase oscillation x orientation).
- Autonomous shared RBF flow (no stimulus input; udim = 0): one flow learns the grating-induced
  oscillation as an autonomous limit cycle. Orientation is NOT in the flow or the initial state
  (reset to 0); it is injected each bin by the encoder from the spikes. The torus emerges from the
  set of FILTERED latent states across directions.
- Flow learner: srrls (square-root RLS).
- Readout: unknown -- projection encoder (recognition reads `C^+ (g~(y) - b)`) + two-timescale online
  readout (CCIPCA + Procrustes refresh).
- RBF placement: DATA-DRIVEN. Centers by k-means (or subsample) on the coverage-warm-up latent
  buffer; widths from local nearest-neighbor spacing (reuse `rbf_width_scale`). ~150-300 bumps on the
  visited torus (not the full box), so srrls stays well-conditioned and real-time in 3-4D. Box-init
  kept as an ablation. (Touches `RBFDS.initialize` -- placement/width only, NOT the ELBO/RLS math.)

## 4. Protocol

Single streaming pass over trials, with a one-time init at the coverage boundary.

- Phase 0 (coverage warm-up, dynamics OFF): stream a STRATIFIED set spanning ALL 72 directions
  (~1-3 trials/direction, gathered regardless of presentation order), encoder-only, per-trial
  q-reset, buffering filtered latent means + readout feature stats. At the boundary, do the one-time
  inits on the all-direction buffer: (a) batch-PCA warm-start the readout `(C, b)`; (b) k-means-seed
  the RBF centers over the full torus; then enable the flow.
  Rationale: you cannot tile a manifold -- or estimate a global readout -- you have not visited. An
  early single-direction window would leave the flow undefined off the seen arc and bias `C`.
- Phase 1 (online, dynamics ON): stream the remaining trials, per-trial q-reset, srrls flow learning,
  online readout refresh. Strictly causal and real-time.
- Centers are seeded once and held: re-seeding mid-stream would invalidate the srrls weights tied to
  them. If the manifold later expands, that is a noted limitation (sec 10).
- Train/test: ~80/20 trial split per direction. Learn on train trials (single pass as above); freeze;
  on test trials infer the posterior and evaluate.

## 5. Evaluation (no ground-truth latent on real data)

- C1: leave-one-neuron-out PLL (bits/spike vs homogeneous Poisson; vLGP Eq. 39) + R^2, vs
  vLGP/GPFA/PLDS and vanilla VJF.
- C2: vector field of the learned flow (projected to the oscillation plane); free-run k-step forecast
  horizon (relative half-skill horizon, as in the synthetic paper).
- C3: per-bin online latency (median/p95) at each bin size, same CPU setup as the synthetic timing;
  one-pass wall time.
- C4: torus extraction (trial-averaged filtered latents per direction, first 3 singular vectors --
  reproduce vLGP Fig 8); single-trial orientation decoding accuracy from the filtered latent.
- Diagnostics: divergence count, readout subspace drift, per-trial reliability.
- Identifiability: latent is identifiable up to an affine map; all latent-space comparisons use a
  best-fit affine alignment (as in the synthetic paper); torus/topology is alignment-invariant.

## 6. Baselines and ablations

- Baselines: vLGP (Yuan's implementation, offline), GPFA, PLDS, vanilla VJF (online; expected to
  collapse/diverge under sparse spikes + unknown readout -- itself a motivating result).
- sVJF ablations: raw-spike vs projection encoder; rls vs srrls; frozen-PCA vs online readout;
  data-driven vs box RBF placement.

## 7. Build units (-> implementation plan)

- U1 Graf V1 loader (`experiments/v1_graf/`): parse all datasets -> (trial, time, neuron) counts +
  direction labels; bin; neuron selection; per-dataset signal metric for strongest-first ordering.
  First step: download + inspect the actual file format.
- U2 Trial-aware `online_filter`: a trial-boundary signal that resets q (-> prior) and skips the
  cross-boundary one-step prediction, model persistent; coverage warm-up that spans conditions.
  Additive to `vjf/realtime.py`.
- U3 Data-driven RBF placement in `RBFDS.initialize` (k-means/subsample + local-spacing widths;
  backward-compatible default = current box init).
- U4 Eval module: leave-neuron-out PLL, orientation decoding, torus, forecast, timing.
- U5 Baseline wiring: vLGP/GPFA/PLDS + vanilla VJF + ablations.
- U6 Driver + figures: dataset ranking, sweeps (L, bin, n_rbf), staged data + plot module (reuse the
  `paper-figures` conventions).

## 8. Milestones (easy-first; each a go/no-go gate)

- M1 establish: strongest-signal dataset @ 10 ms -> full pipeline + all 4 claims. Make-or-break; if
  this fails we learn it cheaply before any sweep. (vLGP's array-5 is a strong known-good candidate
  and gives a direct same-data vLGP comparison, unless the signal scan flags a stronger one.)
- M2 stress bin size: same dataset @ 5 ms then 1 ms -> sparse-bin robustness + the vLGP-resolution
  PLL table.
- M3 replicate: remaining datasets (strongest -> weakest) @ 10 ms (and the winning bin) ->
  cross-session generalization.

Easy-first within each: 10 ms before finer bins; best dataset before the rest.

GCP budget: 24 h wall-clock total across all milestones (shard across up to 4 VMs, per the
synthetic study's pattern). The per-timestep online loop is CPU-bound -- a GPU does not help;
size VMs by core speed, not GPU. Run via the `/gcp_run` skill (git mode: the VM clones the pushed
`feat/v1-graf` branch).

## 9. Open parameters (defaults; revisit with data)

- Bin size: 10 ms primary (easiest -- more spikes/bin, fewer steps, ~20-30 bins/cycle, real-time);
  then 5 ms; then 1 ms (vLGP-matched, sparsest) for the apples-to-apples PLL table. Confirm the
  grating temporal frequency at load (target >= ~20 bins/cycle at 10 ms).
- Spike history: OUT initially (instantaneous `Cx+b`; PLL reported honestly even though vLGP carries
  an AR history term). Per-neuron AR history is a flagged STRETCH unit if the PLL gap demands it.
- Coverage warm-up: ~1-3 trials/direction (sweepable).
- Latent dim: {3, 4} sweep.
- Neurons: 63 well-tuned (head-to-head) + all-148 (payoff angle).

## 10. Risks / open questions

- RBF coverage in 3-4D: mitigated by data-driven placement AFTER full-direction coverage. Centers
  are fixed post-seed; if the visited manifold expands later, the flow is undefined there
  (re-seeding would invalidate the srrls weights). Noted limitation.
- Spike-history disadvantage vs vLGP in PLL (sVJF is instantaneous). If material, add the AR term.
- Stimulus-locking vs autonomous oscillator: a single shared oscillator assumes a common temporal
  frequency across directions -- verify the grating temporal frequency is fixed across directions at
  load.
- vanilla VJF on real data may diverge; this is itself the motivating (C2/C3) contrast.

## 11. Conventions

- Equation immutability: U2/U3 touch scaffolding/initialization, not `gaussian_loss` / RLS / the
  ELBO terms. Confirmed in planning with Memming.
- Seeds: never 42; default date-based (20260609) + per-condition offsets.
- Data and large artifacts gitignored; stage slimmed JSONs for figures.
- Progress tracked in `experiments/v1_graf/IMPL.md` (created at M1 start).
