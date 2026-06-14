# Clean two-section V1 sVJF report: experiment plan

Goal: replace the investigation-arc report with two self-contained experiments --
**(A) single-direction** and **(B) multi-direction** -- each presenting one clean,
best-hyperparameter result with detailed space + time, per-condition + trial-averaged
performance figures. Find the best config by a search first, then run the clean
experiments, then restructure the report. History (M1 NO-GO, RLS divergence, the collapse
arc) is dropped from the spine; the scale fix and denoising stay as methods.

## Success criterion (LOCKED): future forecasted reconstruction, gated by reconstruction
The gold-standard, sole selection metric is **future forecasted reconstruction**: free-run
the learned flow from a time $t_0$ in a held-out trial (no observations after $t_0$),
**decode to a predicted firing rate, and score it against the held-out FUTURE spike
counts** over the next half, one, and two grating cycles ($k{=}8,16,32$ bins; 16 bins/cycle).
- **Skill vs a baseline, no affine alignment.** $\mathrm{skill}_k = 1 - D_{\mathrm{model}}(k)/D_{\mathrm{base}}(k)$,
  where $D$ is the Poisson deviance of the forecast rate against the future spikes; baselines
  = persistence (hold the $t_0$ rate) and the stimulus-locked PSTH. (Scoring on the observed
  spikes in spike-space removes the latent-space affine-overfit that made the old
  $R^2$ gameable at short horizons / high $L$.)
- **Aggregate** over many start phases $t_0$, all held-out trials, and all directions
  (macro-average across directions for multi-dir).
- **Pre-declared weighted score** $S = 0.5\,\mathrm{skill}_8 + 0.3\,\mathrm{skill}_{16} +
  0.2\,\mathrm{skill}_{32}$, with hard floors on $\mathrm{skill}_8,\mathrm{skill}_{16}$.
- **Reconstruction gate.** A config is eligible only if its filtered (with-observations)
  leave-one-neuron PLL is decent (within a margin of the PSTH ceiling). *If reconstruction
  is bad, everything is bad -- forecasting is only meaningful once reconstruction is decent.*
- **Decode accuracy is reported but does NOT drive selection.**
- **Compute is not a constraint.** Search thoroughly (exponentially scaled RBF budgets,
  many epochs, multiple seeds, full direction set). The Phase-1 search is **parallelized
  across up to 8 simultaneous `/gcp_run` instances**; use a validation split for the search
  and reserve the test trials for the final clean experiments.

## Rationale (why these criteria)
- **Forecasting the spikes, not classification, is the goal.** We want a model that has
  internalized the neural *dynamics* -- a flow whose autonomous free run, decoded, predicts
  the *future spikes*. Orientation decoding measures whether the encoder *separates*
  conditions (a readout property); it can be high while the dynamics are wrong, so it is
  not the target.
- **Score in spike space, not latent space.** A per-segment affine-aligned $R^2$ on the
  model's own latent is gameable: with $L{+}1$ regressors fit to only $k$ points, a linear
  drift or even a random high-$L$ trajectory scores $\sim0.8$--$0.9$ at $k{=}8$ (codex
  check). Scoring the decoded rate against the held-out spikes by a Poisson-deviance skill
  vs persistence/PSTH removes that loophole entirely -- there is no per-segment fit to
  overfit, and the target is external.
- **Reconstruction gates forecasting.** A model that cannot reconstruct the held-out spikes
  *with* observations (filtered PLL) cannot be trusted to forecast them *without*; so a
  config must clear a reconstruction floor before its forecast counts.
- **Why half / one / two cycles.** The half-cycle ($k{=}8$, maximal displacement from the
  start) is the discriminating short-horizon test the velocity field must pass; one cycle
  shows the flow closes the loop (right period/phase); two cycles probe whether the free run
  stays on the orbit rather than drifting off.
- **Why compute is unconstrained.** Capacity (RBF count $\sim2^L$, scaled with the number
  of conditions) and training length plausibly gate forecasting; better to over-provision
  and search widely than to under-fit and miss the achievable forecast.

## 0. Lessons baked in (the recipe baseline)
- **Scale-fixed CCIPCA** readout (covariance warm-start) -- essential; pins the latent.
- **SGD/Adam flow** (square-root RLS diverges in free-run); **growing RBF basis** with
  **RAN residual** weight init (SGD cannot fill zero-init centers).
- **Fit-gated denoising noise** (decaying raised sinusoid) -- stabilizes the free-run.
- **Per-trial reset**; **randomized interleaved trial order** (FIX: the current driver is
  direction-blocked -- all of one direction, then the next -- which is bad for online SGD).
- **Capacity must scale.** First scan: RBF basis under-grew at 72 directions (n_basis~159);
  L=6 helped decode at 24 dir (7.4x vs 5.7x chance) but hurt PLL (0.70 vs 0.79), not run at 72.
  So search L and the RBF budget jointly, scaling #centers ~exponentially and epochs with size.

## 1. Required code changes before the search
1. **Randomize trial order** in `run_m1`: after the 1-trial/direction coverage warm-up,
   shuffle the remaining train trials (interleaving directions) with the run seed. For the
   single-direction driver, the replay order across epochs is already per-trial-reset;
   shuffle the per-epoch trial order too.
2. **Scale the RBF cap with L and conditions**: `max_rbf = rbf_base * 2**L`, optionally
   `* ceil(n_dir / n_dir_ref)` for multi-direction (capped by the coverage-window count);
   expose `rbf_base`. Smaller widths so growth fills the larger cap.
3. **Scale epochs**: expose `epochs` (single-dir) / make the multi-dir pass length a
   knob; the search sweeps it.
4. **Logging**: dump the arrays the figures need (Section 4) to a results `npz` per run.

## 2. Phase 1 -- hyperparameter search (GCP, up to 8 parallel instances)
Fixed: scale-fixed CCIPCA, Adam flow, growing basis + residual init, randomized order.
Selection = the locked criterion above (future-forecasted-reconstruction weighted skill,
gated by the reconstruction floor), computed on a **validation split** (test trials
reserved for the clean experiments). Search axes (coarse -> refine):

| axis | single-dir | multi-dir |
|---|---|---|
| latent_dim L | {3, 4} | {3, 4, 6} |
| RBF budget (rbf_base x 2^L) | {25,50,100}x2^L | {50,100,200}x2^L |
| epochs E (replay) | {20, 50, 100} | pass-length / replay {1, 2, 4} |
| lr (Adam) | {1e-4, 5e-4, 1e-3} | same |
| dyn_noise sigma0 / decay / fit_ref | {0,0.2,0.3} / {0.97,1.0} / {0,0.3} | same |
| **selection** | weighted forecasted-reconstruction skill (k=8/16/32) on validation, gated by filtered PLL; decode reported only | same, macro-averaged over n_dir |

- **Parallelization:** shard the grid across **up to 8 simultaneous `/gcp_run` VMs**
  (each VM runs a chunk; results merged). Single-dir search is cheap; run it first, lock
  its best.
- Multi-dir search: run across **all** n_dir in {8,24,72} and select by the macro-average
  (not a single slice), since compute is unconstrained.
- Keep the scale-fixed-srrls arm as a multi-dir **reference** (the decode point), not a
  candidate recipe.
- Output: `best_single.json`, `best_multi.json` (the chosen configs + their metrics).

## 3. Phase 2 -- clean experiments (best config), fully logged
**A. Single direction** (dir 225, 40 train / 10 test, best config). Log per test trial:
inferred latent path x_{1:T}; leave-one-neuron predicted rate lambda(t) per neuron; the
observed counts/PSTH; free-run forecasts from several start phases at early/mid/best
checkpoints; PLL, PSTH ceiling, forecast R2.

**B. Multi-direction** (n_dir in {8,24,72}, best L/config + srrls reference). Log per
direction: trial-averaged inferred latent path; predicted rate; single-trial latent
summary + decode; PLL; forecast. Plus the per-n_dir aggregates.

## 4. Figures (detailed: space + time, per-condition + averaged)
**Section A (single direction):**
- TIME: latent x(t) over a trial (10 trials faint + average bold); reconstruction
  lambda(t) vs observed PSTH for ~6 example neurons; forecast-over-time at early/mid/best.
- SPACE: latent x1-x2 phase plane (per-trial faint + average cycle, colored by phase);
  predicted-vs-observed rate (per neuron).
- METRICS: PLL vs ceiling; forecast R2.

**Section B (multi-direction):**
- SPACE: orientation torus -- per-direction trial-averaged latent in PC space, consistent
  circular (hsv) color; a few per-direction phase planes.
- TIME: per-direction average latent x(t) (circular color); reconstruction lambda(t) for an
  example neuron across directions.
- PERFORMANCE: decode/chance, PLL, forecast vs n_dir.

## 5. Phase 3 -- report restructure
Two sections (A, B) + tight Methods (the filtering ELBO; the CCIPCA scale fix; the
denoising equations; Algorithm 1; **the train/test split and multi-epoch online training**,
incl. randomized order) + a short Discussion. Drop the NO-GO / RLS / collapse narrative.

## 6. Flow
Write this plan -> **codex-review the plan** -> implement Section 1 code changes ->
**/gcp_run Phase 1 search** -> lock best configs -> Phase 2 clean runs -> figures ->
Phase 3 restructure. Push only at phase boundaries.
