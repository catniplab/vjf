# Plan: online refinement of the readout subspace (beyond frozen PCA)

## Problem

Causal PCA warm-start + **freeze** recovers the oracle bound at mid/high SNR but
(a) caps performance at the first-10*N-bin estimate and (b) is poor at low SNR
(0.61 vs 0.69) where that initial estimate is noisy. We want to **keep refining
the 2-D readout subspace `C` online** from the growing stream — without the two
failure modes we already hit:

1. **NLL-gradient collapse**: learning `C` by SGD on the Poisson reconstruction
   likelihood drives it to the trivial "mean-rate via bias" solution (rate-scaled
   `dNLL/dC` is tiny; trivial basin).
2. **Latent identifiability (rotation/scaling) drift**: the `C`-`x` factorization
   is identifiable only up to an invertible linear transform; a free, co-trained
   `C` rotates/scales the latent coordinate frame, invalidating the dynamics (flow
   learned in the old frame) and the recognition. Unfreezing PCA collapsed for
   this reason.

The clean offline answer is EM (E: filter with current `C`; M: refit `C`; iterate).
VJF is **single-pass online**, so we need an online (recursive) approximation.

## Core idea

Make the readout an **online M-step**, not a free SGD parameter:

- **Regression, not likelihood gradient.** Fit `eta_t = C x_t + b` by recursive
  least squares of a Gaussian surrogate of the counts onto the *filtered latent*
  `x_t` (the recognition's posterior mean). Surrogate target: `g(y_t) =
  log(y_t + c)` or Anscombe `2*sqrt(y_t + 3/8)` (variance-stabilizing). This is
  the same second-moment principle as PCA/PLDS init, but supervised by `x_t` and
  solved as a well-conditioned regression — no rate-scaled gradient, no trivial
  basin.
- **`C` slaved to the recognition frame.** Because `C` is a deterministic
  function (RLS solution) of the recognition's latents, it carries **no
  independent rotation/scaling ambiguity** — it just tracks whatever frame the recognition
  produces. This removes one whole degree of the drift problem (the failure mode
  that killed naive fine-tuning).
- **Filtered latents > raw spikes.** PCA uses the instantaneous spike covariance;
  the filter adds the dynamics prior, denoising `x_t`. Regressing onto `x_t`
  should beat unsupervised online PCA on `g(y_t)` — this is what "expands the
  PCA" (it keeps integrating temporal/dynamical structure), and should most help
  at low SNR.

Mechanically this is **online (stochastic) EM** with the E-step = VJF filtering
and the M-step = recursive readout regression, run in a single causal pass.

## Recursive M-step (reuse the square-root RLS we built)

Maintain running, forgetting-weighted sufficient statistics across bins:
- `S_xx` (2x2), `S_x1` (2x1), `S_1` (scalar)   — latent second moments,
- `S_gx` (N x 2), `S_g1` (N x 1)               — cross moments with `g(y_t)`.

Then the ridge-regularized M-step solution is
`[C | b] = S_gz (S_zz + lambda I)^{-1}` with `z = [x; 1]`. Update `C` every K bins
(slow timescale) from the current stats, or run a per-bin square-root RLS
(`flow_learner='srrls'` machinery already gives a PD, stable, RLS-speed update —
apply the identical Potter update with `x_t` as the regressor and `g(y_t)` as the
multi-output target). Cost is cheap: a shared 2x2 covariance + N x 2 cross term,
O(N) per bin — same order as the decoder forward.

Anchor / regularize toward the PCA init: shrink `C` toward `C_pca` (ridge in the
M-step, or a prior pseudo-count in the sufficient statistics) so early/low-SNR
updates can't run away.

## Latent-frame identifiability & stability (the crux)

Even with `C` slaved, the recognition frame itself can slowly rotate/scale. Plan:

1. **Two-timescale**: fast filtering; slow `C` (and recognition) drift. Update
   `C` from stats on a slow cadence with strong shrinkage to `C_pca`.
2. **Anchor the latent frame**: keep the filtered latent whitened (unit covariance)
   via the running `S_xx`; pick a canonical rotation by Procrustes-aligning the
   updated `C` to the previous `C` each update, and apply the inverse linear map
   to the recognition output / dynamics state so the flow stays in one frame
   (don't let the frame jump between M-steps).
3. **Consistency pressure already present**: the dynamics term penalizes frame
   drift (the flow is learned in the current frame), which helps anchor the latent frame
   — but is not sufficient alone; (1)+(2) are the safeguards.
4. **Damp EM error amplification**: at low SNR the filtered latents are noisy and
   EM can amplify errors; conservative update rate + shrinkage-to-init guard this.

## Experiments to validate

Score everything against the **oracle upper bound** and the **frozen-PCA** baseline
(both already in hand), across the 3 SNR conditions, 1000 s:

- Does online-refined `C` **beat frozen-PCA toward the oracle, especially at low
  SNR** (close the 0.61 -> 0.69 gap)?
- **Subspace angle** between learned `C` and true `C` (principal angles) *over
  time* — does it shrink monotonically (refinement) or drift (frame-identifiability failure)?
- **Stability** over the full 1000 s (no collapse/drift; `n_diverge`=0).
- Forecast horizon, rate-reconstruction corr, per-bin compute (extra M-step cost).

Ablations:
- M-step RLS  vs  NLL-SGD (the collapse)  vs  frozen-PCA (the cap).
- regressor = filtered latent  vs  raw `g(y_t)` (online PCA) — test the "use
  filtered latents" claim.
- with / without frame-anchoring; with / without shrinkage-to-init.
- surrogate `g`: log vs Anscombe; update cadence K; forgetting `lambda`.

## Risks / open questions

- **Frame drift** destabilizing the dynamics is the main risk → (1)+(2) above; if
  it persists, fall back to "refine subspace, but re-anchor the flow by
  re-`initialize`-ing the srrls in the new frame on a window of recent latents".
- **EM feedback runaway** at low SNR → shrinkage + slow cadence; possibly only
  refine when a confidence/SNR proxy (e.g. trace of `S_xx` or rate level) exceeds
  a threshold.
- **Surrogate mismatch**: `log/Anscombe` Gaussianization is an approximation to
  Poisson; if it limits the ceiling, consider one online Newton (IRLS) step on
  the Poisson likelihood *using the RLS solution as the well-conditioned
  preconditioner* (a single damped step, not free SGD) — keeps stability while
  being likelihood-correct.
- Whether to refine `b` jointly (yes — it's part of the same regression) and
  whether to also let the recognition adapt or hold it.

## Phasing

1. Sufficient-statistics online M-step for `C,b` from filtered latents (Gaussian
   surrogate), slaved + shrinkage-to-PCA, **no** frame-anchoring yet — measure drift.
2. Add frame-anchoring (whiten + Procrustes-anchor) if drift appears; verify subspace
   angle shrinks and low-SNR gap closes.
3. Ablations (regressor source, surrogate, cadence, forgetting) + the optional
   single damped Poisson-Newton refinement.
4. Compare to oracle / frozen-PCA / NLL-SGD across SNR; write up.

(Reuses: `srrls` square-root RLS machinery; the experiment's readout option would
gain a `'pca+online'` mode. Keep it a `flow`-style selectable, backward-compatible.)

---

## Revision after adversarial critique (supersedes the optimism above)

A skeptical review found the central premise is wrong and the plan is likely
over-engineered. Key corrections:

- **"C slaved => no rotation/scaling ambiguity" is FALSE.** The recognition net is SGD-trained
  against the same C (shared optimizer), so the rotation/scaling ambiguity lives in the
  (C, recognition) pair; slaving C just makes it follow the recognition's drift.
  This is a closed EM feedback loop with no external anchor except the C_pca
  shrinkage and the dynamics term -- so it may simply stay pinned to C_pca (the
  same "no headroom" outcome we already found for fine-tuning), or slowly collapse.
- **Gaussian surrogate is degenerate on sparse counts.** log(y+c) on mostly-0/1
  bins is ~binary, c-dominated, and biased (Jensen + floor) *worst at low SNR* --
  the regime we want to fix. Must smooth before the surrogate; expect a ceiling.
- **Procrustes-to-previous-C is a random walk, not an anchor** (drift composes).
  Anchor to a FIXED reference (C_pca).
- **Pushing a linear transform through the RBF flow**: a pure rotation can be pushed through by
  rotating the centroids too (distances preserved); scale/general maps cannot,
  so re-initializing the srrls flow in the new frame is the safe primary
  mechanism (budget its forecast perturbation), not a fallback.
- **Subspace angle is blind to frame drift that degrades the flow** -- use
  forecast-horizon-over-time + srrls wmax/trP/n_diverge as the primary stability
  metrics. Single seed (n=1) is insufficient: use >=5-10 seeds, paired, report
  mean +/- SD.

### Decisive cheap experiments to run FIRST (before building anything)

1. **Oracle-x M-step probe.** Regress the (smoothed) spikes onto the TRUE latent
   z and refit (C,b); score. If this barely beats frozen-PCA, the low-SNR gap is
   **information-limited, not estimator-limited** -> STOP, no refinement helps.
2. **Expanding/sliding-window log-PCA baseline.** Recompute PCA on smoothed
   log-spikes over a growing/sliding window, Procrustes-anchored to C_pca, srrls
   re-initialized in the new frame on each update. Causal, library-backed
   (incremental PCA: Oja/CCIPCA/GROUSE), no recognition circularity, no
   surrogate-on-x bias, and it directly accumulates information -- the right cure
   if the gap is noise-limited. If this closes 0.61 -> ~0.69, the online-EM
   M-step is unnecessary.
3. **Longer initial PCA window** (50*N / 100*N vs 10*N) as a trivial control for
   "is the gap just initial-window noise?"

### Revised recommendation

Drop online-EM as the headline. Default to **expanding-window log-PCA refit at a
slow cadence, anchored to C_pca, with the flow re-initialized in the new frame**.
Only pursue the filtered-latent M-step if (1) says the gap is estimator-limited
AND the window-PCA baseline leaves real headroom -- and then with raw smoothed
g(y) for the *subspace* and filtered latents only for *frame alignment*, fixed-anchor
Procrustes, multi-seed, forecast-horizon stability metric.

Full critique: subagent review, 2026-06-04.

---

## OUTCOME (Phase-0 executed, 2026-06-04) — online-EM NOT pursued

Ran the two cheap decisive experiments first (no VJF / fast). Verdict: **the
fancy online-EM is unnecessary; a larger causal PCA window suffices.**

Max principal angle between estimated and true C (deg, mean over 5 seeds):

| condition | PCA @10N | PCA @30k | oracle-x @30k (true latent) |
|---|---|---|---|
| 50n ~3 dB  | 23.8 | 2.3 | 2.1 |
| 150n ~6 dB | 13.8 | 2.8 | 2.6 |
| 250n ~8 dB |  9.4 | 2.8 | 2.6 |

- **Estimator- or information-limited? Neither — initial-window-limited.** PCA at
  the 10*N init is high-variance (esp. low SNR); with more bins it converges to
  ~2-3 deg, i.e. essentially the true subspace.
- **oracle-x (regress on the TRUE latent) ties PCA-on-spikes** (2.1 vs 2.3 deg) ->
  the "filtered latents > raw spikes" premise is false here; the online-EM M-step
  would buy ~nothing.
- **End-to-end (T=40k, seed 20260604):** PCA@10N already ~oracle for 50n at this
  seed (R^2 0.69 vs oracle 0.70); PCA@60-100N is stably at/above oracle across SNR
  (50n 0.71, 150n 0.87). The earlier "0.61 at 50n" was **single-seed noise** at the
  tiny 10*N window (the critique's n=1 warning, confirmed).

**Action taken:** `pca_init_mult` 10 -> 60 (still causal, ~oracle and stable
across SNR; no online-EM, no extra machinery).

**If/when an online-from-the-start refinement is wanted** (so the system isn't
blind for the first 60*N bins), the principled version is **expanding/incremental
window PCA** on smoothed log-spikes (library: Oja/CCIPCA/GROUSE), Procrustes-
anchored to the first estimate, srrls flow re-initialized on C update — NOT the
filtered-latent online-EM (which Phase-0 showed has no headroom). This is the only
remaining item worth building, and only for the streaming-from-t0 use case.

---

## Online incremental-PCA refinement: empirical results (2026-06-04)

Validated the user's directive (cheap small init + slowly improve), in two parts.

**(a) Subspace tracking works.** Incremental PCA (CCIPCA) on *causally* (EMA)
smoothed log-spikes, started from a cheap 10*N batch-PCA init, converges the
subspace toward the batch asymptote (max principal angle to true C, deg):
50n 18.3->1.9, 150n 10.9->2.2, 250n 8.6->2.3 (asymptotes 1.8/2.2/2.3). O(N*2)/bin.

**(b) Feeding the improving C into VJF: directional but slow.** End-to-end R^2
(50n, seed 20260602 = the genuinely-bad-10N-init seed):
- T=40000: online refinement HURTS (0.39 < frozen 0.44) -- not enough recovery time.
- T=120000: online gentle (alpha=0.05, K=1000) **rises 0.35 -> 0.49 and beats frozen
  (0.36)**, still climbing; single-swap@40k ~0.38; oracle ~0.71.

**Conclusion.** The idea has real signal -- continuous *small* updates the
recognition can track > one big swap > frozen, and it improves over time. BUT it
is slow and stays well below oracle, because the recognition net + srrls flow must
**re-adapt by SGD to the moving latent frame and always lag**. Two-timescale
damping trades rate for stability. This is the genuine "online" hard core.

**To actually close it (next):** stop making SGD chase the frame -- **absorb** each
C-update's linear frame change in closed form:
- insert an explicit linear alignment map after the recognition output and update
  it in closed form on each C change (so the encoder net doesn't re-learn), and
- rotate the RBF flow centroids by the same map (pure rotation is exact; scale
  needs width rescaling) so the srrls flow is re-expressed, not re-learned.
Alternatives to compare: faster recognition lr (two-timescale) ; longer T (it was
still rising at 120k) ; refine-then-freeze once the CCIPCA estimate stabilizes.

---

## Absorb-the-frame-change: tried, insufficient (2026-06-04)

Prototyped the redesign: on each readout refresh, re-express in closed form by the
induced 2x2 transform T (recognition mean-head <- T W; rotate RBF centroids by T and
velocity weights by T^T; carry the posterior mean). Result (50n, seed 20260602,
T=120k): **absorb == online, no gain** (both late R^2 0.547; frozen 0.362; oracle 0.705).

**Diagnosis (important):** the readout improvement is an **N-dimensional subspace
tilt** (which neuron combinations C reads), NOT a 2x2 latent rotation. Procrustes
already removes the in-plane component, so T ~= I and the 2x2 absorb is a near-no-op.
The quantity that must change is the **encoder's N->2 readout** (the recognition's
spike-reading weights), which has no closed form -> only SGD re-learns it, and that
is the rate limiter. So "absorb the latent frame" targets the wrong DOF.

**Better lever (next):** refine the **encoder INPUT projection**, not the latent
frame. Feed the recognition a subspace projection of the (smoothed) spikes, e.g.
`x_in = C^T g(y)` (or pinv(C) g(y)), so the encoder operates in 2-D; when the online
PCA improves C, the projection improves and the encoder barely re-learns. This
couples encoder + decoder to the SAME online subspace estimate. Open: still needs
the flow re-expressed under the 2x2 in-plane part (the absorb machinery handles
that), and the recognition currently takes raw y -> a structural change.

Status: online incremental-PCA refinement is **directionally validated** (climbs
0.36 -> 0.55, beats frozen) but **does not reach oracle**; the encoder re-learning
its input subspace is the bottleneck, and the latent-frame absorb does not address it.

---

## SOLUTION FOUND: encoder-input projection (2026-06-04)

Feed the recognition the **subspace projection** of the (causally smoothed) spikes,
`x_in = pinv(C) (g(y) - b)` (2-D), instead of raw spikes; the decoder/likelihood
keep raw counts. The encoder's job becomes a trivial 2->2 map, and when the online
PCA improves `C` the projected input improves automatically -> the encoder barely
re-learns. Online PCA refines `C` (CCIPCA on causal log-spikes), decoder refreshed
periodically (Procrustes-anchored).

**Result (50n ~3 dB, the hard low-SNR case, seed 20260602, T=120k):**

| mode | early | mid | late |
|---|---|---|---|
| frozen_proj (bad 10N C) | 0.314 | 0.350 | 0.314 |
| **proj_online** (refined C) | 0.535 | 0.869 | **0.871** |
| proj_oracle (true C) | 0.575 | 0.865 | 0.874 |

- **Online refinement reaches the ceiling**: proj_online (0.871) ~= proj_oracle
  (0.874), climbing 0.54 -> 0.87. The "cheap start + slowly improve to asymptote,
  fully online" goal is achieved end-to-end.
- **The projection front-end is a strictly better encoder**: its oracle (0.87) far
  exceeds the raw-spike-recognition oracle (0.71) -- inverting N sparse spike
  channels is hard; a denoised 2-D projected input is easy, so the posterior is
  much better. Implication: use the projection front-end even when C is known.

Why this works where the latent-frame absorb failed: the bottleneck was the
encoder re-learning its N->2 readout under a changing subspace. The projection
makes that readout = pinv(C) (closed-form, slides with the online estimate), so
the deep encoder net only ever does a fixed 2->2 refinement. No SGD chase.

Next: confirm across SNR; then promote to a real `readout='pca_proj_online'` mode
in experiment.py (recognition takes the projected input; decoder/likelihood keep
counts; online CCIPCA + Procrustes-anchored decoder refresh).

### Confirmed across SNR (T=100k), and an implementation note

| condition | proj_online (late) | proj_oracle | raw-spike oracle |
|---|---|---|---|
| 50n ~3 dB  | 0.871 | 0.874 | 0.71 |
| 150n ~6 dB | 0.951 | 0.947 | 0.84 |
| 250n ~8 dB | 0.968 | 0.967 | 0.91 |

Online refinement reaches the oracle ceiling at every SNR, and the projection
front-end beats raw-spike recognition everywhere (0.87/0.95/0.97 vs 0.71/0.84/0.91).

**pinv vs least-squares:** the projection operator `pinv(C)` (2xN) is recomputed
ONLY on a decoder refresh (every K bins; C is piecewise-constant), then reused as a
single O(N) mat-vec per bin: `x_in = pinv(C) (g(y)-b)`. Do NOT re-solve lstsq(C,.)
per bin (re-factorizes an unchanged C). C^T C is 2x2 so the pinv is tiny/stable.

**Status: SOLVED.** Promote to `readout='pca_proj_online'` in experiment.py
(recognition input = projected spikes; decoder/likelihood = counts; online CCIPCA
+ Procrustes-anchored decoder refresh; pinv(C) recomputed per refresh, reused/bin).
Consider making the projection front-end the default even for known C (it strictly
beats raw-spike recognition).
