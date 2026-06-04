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
