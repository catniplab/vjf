# Editing audit -- post-E6/E7 integration pass (2026-06-08)

Stage-3 pass with `academic_editor` (diagnosis) + `writing_style_detailed` (voice), after the
E6 motivation and E7 flow-learner figures were integrated and codex round-1 fixes applied.

## 1. Executive summary

The report is in good shape: clear problem -> gap -> contribution -> calibrated-verdict arc, the
author's architectural voice (dark-room analogy, italic crystallizing verdicts), and consistent
hyphenation (`filtered-latent`, `one-step`). The major consistency issues were already resolved in
codex round 1 (filtering vs dynamics vs readout-schedule; SNR-qualified subspace drift; separate-
experiment notes; tau/timing/units). This pass made the remaining small fixes below.

## 2. Issues found and fixed this pass

- **Notation (parameters).** Text/ELBO used `\theta` but Algorithm 1 used `\bm\Theta` for the same
  parameter set. Harmonized to `\theta` (pseudocode/text only; the model equations eq:lik/gen/elbo
  were untouched -- no math change).
- **Repetition.** "We report ... We also report ... We report mean ..." in the Experiments metrics
  paragraph -> smoothed.
- **Verified consistent:** sVJF macro; "readout"/"loading"/`C` usage; acronyms defined at first use
  (RBF, CCIPCA, EMA, RLS, PCA); figure/table cross-references resolve; 0 undefined refs; 15 pp.

## 3. Terminology / concept order -- OK

- VJF + its generative model and ELBO are introduced (Background) before the Method changes.
- `projection encoder`, `two-timescale readout`, `square-root RLS` named in the abstract/intro and
  defined in Method before use.
- `oracle` is explicitly scoped to the readout (not the dynamics) at first use.
- state-SNR vs dynamics-information distinction stated in Experiments and respected downstream.

## 4. Voice (writing_style_detailed) -- OK, light touch

Verdict sentences (abstract, Results, Discussion) are crystallizing and consistent; analogies
(dark room, "each spike is most precious") are content, not meta. Meta-commentary was removed in a
prior step. No further rewrites needed.

## 5. Remaining (deferred, for a copy-edit before submission, not this pass)

- The Experiments metrics paragraph is long; could split setup vs metrics if a journal prefers.
- "frozen PCA" / "frozen batch-PCA" / "frozen principal-component readout" are used interchangeably
  (spelled out at first use, abbreviated later) -- acceptable, but could be standardized.
