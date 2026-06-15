# Editing Audit Report -- clean single-direction report (`report.tex`, c861305)

## 1. Executive Summary

Overall the report is clear, well-structured, and unusually honest; the C-C-C arc (reconstruct
at ceiling -> does it forecast? -> ceiling is real, not an artifact) is easy to follow. The
issues are mostly calibration/labeling and terminology harmonization, not structure.

**Top revision priorities**
1. **Val-vs-test labeling of the headline forecast numbers** (claim calibration / honesty).
   The reported skill (`S~+0.03`, vs-PSTH `-0.03..-0.04`, Fig 5B) is the *validation* number
   used for *selection*; the reserved *test* number is lower (`S_persist~+0.017`). Report the
   test-set evaluation of the selected config, or label the numbers as validation and give the
   test values alongside. As written it is mildly selection-optimistic.
2. **Harmonize the metric name.** "forecast-reconstruction criterion", "forecasted-reconstruction
   skill", "forecast skill", "forecasted reconstruction" all denote one thing. Pick one (suggest
   *forecasted-reconstruction skill*) and use it everywhere.
3. **Readout / loading / decoder drift.** Define once that the *readout* = the *loading* `(C,b)`,
   and that the *decoder* is where `(C,b)` is applied; then use consistently.
4. **Expand PSTH at first use** (abstract) and state the criterion's aggregation (how many start
   phases, averaged over which trials).
5. Minor proofreading (below).

No fabricated content, no overreach in the discussion. The limitations paragraph is present and
fair.

## 2. Major Issues

### Issue 1: Headline forecast numbers are the validation (selection) values, not test
**Location:** Abstract; Results "The autonomous free-run beats persistence but not the PSTH";
Fig 5B caption.
**Type:** Claim consistency / Evidence.
**Problem:** The Methods state selection is on validation and "the test trials are reserved for
the figures here," but the reported forecast skill (`S~+0.03`; vs-PSTH `-0.03..-0.04`; Fig 5B
bars) is the validation best-config record. The reserved-test skill of the same config is lower
(`S_persist~+0.017`).
**Why it matters:** Reporting the selection number as the held-out result is selection-optimistic
and contradicts the "test reserved for the figures here" sentence.
**Recommended action:** Evaluate the selected config on the test split (all start phases) and
report those numbers in the Results and Fig 5B; or explicitly label the numbers "(validation)"
and add the test values. Keep the PLL already test-based (0.67 vs ceiling 0.64).

### Issue 2: Metric name not harmonized
**Location:** Abstract, sec:criterion, Results headers/body, Fig 5/6 captions.
**Type:** Terminology.
**Problem:** Four phrasings for one metric (see Summary #2).
**Why it matters:** A reader cannot be sure the criterion, the score, and the figure axis are the
same quantity.
**Recommended action:** Define "forecasted-reconstruction skill $S$" once in sec:criterion; use
that term and `$S$`/`$\mathrm{skill}_k$` everywhere; reserve "criterion" for the selection rule.

### Issue 3: readout / loading / decoder synonyms
**Location:** Abstract ("unknown readout ... estimated ... by CCIPCA"); Method ("the loading
(C,b) is estimated"); Algorithm/Method ("anchored into the decoder").
**Type:** Terminology.
**Problem:** Three words for closely related objects without an explicit link.
**Recommended action:** One sentence: "the readout (the loading $(C,b)$) ... is anchored into the
decoder every $K$ steps." Then keep "readout/loading" for the estimate and "decoder" for use.

### Issue 4: Criterion under-specified for an external reader
**Location:** sec:criterion.
**Type:** Clarity / Missing detail.
**Problem:** "from many start phases $t_0$" -- how many, spaced how, and averaged over which
trials (the 10 test? aggregated how)? The search breadth ("a search of latent dimension ...") is
not quantified (how many configurations).
**Recommended action:** Add one clause: start phases every half-cycle from end-of-cycle-1 to the
last that admits 2 cycles, deviances summed over starts $\times$ held-out trials; and state the
grid size (e.g., "~150 configurations across three search rounds").

### Issue 5: "improves generalization" has no number or figure
**Location:** Results, curvature-penalty paragraph ("The penalty also improves test-set
generalization").
**Type:** Evidence.
**Problem:** Qualitative claim with no value or reference; the supporting comparison
(test `S_persist` +0.017 regularized vs +0.003 unregularized) is not shown.
**Recommended action:** Give the two numbers, or soften to "modestly improves," or add to Fig 5.

## 3. Terminology and Notation Audit

| Term / symbol | Where issue occurs | Problem | Recommended action |
|---|---|---|---|
| PSTH | abstract, captions, criterion | used before expansion | expand "peri-stimulus time histogram" at first use (abstract) |
| forecasted-reconstruction skill / forecast skill / criterion | throughout | 4 phrasings, 1 concept | unify; reserve "criterion" for the selection rule |
| readout / loading / decoder | abstract, method, alg | synonym drift | link once: readout = loading $(C,b)$; decoder = where applied |
| $S$ vs $\mathrm{skill}_k$ | criterion, results | $S$ is the weighted score; used loosely | always write $S$ for the weighted score, $\mathrm{skill}_k$ per-horizon |
| "filtered" vs "free-run" | results | central contrast | OK, but state once "with observations (filtered) vs without (free-run)" |
| $\tilde g$, $\nu_t$, $c$ (offset) | readout para | $c$ offset vs $c_j$ RBF center collide | rename the log offset (e.g. $\epsilon$) to avoid clash with RBF center $c_j$ |
| val vs test | methods vs results | which split each number is from | label every reported number's split |

## 4. Section-by-Section Notes

- **Abstract.** Purpose clear; slightly long (one dense paragraph). Could shed one clause
  (the SNR/array detail is not abstract-worthy). Harmonize the metric name; expand PSTH; ensure
  the headline numbers match whatever split Results report.
- **Introduction.** Tight and well-motivated. "deliberately strict question" is good framing. No
  changes beyond ensuring the criterion name matches.
- **Dataset and task.** Clear. The "fixed preprocessing step" framing of neuron selection is the
  right honest move. OK.
- **Method.** Dense but ordered. The CCIPCA scale-fix paragraph + Fig 4 is a method justification
  on *synthetic* data -- fine, but add half a sentence saying why it is shown (it is the
  condition under which the V1 latent does not inflate). Resolve the $c$ symbol clash (Issue/table).
- **sec:criterion.** Add the aggregation/grid detail (Issue 4). Otherwise strong.
- **Results.** Good topic sentences. Fix val/test (Issue 1) and the generalization number
  (Issue 5). The "ceiling not under-fitting" argument is well made.
- **Discussion.** Fair and calibrated. The "near-oracle PSTH" interpretation is the key insight
  and is stated well. Limitations present.

## 5. Local Clarity and Style Issues

- **Location:** Abstract, "an 8-fold drop in trajectory roughness."
  **Issue:** "trajectory roughness" undefined at this point.
  **Suggestion:** "(mean second difference of the free-run)" in parentheses, or defer to Results.
- **Location:** Results, "the largest models score near zero."
  **Issue:** "near zero" -- vs which baseline?
  **Suggestion:** "score near zero vs persistence."
- **Location:** Method, "candid covariance-free incremental PCA (CCIPCA)."
  **Issue:** acronym fine; ensure expansion appears once (it does).
- **Location:** Results, real-time paragraph, "about 30x inside the 10 ms bin."
  **Issue:** clear; consider "30x faster than the 10 ms bin budget."

## 6. Proofreading and Formatting List

- Expand "PSTH" at first use (abstract).
- RBF log-offset symbol `c` clashes with RBF center `c_j`; rename one.
- Ensure all `\cref`/`\ref` resolve (build shows no undefined refs -- OK; re-verify after edits).
- Consistent hyphenation: "free-run" (noun/adj) vs "free run" -- pick one (suggest "free-run").
- "leave-one-neuron" vs "leave-one-neuron-out" -- pick one.
- Tense: keep results in present tense (mostly consistent).
- Figure order: Figs 5-6 float to the end (after Discussion). Acceptable, but consider `[t]`/
  placement so they sit nearer their first reference in Results.

## 7. Priority Revision Plan

1. **Fix the val/test claim** -- report the selected config on the reserved test split (or label
   validation + add test). Highest priority (honesty).
2. **Harmonize the metric name** to "forecasted-reconstruction skill $S$" everywhere.
3. **Link readout = loading $(C,b)$; decoder = where applied**, once.
4. **Add criterion aggregation + search-size detail**; expand PSTH; give the generalization number.
5. **Rename the RBF log-offset symbol** to remove the $c$/$c_j$ clash; minor hyphenation/wording.
6. Optional: tighten the abstract by one clause; nudge Fig 5-6 placement nearer their references.

---

## 8. Resolution log (revision pass, 2 codex rounds + academic_editor)

**Fixed:**
- Headline numbers now reported on the reserved TEST split (test_eval.py): filtered PLL 0.68
  (ceiling 0.64); forecast S~+0.015, vs-PSTH -0.03..-0.04. Fig 5B regenerated on test.
- Corrected an error the test eval exposed: the curvature penalty does NOT improve test forecast
  skill (apples-to-apples L=4: lambda=0 +0.019 vs lambda=1e-3 +0.015, within noise). Dropped the
  "improves generalization" claim; reframed the penalty's value as a smooth, interpretable free-run.
- "smooth limit cycle" -> "smooth free-run/trajectory" everywhere (the flow contracts, not a
  sustained cycle); paragraph header too.
- Objective-sign convention (ELBO - lambda R ascended); "single-pass" -> replay epochs;
  Algorithm feature buffer G vs latent buffer X separated; deviance count/bin + axes + n=0;
  readout=loading=(C,b)/decoder mapping; log-offset c -> epsilon; defined m, K, theta_grow; dropped
  unused tau; removed the W (RBF weights vs warm-up trials) clash.
- Softened "ceiling, not under-fitting" -> "within this search, capacity is not the limiting
  factor"; added the E-sweep evidence to Table 1 caption; softened the generality claim.
- Removed process/history leakage ("bug/fix", "we then", "Inspecting", "we study separately",
  "additions over the original loop"); regenerated scale_fix.png legend (dropped "bug"/"fix").
- Fig 5A caption: skill "flat across lambda~1e-3..1e-2" (not a peak at 1e-3); noted lambda=1e-3 is
  the validation pick.
- Metric name harmonized to "forecasted-reconstruction skill".

**Intentionally deferred / accepted:**
- lambda is still used for the Poisson rate (lambda_t), CCIPCA eigenvalue (lambda^g_i), and the
  curvature weight (bare lambda); disambiguated by subscript + an explicit parenthetical. Kept
  (the figure axis + code use lambda for the weight).
- forecast_3d.png title shows PLL 0.67 (the video run, a separate training) vs the report's 0.68
  (test_eval aggregate) -- a rounding-level diff between two trainings; the figure is labeled "a
  frame from the rotating video" (representative run).
- Fig 5-6 float to the end (LaTeX); acceptable.
