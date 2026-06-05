# Editing Audit Report

Audit target: `/Users/memming/Dropbox/_projects/vjf/paper/main.tex` (full document, 300 lines).
Audit type: diagnostic (not a rewrite). Rubric: standard editing-audit + Scientific C-C-C (Standard A) + author narrative voice (Standard B).

---

## 1. Executive Summary

**Overall readability.** The paper is technically dense but well-organized at the section level, and it is admirably honest about its mixed results. The Method and Results sections largely follow a question -> evidence -> answer logic and the "Difference from VJF" markers are an effective recurring device. The central weakness is that the **narrative onramp (Abstract and Introduction) is overloaded with formal notation**, which violates both Standard A (naive-but-smart reader first; motivate in prose) and Standard B (no inline math in motivational passages; vivid analogy before abstraction). A second weakness is a **counting/structure inconsistency** ("two obstacles" / "two issues" vs. "We address all three" / "three coupled changes"). A third is **terminology drift** around the core contributions (the same three ideas are renamed across abstract, intro, and method headings), which undercuts the C-C-C requirement of one restated contribution.

**Top revision priorities.**

1. **De-mathematize the Introduction and Abstract.** Remove inline symbols ($\vC$, $\vb$, $\partial\hat{\mathcal L}/\partial\vC$, $n\!\to\!m$, $\sim 10^{-1}$) from motivational prose; replace the "gradient is scaled by the per-bin rate" passage with a plain-language failure story plus one concrete analogy. Defer all formalism to Background/Method.
2. **Fix the two-vs-three counting mismatch.** The intro frames "two issues" but then says "We address all three." Reconcile the enumeration so the reader can map problems to contributions one-to-one.
3. **Lock the terminology for the three contributions.** Pick one canonical name for each (e.g., "projection encoder," "decoupled online readout," "square-root RLS flow learner") and use it identically in abstract, intro, method headings, results, and discussion.
4. **Add a crystallizing one-sentence summary** (Standard B) of the honest verdict, ideally italicized, in both abstract and discussion.
5. **Reconcile reported numbers and comparison classes** between text, table, and the ground-truth result set (one-step R^2 quoted as 0.58/0.71/0.71 in one place and as part of broader comparisons elsewhere; "within 0.02 of ceiling" needs a single consistent referent).

**Recurring patterns of confusion.**

- **Formalism-before-motivation** in the front matter (Standard B violation), then again at the top of each Method subsection where the failure mode is restated symbolically.
- **Synonym drift** for the same three contributions and for the same metric ("filtered-latent R^2" vs "aligned latent R^2" vs "latent R^2").
- **Hedge-stacking** ("honest," "honestly mixed," "mixed picture," "an effect we ... leave as an open problem") repeated enough that the modesty becomes a refrain rather than a single calibrated statement.

---

## 2. Major Issues

### Issue 1: Introduction and Abstract carry inline math and formal notation
**Location:** Abstract (lines 30-45); Introduction, first paragraph (lines 47-60), esp. lines 54-57.
**Type:** Clarity / Conceptual order / Style (Standard A and B violation).
**Problem:** The motivational onramp embeds formal symbols and notation before they are defined: "the loading matrix $\vC$ and bias $\vb$" (line 54-55), "the gradient of the variational objective with respect to $\vC$ is scaled by the per-bin rate ($\sim 10^{-1}$)" (lines 55-57), "$n\!\to\!m$" inversion language is foreshadowed, and the abstract leans on "$\sim$3 ms per 5 ms bin," "$0.02$," and SNR specifics. The contribution sentence in the abstract also front-loads three parenthesized mechanisms.
**Why it matters:** Per Standard A the first read should serve a naive-but-smart reader: context in prose, formalism deferred. Per Standard B the Introduction and Abstract must motivate in prose with a vivid concrete analogy *before* abstraction; inline math here forces the reader to hold undefined notation ($\vC$, $\vb$, the ELBO gradient, the rate scale) in mind with no setup. The "$\sim 10^{-1}$" and "$\partial\hat{\mathcal L}/\partial\vC$" are exactly the kind of formal detail that belongs in Background/Method.
**Recommended action:** Strip all symbols from the Abstract and Introduction. State the collapse failure as a phenomenon: when the readout is unknown and spikes are rare, naive gradient learning of the readout has almost nothing to push on each bin, so it gives up and explains only the average firing rate. Introduce one tectonic contrast (e.g., *known map vs. unknown map*, or *dense, informative observations vs. sparse, near-silent spikes*) and one concrete analogy before any abstraction. Move "$\vC$, $\vb$," the rate scaling "$\sim 10^{-1}$," and "$\partial\hat{\mathcal L}/\partial\vC$" to Background (Sec. 2) and Method (Sec. 3.2), where they already reappear.
**Suggested wording (analogy, prose only):** "Learning the readout from sparse spikes online is like trying to calibrate a camera from a nearly dark room: most frames carry almost no signal, so a naive estimator defaults to reporting the average brightness and never recovers the scene. We sidestep this by estimating the readout with a dedicated, well-conditioned online rule rather than the variational gradient, and by handing the recognition network the readout's own approximate inverse instead of the raw spikes."

### Issue 2: "Two obstacles / two issues" vs. "We address all three / three coupled changes"
**Location:** Abstract "two obstacles remain" (line 32); Intro "Two issues stand between VJF and a turnkey real-time tool" (line 53), enumerated "First ... Second ..." (lines 53-60); then "We address all three" (line 62) and "We make three coupled changes" (line 93).
**Type:** Logic / Structure / Claim consistency.
**Problem:** The problem statement enumerates two issues (unknown readout; correct streaming control flow). The contribution statement enumerates three changes (projection encoder, decoupled online readout, square-root RLS). The abstract similarly lists "two obstacles" but then (i)/(ii)/(iii) three mechanisms. The reader cannot map problems to solutions because the counts do not line up, and "all three" has no antecedent set of three problems.
**Why it matters:** C-C-C requires a clean problem -> contribution mapping. "We address all three" is a dangling reference; the third issue (low-rate collapse) is buried *inside* the first ("First, the observation model ... is unknown ... and the fit collapses") rather than listed as a peer.
**Recommended action:** Either (a) split the first issue into two ("the readout is unknown" and "naive gradient estimation of it collapses"), yielding three problems that map to the three contributions plus the control-flow packaging, or (b) keep two problems and rephrase "We address all three" to "We make three coupled changes that resolve both." Make the abstract's "two obstacles" consistent with whatever count the intro uses.

### Issue 3: Inconsistent names for the three contributions across sections
**Location:** Abstract (lines 36-39); Intro (lines 62-66); Method headings (lines 97, 123, 162); Discussion (lines 285-292).
**Type:** Terminology.
**Problem:** The same three ideas are renamed at each appearance. Examples: abstract "subspace projection of the observations" -> intro "projection-input recognition" -> method "Projection-input recognition" / results "projection input" / "projection encoder" / "proj. enc." Abstract "estimates the readout online and decoupled from the variational gradient by incremental PCA with an orthogonal-Procrustes anchor" -> intro "decoupled online readout estimated by incremental PCA" -> method "Decoupled online readout estimation" -> results "online readout" / "online $\hat\vC_t$." Abstract "square-root recursive least squares flow learner" -> intro "square-root RLS dynamics learner" -> method "Square-root RLS dynamics."
**Why it matters:** Standard A forbids synonym drift; the reader must re-identify each contribution on every mention. "Projection encoder" vs "projection-input recognition" vs "projection input" especially blurs whether these are the same thing.
**Recommended action:** Choose one canonical short name per contribution and use it verbatim everywhere (a one-line glossary in the intro helps): e.g., **projection encoder**, **decoupled online readout (CCIPCA + Procrustes)**, **square-root RLS flow learner**. Allow the longer descriptive form only at first definition.

### Issue 4: Metric naming drift ("filtered-latent" vs "aligned latent" vs "latent" R^2)
**Location:** Abstract "recovers the latent dynamics" (line 41); Results "filtered-latent $R^2$" (lines 189, 202, 207); Fig. curves "Aligned latent $R^2$" (line 256); text "latent $R^2$" (line 238); "instantaneous filtering is accurate" (line 292).
**Type:** Terminology / Notation.
**Problem:** The primary metric "filtered-latent $R^2$ (best affine image of the posterior mean vs. the true latent)" is later called "aligned latent $R^2$" (figure 4 caption) and "latent $R^2$" (line 238). A reader cannot be certain these are the same quantity, especially since "best affine image" and "aligned" each describe the alignment step differently.
**Recommended action:** Use "filtered-latent $R^2$" consistently, including in the Fig. `curves` caption. If "aligned" is meant to emphasize the affine alignment, define once and reuse.

### Issue 5: One-step R^2 numbers and comparison classes are not internally consistent
**Location:** Results (1): "most strongly on one-step prediction ($0.84/0.95/0.97$ vs.\ $0.60/0.85/0.92$)" (lines 224-225); Results (2): "online one-step $R^2$ ($0.58/0.71/0.71$)" (line 233).
**Type:** Evidence / Claim consistency.
**Problem:** Three different one-step triples appear (projection+oracle 0.84/0.95/0.97; spike+oracle 0.60/0.85/0.92; online 0.58/0.71/0.71) but only the first two are labeled by configuration in-line; the third (0.58/0.71/0.71) is introduced as "online one-step $R^2$" without stating it is the $\tau{=}8$ online configuration, and these one-step numbers never appear in Table 1 (which is filtered-latent only). The reader cannot verify the comparison or reconstruct the full one-step picture. Cross-check note: the ground-truth set the author supplied lists projection+oracle one-step 0.84/0.95/0.97 and spike+oracle 0.60/0.85/0.92, which match; the online one-step 0.58/0.71/0.71 is not in that supplied set, so confirm it is correct and from the same run.
**Why it matters:** One-step prediction is load-bearing for the central "honest verdict" (online readout trails on dynamics at high SNR). Numbers scattered in prose, partly unlabeled, partly absent from any table, are hard to audit and easy to mis-cite.
**Recommended action:** Add the one-step prediction R^2 (and rate correlation, forecast horizon) as columns or a companion table so every quoted triple has a table home. Label the 0.58/0.71/0.71 triple explicitly as "online $\tau{=}8$." Verify the online one-step values against the run logs.

### Issue 6: "Within 0.02 of the ceiling" has shifting referents
**Location:** Abstract "comes within $0.02$ of the known-readout ceiling at low SNR" (line 43); Results (2) "within $0.02$ of the projection ceiling ($0.84$ vs.\ $0.86$)" (line 228).
**Type:** Claim consistency / Clarity.
**Problem:** The abstract says "known-readout ceiling"; the body says "projection ceiling (projection+oracle)." These are the same row (projection+oracle = the ceiling), but the abstract's phrase "known-readout ceiling" could be read as spike+oracle (also a known readout). The 0.02 gap is filtered-latent R^2 only and at 3 dB only; the abstract should say so.
**Recommended action:** In the abstract, name the ceiling consistently ("the projection+oracle ceiling") and scope the 0.02 claim to filtered-latent R^2 at low SNR. State plainly that it does *not* close the gap on one-step prediction or forecast horizon, matching the honest verdict in Results (2).

### Issue 7: Honesty refrain is repeated rather than crystallized
**Location:** Abstract "an honest, mixed picture" idea; Intro "report an honest, mixed picture" (line 69); Results "the picture is honestly mixed" (line 203); plus "leave as an open problem" (line 44).
**Type:** Style / Clarity (Standard B: one crystallizing sentence).
**Problem:** The modesty is asserted three times in nearly identical words but never distilled into a single calibrated takeaway sentence. Standard B asks for one crystallizing (often italic) summary of the main result.
**Recommended action:** Replace the repeated "honest/mixed" assertions with one crystallizing sentence and reuse it (italicized) in abstract and discussion. Candidate verdict (matches the data): *the projection encoder is a robust fixed-readout win; the online readout reaches the ceiling only at low SNR, where periodic-refresh frame churn does least damage to the dynamics; smoothing buys rate fidelity in the noisy regime.*

### Issue 8: Causal/mechanistic claim ("frame churn caps high-SNR accuracy") stated more firmly than the evidence
**Location:** Abstract "an effect we trace to the frame motion induced by periodic readout refresh" (lines 43-44); Results (2) "The periodic Procrustes refresh keeps moving the latent frame, so the dynamics fit a moving target" (lines 231-232); Fig. forecast "The smaller-orbit relaxation is the frame-churn effect that caps high-SNR one-step accuracy" (lines 279-280); Discussion (lines 290-292).
**Type:** Claim consistency / Evidence.
**Problem:** The frame-churn mechanism is presented as established cause ("trace to," "is the frame-churn effect that caps") but the paper shows correlation (refresh present + dynamics metrics trail) and a plausible argument (moving target), not a controlled test (e.g., refresh-frozen ablation, or the proposed RBF-rotation fix actually run). The figure caption ("is the frame-churn effect") is the strongest of the three and exceeds the hedged Discussion ("Promising remedies, left as future work").
**Why it matters:** A mechanistic claim asserted in a caption but only argued (not isolated) in the text is exactly the caption-stronger-than-text pattern to avoid.
**Recommended action:** Soften the caption and abstract to attributive/consistent-with language unless an ablation exists. If a refresh-frozen or RBF-rotation experiment was run, cite it; if not, say "consistent with" / "we attribute this to." Align all three mentions to the same strength as the Discussion's future-work framing.

### Issue 9: "We package the streaming loop as a reusable per-sample primitive" is a contribution with no evidence section
**Location:** Abstract (iii region) and Intro "we also package the streaming loop as a reusable per-sample primitive" (lines 65-66); Method 3.3 "encapsulated as a single per-sample generator" (lines 172-173).
**Type:** Structure / Claim consistency.
**Problem:** The control-flow packaging is listed among contributions (and is arguably "the third issue") but the Results/Experiments never evaluate or demonstrate it beyond the per-bin timing. It is an engineering claim asserted, not shown (no code reference, no interface sketch, no reproducibility statement).
**Recommended action:** Either demote it from a headline contribution to an implementation note, or support it (point to released code, show the one-function interface, or state reproducibility). Decide its role in the two-vs-three count (Issue 2).

### Issue 10: Figure captions vary in whether they state a conclusion
**Location:** Fig. raster (lines 196-198); Fig. summary (lines 247-249); Fig. curves (lines 256-259); Fig. kstep (lines 266-269); Fig. forecast (lines 277-280).
**Type:** Structure (Standard A: captions should state the conclusion).
**Problem:** Some captions state a conclusion well (summary, curves, forecast); raster is mostly descriptive ("Example input ... rotational structure is barely visible at $n{=}50$ and sharp at $n{=}250$" does carry a point, acceptable). kstep is largely descriptive of the curves. Mixed conclusion-bearing vs label-only captions.
**Recommended action:** Ensure each caption ends with the takeaway. For kstep, add the conclusion explicitly (e.g., "the online flow forecasts roughly two cycles ahead at high SNR but loses skill within ~12 bins at 3 dB"). For forecast, see Issue 8 (soften the mechanistic claim).

---

## 3. Terminology and Notation Audit

| Term / symbol | Where issue occurs | Problem | Recommended action |
|---|---|---|---|
| $\vC$, $\vb$ | Abstract/Intro (lines 43, 54-57) | Used in motivational prose before Background defines them | Remove from front matter; first define in Sec. 2 (eq. lik) |
| $\partial\hat{\mathcal L}/\partial\vC$ | Intro (line 56), Method (lines 95, 124) | Formal gradient in Introduction | Defer to Method 3.2; describe in prose in intro |
| $n\!\to\!m$ / $m\!\to\!m$ / $m\!\ll\!n$ | Intro foreshadow; Method (lines 98, 114-117, 226) | Dimension-arrow notation leaks toward narrative | Keep in Method; in intro say "high-dimensional spikes vs low-dimensional latents" |
| $\sim 10^{-1}$ | Intro (line 57), Method (line 126) | Numeric rate scale in motivational prose | Remove from intro; retain in Method 3.2 |
| "projection-input recognition" / "projection input" / "projection encoder" / "proj. enc." | lines 36, 62, 97, 188, 214, 223, 287 | Four names for one contribution | Standardize on "projection encoder" |
| "decoupled online readout" / "online $\hat\vC_t$" / "online readout estimation" / "subspace projection ... readout" | lines 36-38, 62-64, 123, 216-217, 227 | Multiple names | Standardize on "decoupled online readout" |
| "square-root RLS flow learner" / "square-root RLS dynamics learner" / "square-root RLS dynamics" | lines 38-39, 65, 162 | Naming drift | Standardize on "square-root RLS flow learner" |
| "filtered-latent $R^2$" / "aligned latent $R^2$" / "latent $R^2$" | lines 189/207 vs 256 vs 238 | Same metric, three names | Use "filtered-latent $R^2$" everywhere incl. Fig. curves |
| "readout" / "observation model" / "loading matrix $\vC$" / "decoder" | lines 30-31, 54, 144, 146 | Three+ terms for the $\vC$-map; "decoder" appears late (line 144) without binding to "readout" | Define "readout = observation model = $\vC,\vb$" once; pick one term; bind "decoder" to it |
| $\vr_t$ (centered feature) vs $\vR$ / $\vR^\star$ (rotation) vs $\vu_t$ (control) | lines 130-137, 155, 295 | Author already flags $\vr_t$ "distinct from the control $\vu_t$"; $\vr$/$\vR$ visual clash | Consider renaming centered feature to avoid $\vr$/$\vR$ collision; keep the existing disambiguation note |
| $\vpi_t$ (projection) vs $\vpi$ as constant | line 16 defines `\vpi`; used as projection (eq. proj) | $\bm\pi$ conventionally a probability/constant; here it is the projection feature | State explicitly "$\vpi_t$ denotes the projection feature" at first use (it is partly stated) |
| "$g$" link vs "$\tilde g$" approx inverse link | eq. lik (line 75), eq. proj (lines 103-109) | $\tilde g$ introduced as "variance-stabilizing, approximate inverse link"; relation to $g=\exp$ stated only via the point-process instance | Add one clause: $\tilde g \approx g^{-1}$ in general; $\tilde g=\log(\cdot+c)$ for Poisson |
| "SNR" in dB vs $n$ neurons | Abstract (line 43), Exp (line 183), Table 1 | dB values ($3/6/8$) introduced in Exp; abstract says "low SNR"/"high SNR" without the dB anchor | Acceptable, but define the dB-to-$n$ mapping at first dB mention |
| $K$ (readout write interval) vs $k$ (forecast horizon) | lines 146, 242, 266 | Case-sensitive collision ($K$ vs $k$) | Rename one (e.g., readout interval $K_{\mathrm{ref}}$) to avoid $K$/$k$ confusion |
| $\theta$ | eq. ELBO (lines 84-85) | Used as parameter bundle without definition | Add "$\theta$ collects all model parameters" |
| $H(q)$ | eq. ELBO (line 85) | Differential entropy not named | Name it "entropy term $H(q)$" inline |
| "$60n$-bin window" / "$\sim2\times10^5$-step" / "$200{,}000$ bins" | lines 167, 186, 180 | Mixed notations for stream length and window | Standardize numeric style |

---

## 4. Section-by-Section Notes

**Abstract (lines 29-45).**
- Purpose: clear (mini-story present: gap, approach, result, verdict). Good.
- Issues: inline math and numerics (Issue 1, 6); "two obstacles" vs three mechanisms (Issue 2); "known-readout ceiling" referent (Issue 6); missing a single crystallizing verdict sentence (Issue 7). The final sentence is very long (lines 39-44, one sentence spanning runs in real-time, recovers dynamics, projection win, online within 0.02, trails at high SNR, frame motion, open problem) - split into two.
- Action: rewrite as prose-first; defer symbols; one crystallizing italic verdict.

**1. Introduction (lines 47-69).**
- Purpose: mostly clear, but does not follow field-gap -> subfield-gap -> specific-gap -> "Here we..." cleanly. It opens at the subfield (closed-loop streaming inference) and jumps to VJF immediately; the broad field foundation and the tectonic contrast/analogy (Standard B) are missing.
- Conceptual order: failure mode (collapse) is stated symbolically before the reader has the generative model (Issue 1). The third "issue" is hidden inside the first (Issue 2).
- Flow: "We address all three" (line 62) has no three-item antecedent.
- Action: add one broad opening + one analogy; remove symbols; align the count; end with a crisp "Here we..." that names the three canonical contributions.

**2. Background (lines 71-90).**
- Purpose: clear and appropriately formal. This is the right home for $\vC$, $\vb$, the ELBO, identifiability.
- Flow: good. The identifiability paragraph (lines 87-90) sets up the Procrustes need well.
- Minor: define $\theta$ and name $H(q)$ (table). "$g=\exp$" then "$P(g(\cdot))$" - state $P$ is the point-process/Poisson likelihood explicitly.

**3. Method (lines 92-176).**
- Purpose: clear; the "three coupled changes" framing and per-subsection "Difference from VJF:" markers are excellent C-C-C devices - keep them.
- 3.1 Projection (lines 97-121): strong. Restates the failure ("hard to learn at low SNR") in prose - good. The ELBO-validity argument (lines 117-119) is a nice defensive point. $\tilde g$ relation to $g$ could be one clause clearer (table).
- 3.2 Readout (lines 123-160): restates the $\sim10^{-1}$ collapse symbolically again (fine here, the right place). CCIPCA derivation is dense but correct in structure. The Procrustes paragraph is well-motivated. Consider a one-line intuition before the Hebb/deflate equations for the naive-but-smart reader.
- 3.3 Square-root RLS + loop (lines 162-176): two distinct contributions (SR-RLS *and* the control-flow packaging) are merged into one subsection. The packaging is a list of mechanisms (warm-up, divergence recovery, refresh, timing) crammed into one long sentence (lines 168-174). Split into (a) SR-RLS and (b) the per-sample loop, and decide whether the loop is a headline contribution (Issue 9).

**4. Experiments (lines 178-199).**
- Purpose: clear. The five-configuration cross (recognition input x readout source) is well-specified.
- Flow: good. Defines all metrics up front (lines 189-191).
- Minor: the dB-to-$n$ mapping ("$\sim3/6/8$ dB") appears here first; ensure the abstract's "low/high SNR" ties back. "$60n$-bin window" for frozen PCA - state the absolute window length for at least one $n$.

**5. Results (lines 201-242).**
- Purpose: clear; three numbered points map to the three findings. Good question->evidence->answer structure within each.
- Issues: numbers scattered and partly unlabeled / not in any table (Issue 5); "within 0.02" referent (Issue 6); mechanistic claim strength (Issue 8); metric-name drift (Issue 4). The "honestly mixed" refrain (Issue 7).
- Point (2) is a single very long sentence chain (lines 230-235) mixing filtered R^2, one-step R^2, forecast horizon, and a figure pointer - split.
- Action: add a one-step/rate/forecast table; label every triple; soften captions.

**6. Discussion (lines 284-296).**
- Purpose: clear and appropriately scoped; the future-work remedies (anneal/freeze refresh, rotate RBF flow by $\vR^\star$) are concrete and well-judged.
- Issues: this is the best-calibrated statement of the verdict - promote a one-sentence version of it to the abstract (Issue 7). Mechanistic claim here is correctly hedged ("Promising remedies, left as future work"); make caption/abstract match this strength (Issue 8).
- Missing: an explicit limitations sentence beyond the high-SNR readout gap - e.g., single synthetic benchmark, one trajectory class (limit cycle), no real data. State scope ("to our knowledge," "in practice," "on this benchmark") per Standard B.

---

## 5. Local Clarity and Style Issues

- **Location:** Abstract, lines 39-44.
  **Issue:** Single sentence ~6 clauses long ("the method runs ... and ... recovers ...; the projection encoder is ... while the online readout comes within ... and trails it ... --- an effect we trace ... and which we leave as an open problem").
  **Suggestion:** Split into three: (timing+recovery), (projection win + online verdict), (open problem).

- **Location:** Line 62, "We address all three."
  **Issue:** Dangling reference (see Issue 2).
  **Suggestion:** "We make three coupled changes that resolve both obstacles" (or re-enumerate to three problems).

- **Location:** Line 95, "Vanilla VJF is recovered by setting $\vpi_t:=\vy_t$ and learning $\vC$ by $\partial\hat{\mathcal L}/\partial\vC$."
  **Issue:** Good and precise - keep. Ensure "vanilla VJF" is the consistent baseline name (also called "raw-VJF," line 188, 215; "plain RLS," line 167).
  **Suggestion:** Standardize "vanilla VJF" vs "raw VJF."

- **Location:** Lines 130-131, "(which serves as $\vb$)" and "(distinct from the control $\vu_t$)."
  **Issue:** Two parentheticals close together; the second is a good disambiguation - keep; the first is slightly cryptic.
  **Suggestion:** "running mean $\bar\vb_t$, which we use as the bias $\vb$ in eq. (lik)."

- **Location:** Line 203, "Three points stand out, and the picture is honestly mixed."
  **Issue:** Weak/colloquial topic sentence ("honestly mixed"); see Issue 7.
  **Suggestion:** Replace with the crystallizing verdict sentence.

- **Location:** Line 228-229, "and well above frozen-PCA $0.72$ and spike-oracle $0.69$".
  **Issue:** Parenthetical comparison nested inside an already-parenthetical claim; comparison class ("well above") undefined in magnitude.
  **Suggestion:** Pull the comparison into the table reference; keep prose to the headline gap.

- **Location:** Line 240-242, "Per-bin online cost is flat at $\sim3$ ms ($50\!\to\!250$ neurons) ... with zero divergences over the $1000$ s pass. At good SNR the learned flow forecasts $\sim2.4$ cycles ahead".
  **Issue:** "$50\!\to\!250$ neurons" uses arrow notation for a range; "$\sim2.4$ cycles" introduced here but the cycle-length anchor (~42 bins) is only in Fig. kstep caption.
  **Suggestion:** "from 50 to 250 neurons"; state cycle length in text once.

- **Location:** Line 292, "even when instantaneous filtering is accurate."
  **Issue:** "instantaneous filtering" is a new term for "filtered-latent R^2"; synonym drift.
  **Suggestion:** "even when the filtered-latent estimate is accurate."

- **Location:** Lines 168-174 (the per-sample-loop sentence).
  **Issue:** One sentence lists 6+ mechanisms with nested parentheticals; main point (one function per sample) buried at the end.
  **Suggestion:** Lead with the takeaway, then bullet or itemize the mechanisms.

- **Location:** Abstract line 31, "online, one observation at a time."
  **Issue:** Good plain-language phrasing - this is the register the whole abstract should keep.

---

## 6. Proofreading and Formatting List

- Line 4: `\documentclass[11pt]` with `\usepackage[preprint]{catniplab}` - confirm 11pt is intended for preprint class (not a class-option clash).
- Line 22: title uses `\\` line break inside `\title{}` - fine, but verify it renders in the preprint style.
- Equation tags: the paper uses manual `\tag{lik}`, `\tag{gen}`, `\tag{recog}`, `\tag{ELBO}`, `\tag{recog$'$}`, `\tag{EMA}`, `\tag{Hebb}`, `\tag{deflate}` alongside `\label{eq:...}`. `recog'` (line 105) has a `\label` ? No - eq. recog' (line 105) has a `\tag` but **no `\label`**, while recog (line 83) is labeled. If recog' is never `\cref`'d this is fine; otherwise add a label. Verify all `\cref{eq:...}` targets exist.
- `\cref` usage: `\Cref{sec:exp,sec:results}` (line 68), `\Cref{fig:summary}` etc. - confirm `cleveref` + the manual `\tag` equations cross-reference correctly (manual `\tag` can break `\cref` to equations; check `\cref{eq:elbo}`, `\cref{eq:lik,eq:gen}`, `\cref{eq:proj}`, `\cref{eq:recog}` all resolve to numbers, not "??").
- Line 95 vs 188/215: "vanilla VJF" vs "raw-VJF" vs "raw VJF" - unify hyphenation and term.
- Line 167: "plain RLS" vs "vanilla" elsewhere - unify.
- Hyphenation: "real-time" (consistent, good); "low-rate" / "low rate" - check uniformity; "square-root RLS" (consistent); "frozen-PCA" (line 218) vs "frozen PCA" (line 186, 289) - unify.
- "$\hat\vC_t$" vs "$\hat\vC$" (line 146 vs 155) - subscript usage inconsistent; pick one for the current estimate.
- Line 113: "$\tau{=}1$ disables smoothing" - good; ensure "$\tau$" defined as integer/real consistently (memory "$\sim\tau$ bins").
- En dash usage: the source uses `---` (em dash) in several places (lines 43, 116, 230, 240, 287). Per the author's ASCII/typography rule, em dashes should be hyphens; in LaTeX prose the author may intend em dashes, but global instruction says use hyphens - confirm intent. (Flag, do not auto-change.)
- Line 197: "$n{=}50$" and "$n{=}250$" use `{=}` tight spacing consistently - good; ensure all numeric "=" use the same convention (line 216-218 table uses `$\tau{=}8$` in body but table cells "$\tau{=}8$"/"$\tau{=}1$" - consistent).
- Line 233: "$0.58/0.71/0.71$" - verify against run logs and label as online $\tau{=}8$ (Issue 5).
- Table 1 (lines 210-220): bold marks "best achievable per SNR" - at 3 dB bold is 0.84 (online $\tau{=}8$); at 6/8 dB bold is frozen-PCA 0.87/0.92. Caption says bold = best non-oracle; verify 0.88 ($\tau{=}8$, 6 dB) vs 0.87 (frozen, 6 dB): **0.88 > 0.87**, so the bold at 6 dB may be on the wrong cell - the $\tau{=}8$ value 0.86 (line 216) and frozen 0.87 - recheck. Actually row $\tau{=}8$ 6 dB = 0.86, frozen 6 dab = 0.87, so frozen bold is correct. But cross-check 8 dB: $\tau{=}1$ = 0.88 vs frozen 0.92 - frozen correct. **Confirm 3 dB: online $\tau{=}8$ 0.84 vs frozen 0.72 - online bold correct.** No error found, but the near-ties warrant a re-verify pass.
- Line 256 caption: "Aligned latent $R^2$" - change to "Filtered-latent $R^2$" (Issue 4).
- Line 295: "$\bm{\xi}_i\mapsto\vR^\star\bm{\xi}_i$" - `\xi` RBF centers introduced only here; if RBF centers appear in Method (f=W phi), define $\bm\xi_i$ at first use in 3.3, not in Discussion.
- Tense: Methods mostly present tense (good); Results mix present ("stand out," "beats") and is consistent enough. Discussion present - fine.
- Citation formatting: `\autocite` used consistently; verify `references.bib` has all keys (zhao2020variational, kingma2014auto, weng2003candid, macke2011empirical, yu2009gaussian, haykin2014adaptive, schoenberg1983implementation) - 7 keys, confirm all resolve.
- Line 109: "$\tilde g(\vy_t)=\log(\vnu_t+c)$" - constant $c$ undefined (pseudocount). Define $c$.
- Line 142: "$\vc_i=\hat\vv_i\sqrt{\norm{\vv_i}}=\ve_i\sqrt{\lambda_i}$" - check: at convergence $\norm{\vv_i}=\lambda_i$, so $\sqrt{\norm{\vv_i}}=\sqrt{\lambda_i}$, consistent; but loading column = eigenvector x sqrt(eigenvalue) requires unit-variance latent assumption - already stated, good.

---

## 7. Priority Revision Plan

1. **De-mathematize the Abstract and Introduction** (Issue 1): remove $\vC$, $\vb$, $\partial\hat{\mathcal L}/\partial\vC$, $n\!\to\!m$, $\sim 10^{-1}$, and most numerics; motivate in prose with one tectonic contrast and one concrete analogy; defer formalism to Background/Method.
2. **Fix the two-vs-three counting** (Issue 2) and add a clean problem -> contribution mapping ending in a crisp "Here we..." that names the three canonical contributions.
3. **Lock terminology for the three contributions and the primary metric** (Issues 3, 4): one canonical name each, used verbatim across abstract, intro, method headings, results, captions, discussion.
4. **Reconcile and table-ize the numbers** (Issue 5): add a one-step / rate-corr / forecast-horizon table; label every quoted triple (esp. online 0.58/0.71/0.71); re-verify table bolding near-ties.
5. **Calibrate the frame-churn mechanistic claim** (Issue 8): soften abstract and Fig. forecast caption to match the Discussion's future-work hedge unless an ablation exists; if it exists, cite it.
6. **Add one crystallizing verdict sentence** (Issue 7) and remove the repeated "honest/mixed" refrain; add an explicit limitations/scope sentence to the Discussion (single synthetic benchmark, limit-cycle only, no real data).
7. **Restructure Method 3.3** (Issue 9): split SR-RLS from the per-sample loop; decide whether the loop is a headline contribution or an implementation note, and support it accordingly.
8. **Caption pass** (Issue 10): ensure every figure caption ends with its conclusion.
9. **Proofreading and formatting pass** (Section 6): unify hyphenation (frozen-PCA, raw/vanilla VJF), define $c$, $\theta$, $\bm\xi_i$, $K$ vs $k$; verify all `\cref`/`\autocite` targets resolve; confirm em-dash vs hyphen intent.

---

*Audit notes: This is a diagnostic report; no manuscript text was modified. Numeric claims were cross-checked against the author-supplied ground-truth result set where overlap existed (filtered-latent R^2, one-step projection/spike+oracle, rate corr tau=8/tau=1, forecast horizon k=11/82/100, ~3 ms/bin) and matched, except the online one-step triple 0.58/0.71/0.71 which is not in the supplied set and should be verified against run logs.*
