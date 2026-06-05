# Experiment plan: make every figure test a claim (codex-reviewed, v4)

Goal: replace illustrative or weak figures with experiments that directly characterize the
paper's claims. The earlier draft inferred the central mechanism from downstream performance;
this version (after `/codex-review`) adds a **direct measurement** of the drifting readout
subspace and a **clean positive control**, removes confounded comparisons, and strengthens the
statistics. Plan first -> already codex-reviewed -> implement -> run on GCP. Nothing runs until
approved.

**Terminology.** The readout $\vC$ is a factor loading, identifiable only up to a rotation (and
scaling) of the latent factors -- the standard **factor-rotation invariance**,
$\vC\vx=(\vC\vR)(\vR^{-1}\vx)$. When the online readout is refreshed, each rewrite of $\vC$
re-aligns the estimated factor subspace; the accumulation of these re-alignments is **subspace
drift** (equivalently, drift of the latent alignment). Because the recursive dynamics
$f=\vW\vphi$ are learned in the current alignment, subspace drift forces them to track a moving
target. We quantify drift by the principal angle between successive readout subspaces (and
relative to the oracle subspace).

## 1. Claims -> current evidence (audit)

| # | Claim | Current figure(s) | Verdict |
|---|-------|-------------------|---------|
| C1 | Projection encoder beats raw-spike at **fixed** readout | Fig 2, Tab 1/2 | supported; **mechanism not isolated** |
| C2 | Online readout reaches the ceiling only at low SNR | Fig 2, Fig 3 | supported (mostly mid/high SNR) |
| C3 | High-SNR shortfall is **caused by subspace drift** | none (asserted) | **not tested** -- the key gap |
| C4 | Smoothing helps rate fidelity / low SNR | Fig 2 (only two tau) | weak: too few points for a trend |
| C5 | Real-time (~3 ms/bin) | text only | **not actually designed** |

Non-claim figures: Fig 1 (raster, setup) and Fig 5 (single-trial forecasts, qualitative) -> appendix.

## 2. Cross-cutting methodology

- **SNR conditions extended to the neural regime: $\{-3, 0, 3, 6, 8\}$ dB** (Fisher-information SNR),
  set by population size and gain (smaller populations / lower gain for the new $0$ and $-3$ dB
  points; roughly $n\approx 15/30/50/150/250$). The $-3$ and $0$ dB regimes are typical of neural
  population recordings and are precisely where an online readout should matter most, so they are
  central, not peripheral.
- **Affine-aligned filtered-latent $R^2$ can hide subspace drift** -> use it only as a
  recovery-quality metric, never as evidence for or against drift.
- **The statistical unit is the whole $200$k-bin stream.** Use **paired seeds** across conditions and
  report **per-seed paired differences** (not only mean $\pm$ s.e.). Pilot $=3$ seeds; the
  claim-critical C3 comparisons use **$\ge 8$ independent streams**.
- **Forecasting:** raise the evaluation limit to $k=200$ (a *higher cap*, not "uncapped"). With an
  orbit period of $\sim 42$ bins, "largest $k$ with $R^2\ge 0$" is misleading; report the **full
  $k$-step $R^2$ curve, the area under it to $k=200$, and the first sustained threshold crossing**.
- **Pre-register a decision rule** for each experiment; each must be able to falsify its claim.

## E0 -- Direct measurement of subspace drift (underpins C3; prerequisite for E1/E2)

Log per step / per refresh and persist to `summary.json`:
- principal angles between $\mathrm{span}(\vC_t)$ and (i) the oracle $\mathrm{span}(\vC)$, (ii)
  the previous estimate $\mathrm{span}(\vC_{t-K})$;
- per-refresh re-alignment: the Procrustes rotation angle, any reflection/sign flips, and the
  **cumulative re-alignment $\sum_t|\theta_t|$**;
- readout drift: bias $\vb_t$, column norms, singular values, condition number $\mathrm{cond}(\vC_t)$;
- **projection jump** on a fixed held-out $\vy$: $\lVert \vC_\text{new}^+(\tilde g-\vb_\text{new}) -
  \text{align}(\vC_\text{old}^+(\tilde g-\vb_\text{old}))\rVert$ immediately before/after each refresh;
- **event-triggered** traces around refreshes: one-step error, latent jump, RLS update norm, forecast loss.

This turns C3 from an inference into a measurement and gives the x-axis ("measured subspace re-alignment")
for E1/E2.

## 3. Experiments

### E1 -- Causal test of subspace drift + the fix (C3; centerpiece)
High SNR ($n=250$ / $8$ dB, where the shortfall is largest), projection encoder, **$\ge 8$ paired
streams**. Conditions:
- (a) **baseline** online readout, refresh every $K$ (current behavior).
- (b) **subspace-tracking fix**: at each refresh apply the re-aligning rotation $\vR^\star$ to the RBF
  centers ($\vxi_i\mapsto\vR^\star\vxi_i$) and $\vW$, so the flow travels with the factor subspace
  (the Discussion's remedy).
- (c) **oracle readout with imposed periodic factor rotations** (the clean positive control): the true
  $\vC$ (no estimation error), but with a known periodic re-alignment of the factor subspace applied.
  If imposed re-alignment degrades the dynamics and the subspace-tracking fix removes the deficit, drift
  is the cause -- isolated from readout-estimation error.
- (d) **decoder-frozen-after-$S$** (CCIPCA keeps estimating internally, but $\vC$ is *not written* to
  the decoder/projection after step $S$): separates "estimate keeps moving" from "model alignment keeps moving".
- (e) **post-hoc flow refit** on final-alignment latents: if it recovers the ceiling, online
  moving-target training is implicated; if not, the deficit is latent quality / model capacity, not drift.
- (controls) frozen batch PCA at the same $S$/warm-up budget; plain freeze-after-$S$ (reported only as a
  reference, *not* as C3 evidence -- it is confounded).
**Metrics:** one-step $R^2$, $k$-step AUC/horizon, filtered $R^2$ (quality only), against the
projection+oracle ceiling, all plotted **versus the measured cumulative re-alignment (E0)**, plus
event-aligned error.
**Decision rule (pre-registered; all required for C3):** (i) measurable re-alignment is present in the
baseline; (ii) errors rise at refresh events; (iii) **imposed re-alignment (c) reproduces the deficit**
at the oracle readout; (iv) **the subspace-tracking fix (b) closes $\ge 50\%$ of the
baseline$\to$oracle one-step gap**. If imposed re-alignment does *not* degrade the dynamics, C3 is
falsified and we retract it.
**New code (flagged, default off):** subspace-tracking in `OnlineReadout.maybe_refresh` (rotate
`transition.velocity.feature.centroid`/`w_mean`); an imposed-rotation injector; a decoder-freeze flag.

### E2 -- Refresh-interval $K$ sweep (the drift<->adaptation trade-off; supporting, not causal)
projection+online, $K\in\{250,1000,4000,\text{frozen-after-warm-up}\}$, at a low ($-3$ or $0$ dB) and a
high ($8$ dB) SNR, paired seeds. Define "frozen" precisely (frozen after warm-up). **Plot performance
versus the measured cumulative re-alignment (E0), not only versus $K$.** Metrics: high SNR -> one-step
gap closure + $k$-AUC; low SNR -> time to $90\%$ of final filtered $R^2$ (adaptation speed). A monotone
re-alignment$\to$deficit relation supports (does not prove) C3; a flat relation argues against it.

### E3 -- Smoothing $\tau$, with controls (C4 as a trend, de-confounded)
$\tau\in\{1,2,4,8,16,32\}$ across $\{-3,0,3,6,8\}$ dB. Controls to localize where $\tau$ acts:
- **projection+oracle** $\tau$ sweep (isolates recognition-input smoothing from readout estimation);
- if feasible, sweep the recognition-EMA and the readout-EMA **separately**.
**Report phase lag / time-shifted latent $R^2$** (large $\tau$ may look worse only because of the causal
delay) and **held-out Poisson NLL / deviance**, not only rate correlation.
**Decision rule:** at low SNR, rate NLL improves over $\tau=1$ with $<X$ drop in one-step/filtered $R^2$;
large $\tau$ shows the expected lag penalty.

### E4 -- Projection mechanism ablation (C1: dimensionality vs correct subspace)
At the oracle readout (matched EMA/log/scaling throughout), feed the recognition:
(i) raw spikes; (ii) the **full $\tilde g(\vy)-\vb$ in $n$ dimensions** (same preprocessing, no
projection -> isolates dimensionality from the link transform); (iii) $\vC^+(\tilde g-\vb)$ [ours];
(iv) a random orthonormal $m$-dim projection ($\ge 5$ draws); (v) a **shuffled / wrong-$\vC$** readout
(preserves norms, destroys tuning correspondence); (vi) top-$m$ PCA (with whitening) of $\tilde g$, fit
on warm-up.
**Report performance versus the principal angle to the true $\mathrm{span}(\vC)$** -> a mechanism curve,
not a categorical bar. **Decision:** if random/shuffled match (iii), the win is dimensionality; if
PCA(vi)$\approx$ours and random$\ll$ours, the win is the *correct subspace*.

### E5 -- Real-time timing (C5; currently undesigned)
Single pinned CPU core, warm-up excluded (stated). Report **p50/p95/max per-bin wall time**, with
**refresh-bin latency reported separately** from ordinary bins, across $n\in\{50,150,250\}$ (and the
smaller low-SNR populations). **Pass/fail:** p95 $<5$ ms/bin AND bounded refresh spikes. An average
alone is insufficient.

## 4. Figure plan (claims-first)

- Fig A (C1): Fig 2 summary across the five SNR levels -- keep. Fig B (C2): Fig 3 curves -- keep.
- **Fig C (C3, NEW):** E0/E1 -- measured subspace re-alignment + event-aligned errors + baseline /
  subspace-tracking / imposed-rotation against the oracle ceiling. Headline.
- **Fig D (C3, NEW):** E2 performance versus measured re-alignment (low vs high SNR).
- **Fig E (C4, NEW):** E3 $\tau$ trend (rate NLL + filtered $R^2$ + lag), with the proj+oracle control.
- **Fig F (C1, optional):** E4 performance versus subspace angle.
- **Fig G (C5, NEW):** E5 per-bin latency (p50/p95/max, refresh vs ordinary).
- Appendix: Fig 1 raster, Fig 5 single-trial forecasts, Fig 4 $k$-step.

## 4b. Figure production standards (publication quality)

- **Vector PDF**, not PNG (`savefig(format='pdf')`, `\includegraphics{figs/*.pdf}`).
- **Proper font sizes at final print size:** size each figure at its embedded width so it is included
  at scale $\sim 1$ (no LaTeX downscaling that shrinks fonts); target $\sim 8$-$9$ pt ticks,
  $\sim 9$-$10$ pt labels/legend against the $11$ pt body. Set once via rcParams.
- **Editable fonts** (`pdf.fonttype=42`, no Type-3 bitmaps); a consistent per-method/per-SNR color map
  across all figures; thin spines; line widths $\ge 1.5$ pt; minimal chartjunk.
- One plotting module regenerates every figure from the `summary.json` artifacts; replace
  `paper/figs/*.png` with `*.pdf`.

## 5. Logistics, time budget, and tracking

- **Time budget: target completion within 24 h wall-clock.** Estimate the full grid from the prior
  $\sim 11$ min/condition; if a single VM would exceed 24 h, **shard conditions across up to 4 parallel
  GCP instances** (`/gcp_run`, e2-standard-4, git mode pinned to one commit). Independent conditions
  (SNR levels, seeds, $K$/$\tau$ values, E1 arms) parallelize cleanly.
- **Keep timing clean:** the per-bin latency experiment (E5) runs alone on one dedicated VM (no
  co-tenancy) so the numbers are not contended; everything else may shard.
- **Progress document:** maintain `paper/EXPERIMENT_PROGRESS.md`, updated on each launch and each pull
  -- per experiment/condition: VM, commit, seed(s), status (queued/running/done/failed), wall time,
  results path, and key numbers. This is the single source of truth for run state.
- **Code home:** consolidate the harness (`experiment.py`, `lc_data.py` from
  `exp/lc-poisson-stream-20260602`) and the new instrumentation/flags (subspace-tracking,
  imposed-rotation injector, decoder-freeze, encoder ablation, timing) into `vjf/` on a branch off
  `feat/realtime-framework`, so paper, code, and experiments share commits.
- **Sequence (claim-critical first):** E0 instrumentation -> **E1 + E5** (C3 and C5, the un-tested
  claims) -> review the numbers -> E2/E3/E4 if they earn their figure.

## 6. Risks

- Subspace-tracking and the imposed-rotation injector touch dynamics state (RBF centers/weights) --
  equation-adjacent; exact only for a pure rotation (the re-aligning $\vR^\star$ is, by construction).
  Consult before changing any equation code; gate behind flags (default off).
- If imposed re-alignment does not degrade the dynamics (control c), or the subspace-tracking fix does
  not close the gap, **C3 is retracted** -- plan for the honest negative result.
- The oracle-readout control (E1c) is what separates readout-estimation error from subspace drift;
  without it, E1/E2 remain correlational.
