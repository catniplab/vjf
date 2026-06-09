# Experiment plan (v5): low-SNR is where the contributions matter

Method name (locked): **sVJF** (stable VJF). LaTeX macro `\svjf`.

Major revision after the E1 results. The originally hypothesized mechanism (high-SNR shortfall
caused by a rotating latent frame, fixable by rotating the RBF flow) is **refuted by the data and
dropped**, along with that experiment (the flow-tracking and imposed-rotation arms). The honest,
data-supported story is simpler and stronger:

> **The projection encoder and the decoupled online readout deliver in the low-SNR regime** that
> is typical of neural population recordings. At high SNR the problem is easy -- a plain frozen PCA
> readout is already near-oracle -- so the online machinery earns its keep precisely where readout
> estimation is hard (few neurons, sparse spikes). Once the online readout has converged, **freezing
> (or annealing) the refresh** recovers forecast skill that continued rewriting otherwise erodes.

We do not claim frame rotation as the mechanism, and we do not propose rotating the flow.

## 1. Claims (revised)

| # | Claim | Evidence |
|---|-------|----------|
| C1 | Projection encoder beats raw-spike recognition at fixed readout, **most at low SNR** | Fig 2 (5 SNR), Tab 1/2 |
| C2 | The decoupled online readout recovers the latent online; **its value is greatest at low SNR** (at high SNR frozen PCA ~ oracle) | Fig 2, Fig 3 |
| C3 (revised) | Continued readout refresh perturbs the learned dynamics; **freeze/anneal after convergence** recovers forecast skill | E1 (freeze-after vs online vs frozen-PCA vs oracle) |
| C4 | Smoothing helps rate fidelity / low SNR | E3 tau sweep |
| C5 | Real-time (~3 ms/bin, p95 < 5 ms) | E5 timing percentiles |

**Dropped:** the frame-churn causal claim, the flow-rotation remedy, and the subspace-tracking /
imposed-rotation experiment. (The `apply_latent_rotation`/`impose_rotation` code stays in the
library as a tested-but-unused capability; the paper does not feature it.)

## 2. Cross-cutting methodology

- **SNR conditions span the neural regime $\{-3, 0, 3, 6, 8\}$ dB** ($n\approx15/30/50/150/250$).
  Emphasis is on $-3/0/3$ dB; $6/8$ dB are the easy end (shown for completeness / to make the
  "high SNR is easy" point).
- Report mean $\pm$ s.e. over **paired seeds** (E1 uses 8 streams).
- Forecast: $k$ to 200, report full $k$-curve + AUC; do not over-read "largest $k$ with $R^2\ge0$".
- Drift metrics (principal angle, per-refresh rotation) are kept only as **diagnostics** (they show
  the online subspace barely rotates -- 0.6 deg cumulative -- which is *why* flow rotation is moot),
  not as a causal claim.

## 3. Experiments

### E1 -- readout stability: when does the online readout fall short, and how to fix it (C2, C3-revised)
High SNR ($n=250$/$8$ dB) **and** low SNR ($n\le50$) so the regime dependence is explicit. Arms
(already run, 8 seeds): **proj+oracle** (ceiling), **online** (refresh every $K$), **frozen PCA**
(good fixed init, $60n$ window), **freeze-after-$S$** (stop rewriting once converged).
Result (high SNR): one-step $R^2$ / k-AUC -- oracle $0.97/0.95$, frozen-PCA $0.97/0.82$,
freeze-after $0.72/0.63$, online $0.69/0.30$. Freezing roughly doubles forecast AUC; a good fixed
readout is near-oracle. **Decision:** report freeze/anneal as the practical recipe; quantify the
low- vs high-SNR gap to online and to frozen-PCA. (Dropped arms: online_track, oracle_imposed,
oracle_imposed_track.)

### E2 -- Refresh-interval $K$ sweep (adaptation vs stability, by SNR)
$K\in\{250,1000,4000,\text{frozen}\}$ at low and high SNR (done). Low SNR: smaller $K$ adapts faster;
high SNR: larger $K$ / frozen gives better dynamics. Frames the freeze/anneal recipe and shows the
**low-SNR regime is where adaptation matters**.

### E3 -- Smoothing $\tau$ (C4), low-SNR emphasis
$\tau\in\{1,2,4,8,16,32\}$ across the 5 SNR levels, online + proj+oracle control (done/finishing).
Report rate NLL / rate corr + filtered $R^2$ + lag; the smoothing benefit should be **largest at
low SNR**.

### E5 -- Real-time timing (C5)
p50/p95/max per-bin latency + refresh-bin vs ordinary-bin, from the uncontended serial runs.
Pass: p95 < 5 ms/bin.

### E4 -- Projection mechanism ablation (optional, C1) [DROPPED 2026-06-09]
Would have been: full-$n$ vs random vs shuffled-$C$ vs PCA vs $\vC^+$ projection, performance vs
subspace angle, at low SNR, to sharpen *why* the projection helps. **Dropped:** not required for the
story; C1 is already carried by the projection-vs-oracle rows in Tab 1/2 and Fig A.

### E6 -- Original VJF fails to converge (motivating figure, Sec 2 -> 3) [DONE]
Demonstrate the *problem* before the fix: original VJF (raw-spike encoder; readout $\vC$ learned by
the SGD/Adam ELBO gradient; flow by SGD; NO readout warm-start) on the synthetic stream. Log
filtered-latent $R^2$ and one-step prediction $R^2$ vs stream position (x = time step, y = $R^2$).
Expectation: both stay near zero (the readout collapses to the mean rate under sparse spikes), so
the unknown-readout problem is fully motivated before introducing the projection encoder + decoupled
readout. Self-contained (vjf.synthetic, flow_learner='sgd'); a single representative SNR (e.g. 3 dB).
**Done** (`motivation_compare.py`, 5 seeds): 4 original arms {random, oracle $\vC$} x {Adam, SGD} +
sVJF -> `data/extra/motivation_compare.json` -> Fig 1 (`motivation.pdf`). Random-init stalls (one-step
$R^2$ 0.2-0.4); oracle-init converges (~0.94) and does not drift back; sVJF reaches ~0.93 from spikes.

### E7 -- Flow learner: SGD/Adam vs square-root RLS (numerical comparison) [DONE]
Isolate the flow learner. Identical setup (projection encoder, a fixed/oracle readout so only the
flow differs), vary ONLY how $\vW$ is learned: SGD/Adam (the original VJF) vs square-root RLS (sVJF).
Compare online convergence speed and stability over the long stream: filtered-latent $R^2$ and
one-step $R^2$ vs stream position, plus wall-time to a target $R^2$. Expectation: square-root RLS
converges far faster (W enters linearly) and stays stable, motivating the srrls contribution
(\cref{ssec:srrls}). Note: `flow_learner` in {'sgd','srrls'}; the gradient optimizer is now
selectable via `VJF.make_model(..., optimizer='adam'|'sgd')` (Adam restored). The original VJF used
Adam (it had been swapped to SGD on 2021-07-27, commits cda9668/d2e687e, during a refactor), so the
exact-original arm is `flow_learner='sgd', optimizer='adam'`.
**Done** (`flow_compare.py`, oracle readout fixed, 5 seeds) -> `data/extra/flow_compare.json` ->
Appendix `fig:flow`. With a clean readout the flow learner barely affects accuracy (one-step $R^2$
0.88-0.97 for srrls/Adam/SGD); srrls converges fastest early and stays stable, plain RLS diverges.

## 4. Figure plan (claims-first)

- Fig A (C1): metrics by method x **5 SNR** (the low-SNR win is the headline).
- Fig B (C2): per-method convergence curves (low vs high SNR).
- Fig C (C3-revised): E1 readout stability -- one-step R^2 / k-AUC for oracle / frozen-PCA /
  freeze-after / online, high vs low SNR.
- Fig D (C4): E3 tau trend (low-SNR emphasis).
- Fig E (C5): E5 per-bin latency (p50/p95/max, refresh vs ordinary).
- (optional) E2 K-sweep; E4 mechanism.
- Appendix: raster, single-trial forecasts, k-step.

## 4b. Figure production standards
Vector PDF, proper font sizes at final print size (no LaTeX downscaling), `pdf.fonttype=42`,
consistent per-method/per-SNR colors, regenerated from `summary.json` by one plotting module.

## 5. Status / logistics
- **All planned experiments executed except E4 (dropped 2026-06-09).** E1 done (8 seeds + SNR-sweep),
  E2 done, E3 done, E5 done, E6 done (Fig 1), E7 done (fig:flow). Paper has 0 unfilled `\TD{}`; all
  figures staged. This v5 set is complete -- next set planned separately.
- Code on `exp/better-experiments` (pushed). Tracker: `EXPERIMENT_PROGRESS.md`.

## 6. Final delivery (standing)
Regenerate PDF figures -> update `paper/main.tex` (remove frame-churn/flow-rotation; reframe around
low-SNR value + freeze/anneal recipe) -> `/codex-review` -> academic-editor proofread -> revise ->
push to GitHub -> post the PDF to Slack #joint-filtering.
