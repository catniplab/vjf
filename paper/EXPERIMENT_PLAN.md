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

### E4 -- Projection mechanism ablation (optional, C1)
If budget allows: full-$n$ vs random vs shuffled-$C$ vs PCA vs $\vC^+$ projection, performance vs
subspace angle, at low SNR. Sharpens *why* the projection helps. Not required for the story.

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
- E1 done (8 seeds, VM torn down). E2 done. E3 finishing (oracle control). E5 from serial runs.
- Code on `exp/better-experiments` (pushed). Tracker: `EXPERIMENT_PROGRESS.md`.

## 6. Final delivery (standing)
Regenerate PDF figures -> update `paper/main.tex` (remove frame-churn/flow-rotation; reframe around
low-SNR value + freeze/anneal recipe) -> `/codex-review` -> academic-editor proofread -> revise ->
push to GitHub -> post the PDF to Slack #joint-filtering.
