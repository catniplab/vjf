# sVJF on Graf V1 -- progress & findings

See `DESIGN.md` (approved design) and `PLAN_M1.md` (M1 implementation plan).

## M1 (array_5 @ 10 ms, L=3) -- 2026-06-10: NO-GO as configured

First full run on the strongest array. Verdict: the pipeline runs, is stable and real-time,
but the latent does NOT capture the stimulus -- a clear no-go for the autonomous-shared-oscillator
configuration. Detail below.

### Numbers (L=3, N=74 well-tuned, 2880 train / 720 test trials, single online pass)
- Leave-one-neuron-out PLL = **-3.50 bits/spike** (worse than a homogeneous-Poisson baseline).
- Single-trial orientation decode from the filtered latent = **0.014** (chance = 1/72 = 0.0139).
- Free-run forecast R^2 = **0.31** (the autonomous flow DID learn a cycle).
- Timing: median **2.72 ms/bin**, p95 3.30 ms/bin (real-time at 10 ms bins). Diverged = 0.
- Run: ~3.5 h on an e2-standard-4 (CPU); leave-one-neuron-out PLL dominates the cost.
- Result JSON: `gcp_runs/exp-20260609-221137-vjf-v1graf/results/m1_array5_bin10.0_L3.json`.
  (L=4 was NOT run -- a latent-dim bump won't fix a dynamics-dominance issue; budget saved.)

### Diagnosis: sVJF's filtering destroys the orientation information (it is present upstream)
Local control (`diag_pca_decode.py`, no VJF) -- decode grating direction (72-way) from the
per-trial stimulus-window mean of the link-matched log-rate feature:
- raw mean counts (74-d): **0.338**; full log-rate feature: **0.297**; **PCA-3 of the feature: 0.234**.
- sVJF filtered latent: **0.014**.

So orientation is strongly present in the population feature, and even a 3-D PCA (the kind of
subspace the projection encoder's `C` captures) preserves it (0.234). sVJF's filtered latent gives
chance. Therefore the readout/feature is fine; the VJF **filtering** (recognition + dynamics) is
losing the orientation. PLL = -3.5 (rates swing wrongly, not "predict the mean") + forecast R^2 = 0.31
(a learned cycle) are consistent with the **autonomous shared oscillator dominating the posterior**:
the latent free-runs on the learned limit cycle and ignores the orientation-carrying projection
input. Trial-averaging then cancels the uncoupled phase, leaving a tiny (~1e-3) torus with the 3rd
dim dead.

This is the design risk noted at brainstorming: with per-trial reset-to-zero and a single shared
flow + readout, orientation must be injected each bin by the encoder; a self-sustained oscillator
cannot itself represent 72 distinct orientations, and here it overrides the encoder.

### Likely causes (to disambiguate next)
1. Dynamics-dominance: the ELBO dynamics term outweighs the observation/recognition term, so the
   posterior follows the flow rather than the projection. (Prime suspect.)
2. The autonomous-shared-oscillator framing itself: orientation needs a (quasi-static) latent
   coordinate the encoder can set; a single oscillator + reset-to-zero does not provide one.
3. A subtler inference/wiring bug that makes the filtered latent uninformative despite a valid
   projection (less likely -- smoke + reviews verified the projection is fed and read).

### Recommended next steps (decision -- involves the design, loop in Hyungju + Yuan)
- Cheap, no compute: `/codex-review` the eval + driver inference path to rule out cause 3.
- Confirming control (cheap, local/short): run sVJF with the dynamics OFF / frozen (recognition +
  projection only, no autonomous flow) and decode the latent. If decode jumps to ~0.23, the flow is
  the culprit (cause 1/2 confirmed). Also dump filtered latents to verify they don't separate by
  direction.
- If confirmed: design fix -- give orientation a slow/quasi-static latent axis the encoder sets, or
  rebalance the ELBO so the observation constrains the latent, or feed stimulus as input. This is a
  design change to discuss before re-running.

**M2 (5/1 ms) and the baseline plan (U5) are on hold pending this resolution.** The implementation
(Tasks 1-7, branch `feat/v1-graf`) is sound and reviewed; the issue is configuration/design, not code.

### UPDATE 2026-06-10 (diagnostic run + controls -> first hypothesis REFUTED)

A diagnostic retrain (`diag_m1.py`: dynamics-ON vs a no-dynamics control) + a readout-projection
localization (`diag_readout.py`) overturn the "oscillator dominates" read above. Full write-up with
figures: `report_m1/REPORT_M1.md`. Corrected diagnosis -- the orientation is lost in TWO encoder
stages, and the dynamics are exonerated:
- batch PCA-3 of the feature decodes direction at **0.234** (information is present);
- the readout projection pi (recognition INPUT) decodes at **0.126** (warm-start C) / 0.103 (online
  C) -- the online readout subspace is **46.5 deg** off the ideal, ~halving the info;
- the sVJF filtered latent (recognition OUTPUT) decodes at **chance (0.014)** -- and the
  **no-dynamics control is also 0.014**, so the autonomous flow is NOT the cause (with the flow ON the
  trial-averaged latent even shows a faint orientation ring the control lacks).
- Root cause: **Poisson encoder collapse** -- the recognition network drives the posterior to a
  near-constant, collapsed latent, discarding the orientation in pi (the projection encoder reduced
  but did not remove the documented collapse). Secondary: the readout subspace is suboptimal.
- Next (decision): attack the encoder collapse (variance floor / entropy temper / recognition
  warm-up or pre-train pi->latent), and improve the readout subspace (more coverage / longer
  warm-start). Do NOT change the dynamics. See REPORT_M1.md for specifics.
