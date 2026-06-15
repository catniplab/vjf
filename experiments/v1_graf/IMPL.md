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

### UPDATE 2026-06-10b (scaled to single direction + growing RBF). See REPORT_single_dir_and_growth.md

- Scaled the task to ONE direction (auto 225 deg, L=2, replay trials over epochs): sVJF WORKS --
  leave-one-neuron-out PLL +0.61..+0.65 (PSTH ceiling +0.73), forecast +0.30..+0.42, stable. So the
  M1 no-go was multi-condition scaling, not an inability to model V1. (PLL is just below the PSTH
  ceiling -> captures stimulus-locked structure but does not yet beat the trial mean.)
- Training-curve diagnostics: readout C subspace is STABLE (cumulative 2.4 deg / 50 epochs); the
  drift is LATENT-SCALE INFLATION (spread 0.07->0.35, still rising), dragging the latent ~27x off the
  once-seeded RBF centers. (Not a free gauge -- C is column-normalized.)
- Implemented Memming's growing RBF (vjf grow_rbf; commits efdd11f + codex fix 92f5f64): coverage is
  fixed (nearest-center dist 0.07 vs 0.35) BUT it bursts to the cap (100->400 in ~1 epoch, driven by
  the inflation), PLL is UNCHANGED (~0.65), and forecast is WORSE (0.15..0.37 vs 0.30..0.42). So RBF
  coverage is NOT the bottleneck; the 400-center basis fits the inflating spiral and degrades the
  free-run.
- Net: growing RBF is a sound, tested, reviewed capability (kept, off by default) but not the fix
  here. Root cause to tackle next = latent-scale control (inflates single-dir, collapsed full-task);
  beating the PSTH ceiling likely needs readout/encoder work (online readout subspace ~46 deg off
  the data PCA-3). Two dropped hypotheses: "oscillator dominates" (refuted) and "pin latent scale"
  (not a coherent separate knob, per Memming).
- Inflation cause pinned (`inflation_probe.py`): the single-direction latent inflation needs BOTH the
  readout refresh AND the dynamics -- freezing the readout (no refresh) keeps spread flat at 0.034 and
  dynamics-off at 0.045, vs baseline 0.187. It is the CCIPCA sqrt(eigenvalue) rescaling x dynamics
  feedback ratchet. Concrete next lever: FREEZE/ANNEAL the readout refresh after warm-start (already a
  synthetic-study recipe) and re-check PLL/forecast -- a config change, no new machinery.
- Freeze test done (`single_dir.py --refresh-k huge --tag frozen`): freezing C from warm-start is NOT
  a clean win -- PLL DROPS (0.10/0.47/0.52 vs 0.61/0.65/0.65; the 8-trial warm-start C is undertrained
  and the refresh was improving it) while forecast is mixed (E=20 better 0.60 vs 0.30; E=50 worse).
  So the refresh is double-edged (improves C -> PLL; inflates -> forecast erosion). The lever is
  freeze/ANNEAL AFTER C converges (freeze-after-S), NOT from the start -- needs a small freeze-after
  mechanism in OnlineReadout/driver (not yet built). That is the recommended next step.

### UPDATE 2026-06-14 -- forecasting-criterion hyperparameter search (single dir 225)

Switched the model-selection criterion to the gold standard: **future forecasted reconstruction** --
free-run the flow from a time t0 in a held-out trial (no obs after t0), decode, and score the
forecast against the FUTURE spikes by **Poisson-deviance skill** vs two baselines (persistence;
the stimulus-locked PSTH), gated by a decent filtered leave-one-neuron PLL. Selection is on a
VALIDATION split (30 train / 10 val / 10 test reserved), randomized interleaved replay over epochs,
weighted S = 0.5*s8 + 0.3*s16 + 0.2*s32 (k=8/16/32 bins = half/one/two grating cycles). Selection
ranks by persistence-skill; PSTH skill is reported as a phase-aware near-oracle REFERENCE. Code:
`search.py` (sharded over 8 parallel /gcp_run VMs), eval in `eval.py`
(`forecast_reconstruction_deviance` + `forecast_skill_summary`). The old latent-space affine
`forecast_r2` is kept as a (gameable) diagnostic only.

Result (selection on val):
- ALL configs reconstruct well: filtered leave-one-neuron PLL meets/exceeds the PSTH ceiling (best
  ~0.716 vs ceiling 0.683). WITH observations the model is at/above the trial mean.
- Free-run forecast beats persistence MARGINALLY and loses to the PSTH at every horizon:
  base (72 cfg) best S_persist +0.030 / S_psth -0.029 (L=4, 1600 ctr, E=100); xl capacity-push
  best so far S_persist +0.036 / S_psth -0.037 (L=5, 2400 ctr, E=100).
- Capacity is NOT the bottleneck (ceiling reached): L=6 and E=200 do NOT help -- near-zero/worse
  than L=4/5 at E=100. The +0.03..+0.036 persistence gain plateaus and the PSTH gap does not close.
  For stimulus-locked gratings the PSTH is a phase-aware near-oracle, and the autonomous flow does
  not propagate trial-specific structure beyond it.

Wall clock (like the M1 report; per-config wall is recorded as `elapsed_s` in each shard JSON):
- BASE 72 configs, 8x e2-standard-2 (europe-west1-b): per-config min 385s / median 955s (~16 min) /
  max 2112s (~35 min); ~21.5 VM-h total; ~2.7 h wall-clock across the 8 VMs (launched 11:55, merged
  ~14:55, 2026-06-14).
- XL 48 configs (L in {4,5,6}, RBF cap {1600,2400}, E in {100,200}): per-config median ~2118s
  (~35 min), max ~4047s (~67 min, the L=6/E=200 configs); ~6 h wall-clock (launched 16:41). The
  heaviest L=6/max2400/E200 configs cost ~65-80 min EACH.
- Cost driver: the leave-one-neuron PLL eval (74 neurons x 10 val trials x 140 bins, independent of
  epochs) PLUS L>=5 / E>=100 training; a GPU does not help (per-bin CPU online loop). NOTE for
  multi-dir: per-config cost grows steeply with L and basis size -- budget accordingly.
- Operational note: the 8-VM fleet driver hung on the launch-ssh for 4/8 shards (the
  `nohup & disown` ssh channel did not close); compute was unaffected (search ran detached) and was
  recovered by `gcp_runs/graf_xl_recover.sh` (poll -> pull -> teardown -> merge). Fix for next time:
  detach the remote launch with `setsid`/`ssh -n` so the channel closes cleanly.

### UPDATE 2026-06-15 -- curvature smoothness penalty on the SGD flow (NEW equation term, Memming-approved)

The best forecast config free-runs as shrinkage-to-center + high-frequency chaos (3D video):
the SGD/Adam flow's one-step map is UNDER-REGULARIZED (no penalty on the flow weights; the RLS
path has `rls_ridge`, the SGD path had no analog), and `dyn_noise`'s contraction objective
over-collapses the cycle. Fix (Memming chose R2 + backing off denoising):

**Velocity field (SGD flow):** `v(x) = sum_j phi_j(x) W_{j,:}`, `phi_j(x) = exp(-||x-c_j||^2 / (2 w_j^2))`,
`w_j = exp(logwidth_j)` (vjf/functional.py:rbf, vjf/module.py LinearRegression, bayes=False).

**R2 curvature penalty (added to the SGD-flow ELBO loss):**
`d^2 phi_j/dx dx^T = phi_j [ (x-c_j)(x-c_j)^T / w_j^4 - I / w_j^2 ]`, so
`H_a(x) = d^2 v_a/dx dx^T = sum_j W_{j,a} phi_j(x) [ (x-c_j)(x-c_j)^T / w_j^4 - I / w_j^2 ]`, and
`R2(x) = sum_a ||H_a(x)||_F^2`. Loss: `L = L_ELBO + lambda_smooth * mean_t R2(x_t)`, evaluated at
the filtered mean each SGD step; `lambda_smooth = 0` recovers the original VJF. Penalizes only
NON-LINEAR curvature, so an affine/rotational field (the limit cycle) is unpenalized while the
high-frequency wiggle is killed. The `gaussian_loss`/ELBO terms are UNCHANGED -- this only adds a
regularizer term. SGD/Adam flow only (the RLS path keeps `rls_ridge`). Paired with reduced
`dyn_noise` (remove the over-contraction). Re-search over `lambda_smooth` follows.
