# Experiment progress tracker

Single source of truth for the run state of the experiments in `EXPERIMENT_PLAN.md` (v5).
Target: finish within 24 h wall-clock; shard across up to 4 GCP instances if a single VM would
exceed that. Update on each launch and each pull.

## Status legend
queued / running / done / failed. Code is pinned to a single commit per run (git mode).

## Runs

| exp | condition | VM | commit | seed(s) | status | wall | results path | key numbers |
|-----|-----------|----|--------|---------|--------|------|--------------|-------------|
| E0  | instrumentation (drift logging) | - | 491d77c | - | **code done** (maybe_refresh returns drift metrics) | - | - | equivariance unit-tested |
| E1  | readout stability (kept arms: oracle/online/frozen-pca/freeze-after) | exp-...-vjf-e1 (deleted) | 7c8b585 | 20260605..12 (8) | **DONE** (56/56 pulled) | ~6.7h | gcp_runs/exp-20260605-171155-vjf-e1/results | high-SNR onestep/kAUC: oracle .97/.95, frozenPCA .97/.82, freeze-after .72/.63, online .69/.30. **flow-rotation refuted (online_track==online_base; real drift 0.6deg cumulative); imposed control collapsed. Arms online_track/oracle_imposed* DROPPED.** |
| E5  | C5 timing | (from E2/E3 uncontended serial runs) | 1fc65e6 | - | covered (p50/p95/max + refresh-vs-ordinary in summary.json) | - | - | no separate VM needed |
| E2  | K sweep {250,1000,4000,1e5} x 5 SNR | exp-20260605-192841-vjf-e2 | 1fc65e6 | 20260602 | **running** (launched 19:29 UTC, ~2-3h) | - | results/e2_K*/ | per-bin percentiles included (E5) |
| E3  | tau {1,2,4,8,16,32} x 5 SNR, online + oracle ctrl | exp-20260605-192819-vjf-e3 | 1fc65e6 | 20260602 | **running** (launched 19:28 UTC, ~6h) | - | results/e3_*/ | rate corr + filtered R^2 vs tau |
| E4  | projection mechanism ablation | - | - | - | **DROPPED** (2026-06-09; optional) | - | - | not required; C1 carried by Tab 1/2 + Fig A |
| E6  | original VJF fails to converge (Fig 1) | local (self-contained) | - | 20260605..09 (5) | **DONE** | - | data/extra/motivation_compare.json | oracle-init ~0.94 (no drift), random-init 0.2-0.4, sVJF ~0.93 -> motivation.pdf |
| E7  | flow learner: SGD/Adam vs srrls | local (self-contained) | - | 5 | **DONE** | - | data/extra/flow_compare.json | one-step R^2 0.88-0.97 (stable arms); plain RLS diverges -> fig:flow |

## Instance allocation (3 of 4 used)
- inst-1: E1 (exp-20260605-171155-vjf-e1) -- 8 seeds x 7 arms
- inst-2: E3 (exp-20260605-192819-vjf-e3) -- tau sweep online + oracle control
- inst-3: E2 (exp-20260605-192841-vjf-e2) -- K sweep
- inst-4: free (E4 dropped; was its reserve)
- E5 timing: extracted from E2/E3 serial runs (uncontended), not a separate VM

## Log
- (chronological notes: launches, pulls, teardowns, failures, decisions)
- **2026-06-07 (resumed):** two VMs in flight, code pinned (git mode), poller `gcp_runs/poller.sh`
  auto-pulls + tears down each VM when its orchestrator (`run_main.sh`/`run_extra.sh`) exits.
  - **MAIN** `exp-20260607-094255-vjf-main`: `run_main.sh` = 3 seeds {20260605,06,07} x 4 modes
    {spikeoracle, projoracle, online(pca_proj_online), frozenpca}, `experiment.py --proj-tau 8`,
    5-SNR each, ~50 min/invocation -> ~10 h total; started 08:44 UTC. Feeds Tab 1 (r2_final),
    Tab 2 (onestep_r2), Fig A (summary.pdf), Fig B (curves.pdf), and the 5-SNR raster.
  - **EXTRA** `exp-20260607-100741-vjf-extra`: `run_extra.sh` = timing run (x_timing, DONE,
    staged to `paper/data/extra/x_timing.json`) + low-SNR readout-stability (`exp_e1.py`,
    n=30, 0 dB, 3 seeds x 4 arms). Feeds Fig E (timing, DONE) and the LOW panel of Fig C.
  - Staging map: `results/mo_<mode>_s<seed>/summary.json` -> `paper/data/main/mo_<mode>_s<seed>.json`;
    `results_e1/*n30*.json` -> `paper/data/e1_low/`; `results/x_timing/summary.json` ->
    `paper/data/extra/x_timing.json`.
- **Paper rewrite (done, structural):** `main.tex` reframed to v5 plan -- sVJF naming throughout,
  frame-churn/flow-rotation DROPPED, low-SNR framing, two-timescale + warm-start readout +
  freeze/anneal recipe (freeze optional; non-stationarity detection = future work), conceptual
  `fig:timescales` placed by Algorithm 2, 5-SNR experiments, new figure set
  (summary/curves/readout_stability/tau_sweep/timing), Discussion rewritten. Builds clean (11 pp).
- **Numbers:** `paper/figs_src/fill_numbers.py` computes every cited value from the staged JSONs.
  FILLED now (final data): C3 high-SNR readout schedule (drift 0.2 deg; frozenPCA onestep 0.97 vs
  oracle 0.97; slow/locked k-AUC 0.63 vs fast 0.30) and C5 timing (median ~2.6 ms; p95<3.2 ms for
  4/5, 5.5 ms at n=50). PENDING (45 `\TD{}` left): Tab 1/Tab 2 (40 cells) + C1/C2 inline -- need
  MAIN; Fig C low panel + raster -- need EXTRA/MAIN.
- **Quiver warning fix:** `experiment.py:438` now skips the velocity-field quiver when the field is
  ~0 (early snapshot, flow still ~identity) -- removes a matplotlib autoscale divide-by-zero in the
  diagnostic montage; metrics were always intact (spikeoracle R^2 monotone in SNR). Plotting-only.
- **Timing spike investigation (resolved):** the `x_timing` run showed n=50 (3 dB) with p95=5.5 ms
  and ~8% of bins elevated, while n=15/30/150/250 were clean (p95<3.2 ms). Probed on a CLEAN
  dedicated e2-standard-4 (`gc_probe.py`, gc on/off, n=50 & n=250, torch default threads):
  - The n=50 anomaly did NOT reproduce (7.32% -> 0.00% of bins >2x median; p95 5.49 -> 3.58 ms)
    => it was noisy-neighbor VM contention during that serial segment, not algorithmic.
  - NOT garbage collection: gc fires ~1x / 20000 bins (torch refcounts free everything); 0 slow
    bins coincided with a gc stop; `gc.disable()` did not help (max slightly worse 5.60 -> 6.11).
  - Clean-VM distribution is tight: median ~3.1 ms, p95 ~3.5 ms, p99 ~4.2 ms (under the 5 ms
    budget), flat across n=50 and n=250.
  - ACTION: re-running the full 5-SNR timing on a fresh dedicated VM (`run_timing.sh`,
    VM `*-vjf-timing`) to replace the contended `x_timing.json`; paper C5 reframed to
    unoptimized-reference + flat-in-n; drop the "one condition elevated tail" once clean data lands.
- **Forecast metric change (per Memming):** dropped the K-dependent k-AUC. New metric =
  **relative half-skill forecast horizon** = first lead k with R2_fc(k) < 0.5 * R2_filter, reported
  as a TIME in seconds (general predictability time; periods shown as a secondary axis since
  "cycles" only applies to limit cycles). 1 period ~ 0.21 s. K-invariant; threshold relative to the
  achievable filtering accuracy (so not degenerate at low SNR). High SNR: oracle >1.0s (censored),
  frozen 0.95s, slow/locked 0.92s, fast online 0.41s; low SNR ~0 for all (forecasting fails at
  0 dB). Readout figure reworked to 2-row (one-step R2 + horizon) x (high/low). Supplementary
  Appendix figure `fig:kstep` = full R2(k) decay curves with horizon dots.
- **All latent R2 use a best-fit AFFINE alignment** (rotation+scale+reflection+offset; fit once per
  run, not per prediction) -> absorbs the factor/linear invariance. Stated explicitly in Experiments.
- **"oracle" = oracle READOUT (true C) only**, never the dynamics (always learned online). Labeled
  "oracle C" in figures, "oracle $\vC$" in tables, with an explicit Experiments sentence.
- **SNR clarified:** it is the state-estimation SNR (Fisher info of x_t given y_t), NOT the dynamics
  SNR (identifiability of f), which is distinct and unmeasured. Noted in Experiments.
- **Raster:** dedicated self-contained `fig_raster` (5 SNR, panel heights proportional to n, neurons
  sorted by preferred phase, shared latent in a top panel, proper vector fonts, no x1/x2 legend).
  Replaces the stale 3-SNR PNG; writes `figs/raster.pdf`.
- **Timing fully diagnosed (NOT GC):** median ~2.8 ms single-core, flat in n; the tail is
  intermittent host contention (time-correlated, hits the ~3rd condition / ~6-9 min into the VM;
  moves across n run-to-run; 0 divergences, ~0 refresh, gc.disable() no help, denormal flush no
  help). Figure switched violin -> box (IQR + median + 5-95% whisker) so the rare excursion is not
  over-weighted. C5 reports median + host-jitter caveat.
- **E1 SNR-sweep (begmcnne2, VM `*-vjf-e1sweep`):** fills the MISSING readout-schedule regimes
  (-3/3/6 dB; n=15/50/150; 3 seeds x 4 arms). First attempt FAILED (neurofishersnr not on PyPI);
  fixed -> install from `git+https://github.com/catniplab/neurofisherSNR.git@96e83f68025db204a5dc6fde1a74403a045e61fb`
  (the commit the prior VMs used) with a hard import gate. When it lands -> stage all E1 readout
  files (n=15/30/50/150/250) and reshape the readout figure to horizon-vs-SNR + one-step-R2-vs-SNR.
  GCP LESSON: neurofisherSNR is a private catniplab git package, install via that git URL.
- **Remaining to ship:** when MAIN + (E1-sweep) land -> stage data, `fill_numbers.py`, fill `\TD{}`,
  regenerate figs (`plot_results.py`), pull 5-SNR raster, build -> `/codex-review` -> academic_editor
  + writing-style proofread -> revise -> build -> push -> Slack #joint-filtering.

- **2026-06-09 (reconciliation):** E4 (projection mechanism ablation) **DROPPED** -- optional, not
  required (C1 carried by Tab 1/2 + Fig A). E6 (motivation, Fig 1) and E7 (flow learner, fig:flow)
  confirmed **DONE** from self-contained local runs (`data/extra/motivation_compare.json`,
  `data/extra/flow_compare.json`). Paper has 0 unfilled `\TD{}`; all figures staged. The v5 set is
  complete and the plan/progress docs now match the shipped state. Next experiment set planned separately.

## Final delivery (when all experiments done) -- standing instruction

1. Regenerate all figures as publication-quality vector PDF from the run artifacts.
2. Update the paper (`paper/main.tex`) with the new results/figures (C3 causal evidence, C4 trend, C5 timing).
3. `/codex-review` the updated paper; address findings.
4. Proofread/edit pass (academic_editor + writing-style CCC/detailed); revise.
5. Rebuild; verify clean.
6. Push branch(es) to GitHub.
7. Post the compiled PDF tech report to Slack #joint-filtering.

## Pending decisions / blockers
- (resolved) E0 instrumentation + E1 code implemented and run; v5 experiment set complete (E4 dropped).
- Next: scope the next experiment set (real data / richer dynamics / non-stationarity trigger -- TBD).
