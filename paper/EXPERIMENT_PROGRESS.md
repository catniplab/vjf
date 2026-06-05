# Experiment progress tracker

Single source of truth for the run state of the experiments in `EXPERIMENT_PLAN.md` (v4).
Target: finish within 24 h wall-clock; shard across up to 4 GCP instances if a single VM would
exceed that. Update on each launch and each pull.

## Status legend
queued / running / done / failed. Code is pinned to a single commit per run (git mode).

## Runs

| exp | condition | VM | commit | seed(s) | status | wall | results path | key numbers |
|-----|-----------|----|--------|---------|--------|------|--------------|-------------|
| E0  | instrumentation (drift logging) | - | 491d77c | - | **code done** (maybe_refresh returns drift metrics) | - | - | equivariance unit-tested |
| E1  | C3 causal: 7 arms (oracle/online±track/imposed±track/frozen/freeze) | exp-20260605-171155-vjf-e1 | 7c8b585 | 20260605..12 (8) | **running** (launched 16:17 UTC, ~10h) | - | results_e1/ | 8 seeds x 7 arms; early arm-set ~1.3h |
| E5  | C5 timing | (from E2/E3 uncontended serial runs) | 1fc65e6 | - | covered (p50/p95/max + refresh-vs-ordinary in summary.json) | - | - | no separate VM needed |
| E2  | K sweep {250,1000,4000,1e5} x 5 SNR | exp-20260605-192841-vjf-e2 | 1fc65e6 | 20260602 | **running** (launched 19:29 UTC, ~2-3h) | - | results/e2_K*/ | per-bin percentiles included (E5) |
| E3  | tau {1,2,4,8,16,32} x 5 SNR, online + oracle ctrl | exp-20260605-192819-vjf-e3 | 1fc65e6 | 20260602 | **running** (launched 19:28 UTC, ~6h) | - | results/e3_*/ | rate corr + filtered R^2 vs tau |
| E4  | projection mechanism ablation | - | - | - | deferred (optional Fig F; needs encoder-feature modes) | - | - | run if budget allows after E1-E3 |

## Instance allocation (3 of 4 used)
- inst-1: E1 (exp-20260605-171155-vjf-e1) -- 8 seeds x 7 arms
- inst-2: E3 (exp-20260605-192819-vjf-e3) -- tau sweep online + oracle control
- inst-3: E2 (exp-20260605-192841-vjf-e2) -- K sweep
- inst-4: free (reserve for E4 or reruns)
- E5 timing: extracted from E2/E3 serial runs (uncontended), not a separate VM

## Log
- (chronological notes: launches, pulls, teardowns, failures, decisions)

## Final delivery (when all experiments done) -- standing instruction

1. Regenerate all figures as publication-quality vector PDF from the run artifacts.
2. Update the paper (`paper/main.tex`) with the new results/figures (C3 causal evidence, C4 trend, C5 timing).
3. `/codex-review` the updated paper; address findings.
4. Proofread/edit pass (academic_editor + writing-style CCC/detailed); revise.
5. Rebuild; verify clean.
6. Push branch(es) to GitHub.
7. Post the compiled PDF tech report to Slack #joint-filtering.

## Pending decisions / blockers
- Awaiting approval to implement E0 instrumentation + E1 code (subspace-tracking, imposed-rotation
  injector, decoder-freeze) and to start GCP runs.
