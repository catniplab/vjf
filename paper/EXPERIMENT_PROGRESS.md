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
| E5  | C5 timing (dedicated VM) | - | - | - | not started | - | - | - |
| E2  | K sweep (low/high SNR) | - | - | - | not started | - | - | - |
| E3  | tau sweep x 5 SNR | - | - | - | not started | - | - | - |
| E4  | projection mechanism ablation | - | - | - | not started | - | - | - |

## Instance allocation (when sharding, up to 4)
- inst-1: ...
- inst-2: ...
- inst-3: ...
- inst-4: ...
- (E5 timing runs alone on its own VM, no co-tenancy)

## Log
- (chronological notes: launches, pulls, teardowns, failures, decisions)

## Pending decisions / blockers
- Awaiting approval to implement E0 instrumentation + E1 code (subspace-tracking, imposed-rotation
  injector, decoder-freeze) and to start GCP runs.
