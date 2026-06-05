# IMPL: real-time online VJF framework + tutorial + tech report

Branch `feat/realtime-framework` off `master`. Plan:
`~/.claude/plans/jolly-twirling-sifakis.md`. Each unit is `/codex-review`'d before commit.

## Units

- [x] **1. Library** - `vjf/realtime.py` (`online_filter` generator + `OnlineResult`) and
  `vjf/synthetic.py` (self-contained limit-cycle Poisson stream). Additive; reuses
  `VJF.filter`, `OnlineReadout`, `RBFDS.initialize`. codex-review done (8 findings; 1,3-8
  fixed, #2 partial - refresh moved after the accept guard; full state-rollback declined as
  it deviates from the validated `run_condition`).
- [x] **2. Tests** - `test/test_realtime.py` (10 tests: raw-spike equivalence + fields,
  projection supplies valid y_enc, projection-without-readout raises, fixed readout no drift,
  refresh after t0, warm-up boundary init-once + arg alignment, divergence exception + magnitude
  guard paths + q reset, determinism across boundary, synthetic calibration/reproducibility).
  codex-reviewed (coverage gaps closed; finite-warmup equivalence declined - covered by the
  boundary arg-alignment test). 24 tests pass.
- [ ] **3. Tutorial script** - `script/realtime_tutorial.py` + smoke test.
- [ ] **4. Notebook** - `notebook/realtime_tutorial.ipynb` (from the script).
- [ ] **5. Tech report** - `paper/` (VJF-paper form; technical, difference-focused).

## Key design notes

- `online_filter(model, stream, *, readout=None, u_stream=None, adapt_readout=True,
  warmup_steps, rbf_width_scale, logvar_floor, max_abs_state)` - a generator yielding one
  `OnlineResult` per sample. Encapsulates: posterior threading; projection readout
  (`feature/update/project` + Procrustes `maybe_refresh`, refresh only on accepted steps,
  not at t=0); one-time dynamics init at the warm-up boundary (buffers latent means +
  controls, passes `u[1:]`); divergence recovery (catch non-finite, logvar floor,
  repeat-last mean, reset q); per-step timing. No new math.
- `adapt_readout=False` + `OnlineReadout.set_fixed` -> fixed/oracle projection (no drift).
- Refresh cadence is owned by `OnlineReadout.K` (set at construction), not a loop param.
- `vjf/synthetic.poisson_readout` uses a simple analytic gain calibration (NOT Fisher-SNR);
  validates rate targets and resamples all-zero loading rows.

## Verification

- `uv run pytest test/` green; `uv run python script/realtime_tutorial.py` runs in seconds;
  notebook executes; `cd paper && make` builds with TEXINPUTS set.
