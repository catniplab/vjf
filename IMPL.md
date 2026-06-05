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
- [x] **3. Tutorial script** - `examples/realtime_tutorial.py` (synthetic stream -> projection
  encoder + online readout -> `online_filter` -> trailing affine R^2 + per-bin timing + plots)
  and `test/test_tutorial.py` smoke test. (Lives in `examples/`, not `script/`/`notebook/`,
  which the repo gitignores.) Streams only the unseen remainder after the warm-up window (no
  look-ahead). codex-reviewed (faithfulness/test-rigor findings fixed). Full run: R^2~0.94,
  ~0.54 ms/bin.
- [x] **4. Notebook** - `examples/realtime_tutorial.ipynb` (self-contained narrative mirroring
  the script: synthetic stream + raster, projection model + warm-start, live online_filter loop,
  trailing-R^2 + phase-portrait plots). Executes top-to-bottom via nbconvert (R^2~0.92, no
  errors). codex-reviewed (gated the real-time claim on measured p95, softened the oracle-accuracy
  claim, added the script's init_w guard, labeled ground-truth diagnostics).
- [x] **5. Tech report** - `paper/main.tex` (+ Makefile, references.bib, .envrc, figs/). VJF-paper
  form (Intro / Background / Method [the 3 differences] / Experiments / Results / Discussion),
  technical and difference-focused, with the 5 validation figures. Builds clean via TEXINPUTS +
  latexmk (7 pp, 0 warnings). codex-reviewed (12 findings fixed: overclaim vs numbers scoped,
  Moore-Penrose C+, centered-feature renamed r_t vs control u_t, RBF centers xi_i vs loading c_i,
  covariance square-root factor wording, Procrustes prose softened, caption claims corrected).

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
