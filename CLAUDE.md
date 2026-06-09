# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VJF (Variational Joint Filtering): an online, single-pass variational learning
algorithm for nonlinear state-space models, learning latent dynamics and a readout
jointly from a streaming observation. Original method: Zhao & Park, *Variational
Online Learning of Neural Dynamics*, Front. Comput. Neurosci. 2020.

The active research thrust on `exp/better-experiments` / `feat/realtime-framework`
is **sVJF (stable VJF)** -- VJF made stable for long, low-SNR, real-time Poisson
streams with an *unknown* readout. sVJF is vanilla VJF plus three additive changes
(LaTeX macro `\svjf`):
1. square-root (Potter) RLS dynamics (`flow_learner='srrls'`),
2. a warm-started, two-timescale online readout (`vjf/readout.py`),
3. a projection encoder (`encoder='projection'`).
All three are backward-compatible additions; defaults reproduce the original VJF.

## Commands

Run Python through `uv` (never `pip`, never bare `python`).

```bash
# environment (one-time)
uv venv .venv
uv pip install --python .venv/bin/python -e .

# tests
uv run pytest test/                                   # full suite
uv run pytest test/test_realtime.py                   # one file
uv run pytest test/test_realtime.py::test_fixed_readout_no_drift   # one test

# tutorials (self-contained synthetic stream; runs in seconds)
uv run python examples/realtime_tutorial.py
# examples/realtime_tutorial.ipynb mirrors the script and executes top-to-bottom

# the validation experiment (long Poisson limit-cycle stream)
uv run python experiments/lc_poisson_stream/experiment.py            # full T=200000
uv run python experiments/lc_poisson_stream/experiment.py --quick    # smoke test
uv run python experiments/lc_poisson_stream/experiment.py --t-eff 20000
# extra deps for this experiment (e.g. neurofisherSNR, pinned by commit):
uv pip install --python .venv/bin/python -r experiments/lc_poisson_stream/requirements-exp.txt

# tech report
cd paper && make        # needs TEXINPUTS -> catniplab.sty (via paper/.envrc / direnv)
```

Long experiments run on GCP via the `/gcp_run` skill (git mode: the VM clones the
pushed branch). A per-timestep online loop is CPU-bound; a GPU does not help.

## Architecture

The forward pass is one ELBO filtering step (`VJF.forward` in `vjf/model.py`):
`recognition` (encode `y_t` -> posterior `q_t`) -> `transition` (predict, the RBF
flow) -> `decoder`/`likelihood` (decode `q_t` -> reconstruction). The loss
(`VJF.loss`) is `recon - entropy (+ dynamics)`; the dynamics term is dropped during
warm-up. Parameters are learned by a *hybrid* scheme each step: an SGD step on the
ELBO (`filter(..., sgd=True)`) plus closed-form non-gradient `update()` calls on the
likelihood noise and the dynamics weights.

- **`vjf/model.py`** -- the core. `VJF` (the nn.Module), built via the
  `VJF.make_model(ydim, xdim, udim, n_rbf, hidden_sizes, ...)` factory.
  `VJF.filter(y, u, qs, ..., y_enc=)` is the per-sample online primitive (threads
  the posterior); `VJF.fit(y, u)` is the batch/epoch loop with warm-up + convergence
  detection + one-time `transition.initialize`. `RBFDS` is the RBF dynamical system
  (the velocity/flow field, `x_t = x_{t-1} + RBF-velocity`). `LinearDecoder` is the
  affine observation model.
- **`vjf/module.py`** -- `RBF` basis and `LinearRegression` (the flow's weights).
  Three flow learners live here as separate update methods: `rls` (original online
  RLS; precision matrix blows up over long streams), `srrls` (square-root Potter RLS;
  PD by construction, stable over 200k+ steps, RLS-speed -- preferred for sVJF), and
  the `sgd` path (weights are an `nn.Parameter` trained by the ELBO; stable but slow).
- **`vjf/realtime.py`** -- `online_filter(model, stream, readout=, warmup_steps=, ...)`,
  a generator yielding one `OnlineResult` per sample. This is the sVJF real-time
  scaffolding: posterior threading, warm-up buffer + one-time dynamics init at the
  boundary, divergence recovery (logvar floor + repeat-last-mean + q reset), per-step
  timing, and wiring the readout. **No new math** -- it only calls existing
  `VJF`/`OnlineReadout` methods.
- **`vjf/readout.py`** -- `OnlineReadout`: streaming estimate of the loading `(C, b)`
  by CCIPCA on a causal, link-matched feature of the observations (decoupled from the
  ELBO gradient, which collapses under sparse low-rate Poisson). Supplies the
  projection encoder input `pinv(C)(feat - b)` and periodically Procrustes-anchors
  `(C, b)` into the decoder (`maybe_refresh` every `K` steps). Math is in
  `experiments/lc_poisson_stream/note_projection_encoder.tex`.
- **`vjf/synthetic.py`** -- dependency-free 2-D limit-cycle Poisson stream for
  tutorials/tests (analytic gain calibration, *not* SNR-calibrated like the experiment).
- Supporting: `recognition.py` (MLP encoder), `likelihood.py` (Gaussian/Poisson),
  `functional.py` (`rbf`, `gaussian_loss`, entropy), `distribution.py` (the `Gaussian`
  namedtuple `(mean, logvar)`), `kalman.py`, `util.py`, `numerical.py`.

### Two encoder modes (set at `make_model`, gated in `forward`)
- `encoder='spikes'` (default): recognition reads the raw `ydim` observation.
- `encoder='projection'`: recognition reads an `xdim`-dim subspace projection passed
  as `filter(..., y_enc=...)`. This path *requires* a `y_enc` (raw-y raises) and is
  driven by an `OnlineReadout`.

### Key empirical facts (see the two IMPL.md files)
- **Poisson collapse**: a readout learned from random init collapses to the trivial
  mean-rate solution (R^2 ~ 0). Fixes: PCA warm-start + freeze the decoder, or the
  projection encoder. Do not "fine-tune" a good frozen readout -- it has no headroom
  and the low-rate gradient pulls it back to the trivial solution.
- `srrls` is the only flow learner stable *and* fast over 200k-step streams.
- O(T^2) OOM trap: `transition(..., sampling=False)` forms an `NxN` matrix to read its
  diagonal -- chunk any transition call over a long stream, never run it on the full array.

## Conventions specific to this repo

- **Equation immutability**: do NOT change equation/update code (`gaussian_loss`,
  the RLS/sRLS/Kalman math, the ELBO terms) without consulting Memming first. The one
  prior fix (a `gaussian_loss` variance-term bug) was confirmed before landing.
- **Seeds**: never 42. Synthetic data uses `20260605`; the experiment master seed is
  `20260602`. New code: default to `date +%Y%m%d`.
- **Git**: remote is `catniplab/vjf`, default branch `master`. Branch + PR; `git add`
  files by name (never `-A`/`.`/`-u`). Do not commit large result files -- `script/`,
  `notebook/`, `gcp_runs/exp-*`, the experiment `results*/` dirs, and most `*.pdf`/
  `*.json` artifacts are gitignored. Tutorials that ARE committed live in `examples/`.
- **Progress notes**: root `IMPL.md` tracks the sVJF framework units;
  `experiments/lc_poisson_stream/IMPL.md` holds the experiment findings;
  `paper/EXPERIMENT_PLAN.md` + `EXPERIMENT_PROGRESS.md` track the paper's experiments.
- Each unit of work is `/codex-review`'d before commit (see `IMPL.md`).
- **Slack**: this project's channel is `#joint-filtering` (`C0Y47F38U`) on the
  catniplab workspace. Post files with `/slack-upload C0Y47F38U <filepath> [title]`
  (the `@pm4mp` bot must be a member -- `/invite @pm4mp` if not).

Note: `pyproject.toml` still declares Poetry metadata, but the workflow is `uv` + the
editable install above.
