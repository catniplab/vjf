# Implementation plan: projection-input recognition + decoupled online readout

Implements `note_projection_encoder.tex` in `vjf`. General, backward-compatible
(defaults reproduce current VJF). Branch: `feat/projection-encoder`.

## Scope / goal

Two coupled capabilities, both optional and off by default:
1. **Recognition takes a feature input** distinct from the decoder/likelihood target,
   so we can feed `pi_t = C^+ (g~(y_t) - b)` instead of raw `y_t`.
2. **Decoupled online readout estimator** (incremental PCA + Procrustes-anchored
   refresh) that owns `C, b`, shared by the decoder and the projection.

The ELBO/generative model are unchanged (see the note). No new objective.

## File-by-file changes

### `vjf/model.py` (core; the only conceptually-significant edit)
- `VJF.forward(self, y, qs, u=None, y_enc=None)`: use `y_enc` for the recognition,
  `y` everywhere else. Line that is today `qt = self.recognition(y, qs, u)` becomes
  `self.recognition(y if y_enc is None else y_enc, qs, u)`. Decoder/likelihood keep `y`.
- `VJF.filter(self, y, u=None, qs=None, *, y_enc=None, ...)`: thread `y_enc` into
  `forward`. Recon loss still on raw `y`. Backward-compatible (default `y_enc=None`).
- `VJF.make_model(..., encoder='spikes')`: `encoder='projection'` builds the recognition
  with input dim = `xdim` (i.e. `Recognition(xdim, xdim, udim, hidden)`), else `ydim`
  (today). Returns model unchanged otherwise.
- (No change to `LinearDecoder`, `RBFDS`, optimizer construction.)

### `vjf/recognition.py`
- No structural change. The projection encoder is just `Recognition(ydim=xdim, ...)`.
  (Optional: clarify the first arg name to `in_dim`.)

### `vjf/readout.py` (NEW — the online readout front-end)
Class `OnlineReadout(n, m, *, init='pca', smooth_tau=8.0, log_c=1e-2, refresh_K=1000,
forgetting=None)` holding the streaming state and exposing:
- `warm_start(counts_window)`: causal-EMA-log batch PCA over the initial window ->
  `(C, b)`, latent scaled to unit variance; sets the running mean `b` and the
  incremental-PCA state.
- `feature(y_t) -> g~`: causal EMA update + log (Poisson) / identity (Gaussian);
  updates running mean used for `b`.
- `project(g~) -> pi_t`: `C_pinv @ (g~ - b)` (O(nm); `C_pinv` cached, see refresh).
- `update(g~)`: one incremental-PCA (CCIPCA/Oja) step on `g~ - b`.
- `maybe_refresh(decoder, transition, step)`: every `refresh_K`, form `C_new` from the
  PCA state (unit-var scaling), **Procrustes-anchor** to the current `C`, write it to
  `decoder` (`weight=C_new`, `bias=b`), recompute `C_pinv` once, and re-express the flow
  by the induced rotation `R` (rotate `transition.velocity.feature.centroid` by `R` and
  `velocity.w_mean` by `R^T`; for srrls also `w_chol` is in feature space so untouched).
Keep the algorithm here (out of VJF core and out of the experiment driver); reusable.
`pca_readout_init` currently in the experiment moves here as `warm_start`.

### `experiments/lc_poisson_stream/experiment.py`
- cfg: add `readout='pca_proj_online'` (plus existing `oracle|learned|pca`), with
  `proj_tau`, `proj_refresh_K`, `proj_init_mult`.
- `run_condition`: for the new mode, build VJF with `encoder='projection'`, instantiate
  `OnlineReadout`, `warm_start` on the first `init_mult*N` causal bins, then per step:
  `g=front.feature(y); front.update(g); pi=front.project(g);
   model.filter(y, y_enc=pi, ...); front.maybe_refresh(model.decoder, model.transition, t)`.
  Existing modes keep `encoder='spikes'` and the current path.
- Make the projection front-end available to `oracle` too (fixed true `C`) since it is a
  strictly better encoder; expose as `encoder` cfg independent of `readout`.

### `test/`
- `test_model.py`: add (a) backward-compat (default `y_enc=None`/`encoder='spikes'`
  reproduces current forward/filter shapes), (b) a `encoder='projection'` filter step
  runs and returns correct shapes.
- `test_readout.py` (new): `OnlineReadout` unit tests -- warm_start shapes; `project`
  output dim = m; incremental PCA reduces subspace angle to a planted `C` over a stream;
  Procrustes refresh keeps `C` aligned (no sign/rotation jumps); `C_pinv` recomputed only
  on refresh.

## Algorithm (per step), from the note
```
g  = feature(y_t)          # EMA -> log (Poisson) ; identity (Gaussian); updates running b
update(g)                  # CCIPCA step on (g - b)
pi = C_pinv @ (g - b)      # O(nm) projection (C_pinv cached)
qt,... = filter(y_t, y_enc=pi)     # recognition on pi; decoder/likelihood on y_t
every K: C_new = procrustes(scale(pca_subspace), C); decoder<-(C_new,b); C_pinv=pinv(C_new)
         rotate flow centroids by R, velocity weights by R^T   # exact for rotation
```

## Backward compatibility & acceptance
- Defaults (`encoder='spikes'`, `y_enc=None`, `readout` in {oracle,learned,pca}) reproduce
  current behavior; `test_module.py`/`test_model.py` (7) still pass.
- New-mode acceptance (multi-seed, matching the validated prototype):
  - online `C` subspace angle to true `C` decreases over the stream toward the batch
    asymptote (CCIPCA unit test);
  - `pca_proj_online` end-to-end R^2 reaches the projection-`oracle` ceiling across SNR and
    exceeds raw-spike recognition;
  - stable over 1000 s (`n_diverge` small), per-bin compute still O(nm).

## Risks / open
- Flow re-expression is exact only for a **rotation** `R`; the Procrustes anchor yields a
  rotation, but the residual subspace tilt is not a 2x2 map (handled because the encoder
  input `pi` tracks the new `C` -- the flow only needs the in-plane rotation). Validate
  forecast-horizon stability across refreshes.
- The recognition `logvar` head is diagonal; a rotation mixes dims -> approximate. Small
  per-refresh `R` keeps this negligible; monitor.
- `C^+` uses `(C^T C)^{-1}` (m x m, tiny/stable); recompute only on refresh.

## Sequence (commits on feat/projection-encoder)
1. `vjf/model.py` decoupled `y_enc` + `make_model(encoder=...)` + tests (backward-compat). 
2. `vjf/readout.py` `OnlineReadout` + `test_readout.py`.
3. experiment `readout='pca_proj_online'` wiring + medium smoke.
4. multi-seed confirmation (local medium-T or GCP), then update `note`/`IMPL` with results.
