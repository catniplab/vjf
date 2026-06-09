# sVJF on Graf V1 - M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get sVJF running end-to-end on the strongest Graf V1 array (`array_5`) at 10 ms bins, single streaming pass with an unknown readout, and produce the core M1 metrics (leave-one-neuron-out PLL, orientation torus, single-trial decoding, free-run forecast, per-bin timing).

**Architecture:** Add a Graf loader (spike-time cells -> binned trial tensors + labels + neuron quality), a *trial-aware* wrapper over the existing `online_filter` (reset the posterior to the prior at each trial boundary; coverage warm-up spanning all directions; one-time readout + data-driven RBF-center seeding at the boundary), a data-driven RBF placement option in `RBFDS.initialize` (k-means on the warm-up latent buffer; default stays the current box init), and an eval module. The generative model, ELBO, and RLS math are untouched.

**Tech Stack:** Python via `uv`; PyTorch (existing `vjf`), NumPy, SciPy (`scipy.io.loadmat`), scikit-learn (KMeans, already a dev dep), pytest.

**Scope of THIS plan (M1 core):** units U1-U4 + a minimal U6 driver -> the sVJF result + core metrics on `array_5` @ 10 ms. Deferred to follow-on plans: U5 baselines (vLGP/GPFA/PLDS), the polished figure set, M2 (5/1 ms), M3 (other arrays). Each is its own plan.

---

## Confirmed data facts (from inspection 2026-06-09)

- File: `experiments/v1_graf/data/raw/array_{1..5}.mat` (gitignored). MATLAB v<7.3 -> `scipy.io.loadmat`.
- Per array variables: `spk_times` (object array `(N, 3600)`, each cell `(n_spk, 1)` spike times in **ms**), `ori` `(3600,)` direction in degrees (72 unique, 5 deg spacing, 50 reps), `tf_tot` (temporal freq, **6.25 Hz**), `neur_param` `(N, 2)`, plus `t_stim`/`t_pres`/`t_spont`/`spk_res`/`t_vec` timing metadata.
- Trial length 2560 ms: **first 1280 ms stimulus (drifting grating), second 1280 ms blank**. Response latency ~50 ms (vLGP analyzed 150-1150 ms).
- Per-array signal (mean rate / N): array_5 **10.42 Hz / 148** (strongest), array_3 5.00/113, array_2 4.72/113, array_4 2.25/133, array_1 1.30/147.
- Grating period 1/6.25 = **160 ms**; at 10 ms bins = **16 bins/cycle**, 256 bins/trial, 128 bins in the stimulus window.

---

## File structure

- Create `experiments/v1_graf/graf_loader.py` - load a `.mat` array; bin spike-time cells to `(trial, bin, neuron)` counts; orientation labels; per-neuron tuning fit + quality mask; per-array signal metric.
- Create `experiments/v1_graf/eval.py` - leave-one-neuron-out PLL, orientation decoding, torus extraction, free-run forecast, timing summary.
- Create `experiments/v1_graf/run_m1.py` - the M1 driver (load array_5 @ 10 ms, 80/20 trial split, coverage warm-up, online pass, eval, save JSON).
- Modify `vjf/realtime.py` - add `online_filter_trials(...)` (trial-aware reset + coverage warm-up). Additive; `online_filter` unchanged.
- Modify `vjf/model.py` - `RBFDS.initialize(..., rbf_centers=None, rbf_logwidths=None)`; pass through to `velocity.init_srls`.
- Modify `vjf/module.py` - `LinearRegression.init_srls(..., centers=None, logwidths=None)`: use preset centers/widths if given, else current uniform-box behavior.
- Create `test/test_v1_graf.py` - loader + eval unit tests (tiny fabricated data; no dependency on the 172 MB file).
- Modify `test/test_realtime.py` - tests for `online_filter_trials`.
- Modify `test/test_module.py` - tests for data-driven `init_srls` centers.

Seeds: `20260609` + per-use offsets. Never 42.

---

## Task 1: Loader - load a `.mat` array and bin spikes

**Files:**
- Create: `experiments/v1_graf/graf_loader.py`
- Test: `test/test_v1_graf.py`

- [ ] **Step 1: Write the failing test (binning)**

```python
# test/test_v1_graf.py
import numpy as np
import pytest
from experiments.v1_graf.graf_loader import bin_spikes

def _fake_spk(N=3, n_trial=4, seed=20260609):
    rng = np.random.default_rng(seed)
    spk = np.empty((N, n_trial), dtype=object)
    for i in range(N):
        for c in range(n_trial):
            spk[i, c] = (np.sort(rng.uniform(0, 2560, size=rng.integers(0, 20)))
                         .reshape(-1, 1))
    return spk

def test_bin_spikes_shape_and_counts():
    spk = _fake_spk()
    counts = bin_spikes(spk, bin_ms=10.0, t_total_ms=2560.0)
    assert counts.shape == (4, 256, 3)          # (trial, bin, neuron)
    assert counts.dtype == np.float32
    # total binned spikes == total spike times within [0, t_total)
    raw = sum(np.asarray(spk[i, c]).size for i in range(3) for c in range(4))
    assert int(counts.sum()) == raw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_v1_graf.py::test_bin_spikes_shape_and_counts -v`
Expected: FAIL (`ModuleNotFoundError` / `bin_spikes` undefined).

- [ ] **Step 3: Implement loader (`graf_loader.py`)**

```python
"""Load the Graf et al. (2011) macaque V1 drifting-grating dataset.

array_{1..5}.mat hold spk_times (object (N,3600) cells of spike times in ms),
ori (3600,) grating direction in deg, tf_tot (temporal freq, 6.25 Hz),
neur_param (N,2). Trial = 2560 ms (first 1280 ms stimulus, then blank).
"""
from __future__ import annotations
import os
import numpy as np
from scipy.io import loadmat

N_REP, N_ORI, T_TOTAL_MS, STIM_MS = 50, 72, 2560.0, 1280.0
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

def load_array(array_num: int, data_dir: str = DATA_DIR) -> dict:
    """Return raw fields for one array: spk_times (N,3600) object, ori (3600,),
    tf (float), neur_param (N,2)."""
    assert array_num in range(1, 6)
    m = loadmat(os.path.join(data_dir, f"array_{array_num}.mat"))
    return {
        "spk_times": m["spk_times"],
        "ori": np.ravel(m["ori"]).astype(float),
        "tf": float(np.ravel(m["tf_tot"])[0]),
        "neur_param": m["neur_param"],
    }

def bin_spikes(spk_times: np.ndarray, bin_ms: float = 10.0,
               t_total_ms: float = T_TOTAL_MS) -> np.ndarray:
    """Histogram spike-time cells into counts (trial, bin, neuron), float32.
    Spikes are clipped to [0, t_total_ms)."""
    N, n_trial = spk_times.shape
    n_bin = int(round(t_total_ms / bin_ms))
    edges = np.arange(n_bin + 1) * bin_ms
    counts = np.zeros((n_trial, n_bin, N), dtype=np.float32)
    for i in range(N):
        for c in range(n_trial):
            t = np.asarray(spk_times[i, c], dtype=float).ravel()
            t = t[(t >= 0.0) & (t < t_total_ms)]
            counts[c, :, i] = np.histogram(t, bins=edges)[0]
    return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_v1_graf.py::test_bin_spikes_shape_and_counts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/v1_graf/graf_loader.py test/test_v1_graf.py
git commit -m "v1_graf: loader - load_array + bin_spikes"
```

---

## Task 2: Loader - orientation tuning fit, quality mask, signal metric

**Files:**
- Modify: `experiments/v1_graf/graf_loader.py`
- Test: `test/test_v1_graf.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_v1_graf.py
from experiments.v1_graf.graf_loader import tuning_curve, well_tuned_mask, signal_metric

def test_tuning_and_quality_mask():
    # one well-tuned neuron (bimodal cosine of direction) + one flat neuron
    rng = np.random.default_rng(20260609)
    ori_axis = np.arange(0, 360, 5.0)
    dirs = np.repeat(ori_axis, 50)                 # (3600,) sorted; 50 reps
    rad = np.deg2rad(dirs)
    rate_tuned = 1.0 + 0.9 * np.cos(2 * (rad - np.deg2rad(40)))   # orientation (180-periodic)
    spk = np.empty((2, dirs.size), dtype=object)
    for c in range(dirs.size):
        spk[0, c] = np.sort(rng.uniform(0, 1280, int(rng.poisson(8 * rate_tuned[c])))).reshape(-1, 1)
        spk[1, c] = np.sort(rng.uniform(0, 1280, int(rng.poisson(8 * 1.0)))).reshape(-1, 1)
    counts = bin_spikes(spk, bin_ms=10.0)
    tc, axis = tuning_curve(counts, dirs, stim_only=True)
    assert tc.shape == (2, 72) and axis.shape == (72,)
    mask, r2 = well_tuned_mask(counts, dirs, r2_thresh=0.75)
    assert mask[0] and not mask[1]                  # tuned kept, flat rejected
    assert r2[0] > 0.75 and r2[1] < 0.5

def test_signal_metric_monotone():
    spk = _fake_spk(N=2, n_trial=8)
    counts = bin_spikes(spk, bin_ms=10.0)
    sm = signal_metric(counts)
    assert set(sm) >= {"N", "mean_rate_hz", "total_spikes"}
    assert sm["N"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_v1_graf.py -k "tuning or signal" -v`
Expected: FAIL (`tuning_curve` undefined).

- [ ] **Step 3: Implement tuning fit + mask + metric**

```python
# append to experiments/v1_graf/graf_loader.py
from scipy.optimize import curve_fit

def tuning_curve(counts: np.ndarray, ori: np.ndarray, *, bin_ms: float = 10.0,
                 stim_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Mean spike RATE (Hz) per neuron per direction. counts (trial,bin,neuron).
    stim_only restricts to the first STIM_MS of each trial."""
    n_bin = counts.shape[1]
    sl = slice(0, int(round(STIM_MS / bin_ms))) if stim_only else slice(None)
    win_s = (sl.stop if sl.stop else n_bin) * bin_ms / 1000.0
    per_trial = counts[:, sl, :].sum(1)                       # (trial, neuron) counts
    axis = np.unique(ori)
    tc = np.stack([per_trial[ori == d].mean(0) for d in axis], axis=1)  # (neuron, n_ori)
    return tc / win_s, axis                                   # Hz

def _von_mises2(theta, b, a1, mu1, k1, a2, mu2, k2):
    t = np.deg2rad(theta)
    return (b + a1 * np.exp(k1 * (np.cos(t - np.deg2rad(mu1)) - 1))
              + a2 * np.exp(k2 * (np.cos(t - np.deg2rad(mu2)) - 1)))

def well_tuned_mask(counts: np.ndarray, ori: np.ndarray, *, bin_ms: float = 10.0,
                    r2_thresh: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Keep neurons whose direction tuning is fit (R^2 >= thresh) by a sum of two
    von Mises bumps (~180 deg apart), mirroring vLGP's selection."""
    tc, axis = tuning_curve(counts, ori, bin_ms=bin_ms, stim_only=True)
    N = tc.shape[0]
    r2 = np.zeros(N)
    for n in range(N):
        y = tc[n]
        if y.max() <= 0:
            continue
        pk = axis[int(np.argmax(y))]
        p0 = [y.min(), y.max(), pk, 2.0, 0.5 * y.max(), (pk + 180) % 360, 2.0]
        try:
            popt, _ = curve_fit(_von_mises2, axis, y, p0=p0, maxfev=10000)
            yhat = _von_mises2(axis, *popt)
            ss = ((y - y.mean()) ** 2).sum()
            r2[n] = 1.0 - ((y - yhat) ** 2).sum() / (ss + 1e-12)
        except (RuntimeError, ValueError):
            r2[n] = 0.0
    return r2 >= r2_thresh, r2

def signal_metric(counts: np.ndarray, *, t_total_ms: float = T_TOTAL_MS) -> dict:
    """Per-array signal proxy for strongest-first ordering."""
    N = counts.shape[2]
    total = float(counts.sum())
    return {"N": N, "total_spikes": total,
            "mean_rate_hz": float(counts.sum((0, 1)).mean() / (t_total_ms / 1000.0))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_v1_graf.py -k "tuning or signal" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/v1_graf/graf_loader.py test/test_v1_graf.py
git commit -m "v1_graf: tuning-curve fit, well-tuned mask, signal metric"
```

---

## Task 3: Trial-aware online filter (U2)

**Files:**
- Modify: `vjf/realtime.py` (add `online_filter_trials`)
- Test: `test/test_realtime.py`

The existing `online_filter` threads one posterior across a flat stream with a single warm-up boundary and a one-time `transition.initialize`. For trials we (a) reset the posterior to the prior at each trial start (no cross-trial threading, no cross-boundary one-step prediction), (b) buffer warm-up latent means across a *coverage* set spanning trials, and (c) seed the readout + RBF centers once at the boundary, then learn online. We add a thin wrapper that drives the per-trial `VJF.filter` loop directly (re-using its math) rather than overloading `online_filter`'s single-stream contract.

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_realtime.py
from vjf.realtime import online_filter_trials

def test_trials_reset_posterior_each_trial():
    counts = _counts(T=20)
    trials = [counts[:10], counts[10:]]            # two 10-step trials
    m = _model(encoder="spikes")
    seen_steps = []
    results = list(online_filter_trials(m, trials, warmup_trials=1, readout=None))
    # one result per sample, with trial index and in-trial step
    assert len(results) == 20
    assert [r.trial for r in results[:10]] == [0]*10
    assert [r.trial for r in results[10:]] == [1]*10
    # first sample of each trial starts from the prior: pred_mean is None
    assert results[0].pred_mean is None and results[10].pred_mean is None
    # within a trial, later samples have a one-step prediction
    assert results[5].pred_mean is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_realtime.py::test_trials_reset_posterior_each_trial -v`
Expected: FAIL (`online_filter_trials` undefined).

- [ ] **Step 3: Implement `online_filter_trials`**

Add to `vjf/realtime.py` (re-uses `OnlineResult`; add a `trial` field to it and an in-trial `t_in_trial`). Extend the dataclass:

```python
# in OnlineResult dataclass, add:
    trial: int = -1            # trial index (real-time multi-trial use)
    t_in_trial: int = -1       # step within the trial
```

```python
def online_filter_trials(model, trials, *, readout=None, adapt_readout=True,
                         warmup_trials=1, rbf_width_scale=1.0,
                         logvar_floor=math.log(1e-6), max_abs_state=1e3,
                         seed_centers=None):
    """Drive ``model`` over a sequence of independent ``trials`` (each an iterable of
    y_t), resetting the posterior to the prior at every trial start. The model
    (readout, flow) persists across trials. Trials 0..warmup_trials-1 are the
    coverage warm-up (dynamics off, latent means buffered across ALL of them); at the
    boundary the readout is warm-started, the flow is initialized once from the
    buffered means, and (if ``seed_centers``) RBF centers are placed data-drivenly
    (Task 4). Yields one OnlineResult per sample with .trial/.t_in_trial set.

    No new math: every step calls model.filter / OnlineReadout, exactly as
    online_filter, only the reset bookkeeping differs.
    """
    xdim = model.mean.shape[-1]
    warm_means, warm_us = [], []
    initialized = False
    global_t = 0
    for ti, trial in enumerate(trials):
        warm = ti < warmup_trials
        q = None                                   # <-- reset to prior at trial start
        prev_mean = np.zeros(xdim, dtype=np.float32)
        for k, y_t in enumerate(trial):
            # one-time init at the coverage boundary (first sample after warm-up trials)
            if ti == warmup_trials and not initialized and len(warm_means) > 1:
                m = torch.as_tensor(np.asarray(warm_means), dtype=torch.get_default_dtype())
                if seed_centers is not None:
                    centers, logw = seed_centers(np.asarray(warm_means))   # Task 4 helper
                    model.transition.initialize(m[1:], m[:-1], None,
                                                rbf_centers=centers, rbf_logwidths=logw)
                else:
                    model.transition.initialize(m[1:], m[:-1], None)
                    model.transition.velocity.feature.logwidth.data += math.log(rbf_width_scale)
                initialized = True
            pred = None
            if q is not None:
                with torch.no_grad():
                    pred = _mean(model.transition(q.mean, None, sampling=False)).detach().cpu().numpy()[0]
            t0 = time.perf_counter()
            y_enc = None
            if readout is not None:
                g = readout.feature(np.asarray(y_t), update_mean=adapt_readout)
                if adapt_readout:
                    readout.update(g)
                y_enc = torch.as_tensor(readout.project(g))
            try:
                qt, loss, recon, dyn, ent = model.filter(
                    y_t, None, q, sgd=True, update=True, verbose=True, warm_up=warm, y_enc=y_enc)
            except AssertionError:
                model.transition.logvar.data.clamp_(min=logvar_floor)
                q = None
                yield OnlineResult(step=global_t, mean=prev_mean.copy(),
                                   logvar=np.zeros(xdim, np.float32), pred_mean=pred,
                                   loss=float("nan"), recon=float("nan"), dynamics=float("nan"),
                                   entropy=float("nan"), warming_up=warm, diverged=True,
                                   refreshed=False, elapsed_s=time.perf_counter()-t0,
                                   trial=ti, t_in_trial=k)
                global_t += 1
                continue
            model.transition.logvar.data.clamp_(min=logvar_floor)
            mu = qt.mean.detach()
            diverged = (not torch.isfinite(mu).all()) or (mu.abs().max() > max_abs_state)
            refreshed = False
            if not diverged and readout is not None and adapt_readout and not warm:
                refreshed = bool(readout.maybe_refresh(model.decoder, global_t))
            elapsed = time.perf_counter() - t0
            if diverged:
                q = None
                mean_np = prev_mean.copy()
            else:
                q = qt
                mean_np = mu.cpu().numpy()[0].astype(np.float32)
                prev_mean = mean_np
            if warm:
                warm_means.append(mean_np)
            yield OnlineResult(step=global_t, mean=mean_np,
                               logvar=qt.logvar.detach().cpu().numpy()[0].astype(np.float32),
                               pred_mean=pred, loss=float(loss.detach()), recon=float(recon.detach()),
                               dynamics=float(dyn.detach()), entropy=float(ent.detach()),
                               warming_up=warm, diverged=diverged, refreshed=refreshed,
                               elapsed_s=elapsed, trial=ti, t_in_trial=k)
            global_t += 1
```

(Note: pred is computed but the first sample of each trial has `q is None` -> `pred=None`, satisfying the no-cross-boundary requirement.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_realtime.py::test_trials_reset_posterior_each_trial -v`
Expected: PASS.

- [ ] **Step 5: Run the full realtime suite (no regression)**

Run: `uv run pytest test/test_realtime.py -v`
Expected: all PASS (existing `online_filter` untouched).

- [ ] **Step 6: Commit**

```bash
git add vjf/realtime.py test/test_realtime.py
git commit -m "realtime: online_filter_trials (per-trial posterior reset + coverage warm-up)"
```

---

## Task 4: Data-driven RBF center placement (U3)

**Files:**
- Modify: `vjf/module.py` (`LinearRegression.init_srls` accepts preset centers/widths)
- Modify: `vjf/model.py` (`RBFDS.initialize` passes them through)
- Create: `experiments/v1_graf/graf_loader.py` helper `kmeans_centers` (or a small module)
- Test: `test/test_module.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_module.py
import numpy as np, torch
from vjf.module import RBF, LinearRegression

def test_init_srls_uses_preset_centers():
    torch.manual_seed(0)
    feat = RBF(2, 5)
    lr = LinearRegression(feat, 2, bayes=True)
    centers = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,2.]])
    logw = torch.log(torch.full((5,), 0.3))
    x = torch.randn(20, 2); y = torch.randn(20, 2)
    lr.init_srls(x, y, centers=centers, logwidths=logw)
    assert torch.allclose(lr.feature.centroid.data, centers)
    assert torch.allclose(lr.feature.logwidth.data, logw)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_module.py::test_init_srls_uses_preset_centers -v`
Expected: FAIL (`init_srls` has no `centers`/`logwidths` kwargs).

- [ ] **Step 3: Implement preset-centers path**

In `vjf/module.py`, change `init_srls` (currently around lines 175-185) to:

```python
    @torch.no_grad()
    def init_srls(self, x: Tensor, target: Tensor, p0: float = 1.0,
                  centers: Tensor = None, logwidths: Tensor = None):
        """Initialize square-root RLS state. If ``centers`` (n_basis, n_dim) is given,
        the RBF centers/widths are set from it (data-driven placement); otherwise the
        original uniform-box init over the state radius is used."""
        if centers is not None:
            self.feature.centroid.data.copy_(torch.as_tensor(centers, dtype=self.feature.centroid.dtype))
            if logwidths is not None:
                self.feature.logwidth.data.copy_(torch.as_tensor(logwidths, dtype=self.feature.logwidth.dtype))
        else:
            r = x.norm(dim=1).max().item()
            nn.init.uniform_(self.feature.centroid, a=-r, b=r)
            nn.init.constant_(self.feature.logwidth, math.log(r))
        feat = self.feature(x)
        self.w_mean = torch.linalg.lstsq(feat, target).solution
        n = self.feature.n_feature
        self.w_chol = (p0 ** 0.5) * torch.eye(n, dtype=feat.dtype, device=feat.device)
```

In `vjf/model.py`, `RBFDS.initialize` (around lines 438-450):

```python
    @torch.no_grad()
    def initialize(self, xt: Tensor, xs: Tensor, ut: Tensor = None, *,
                   rbf_centers: Tensor = None, rbf_logwidths: Tensor = None):
        xs = torch.atleast_2d(xs)
        xt = torch.atleast_2d(xt)
        xu = nonecat(xs, ut)
        if self.flow_learner == 'srrls':
            self.velocity.init_srls(xu, xt - xs, centers=rbf_centers, logwidths=rbf_logwidths)
        else:
            mse = (xt - xs).pow(2).mean()
            self.velocity.initialize(xu, xt - xs, mse)
        d = self._velocity_mean(xu)
        mse = (xt - xs - d).pow(2).mean()
        self.logvar.data = mse.log()
```

(Equation immutability: this is initialization/placement only - not `gaussian_loss`, the srls update, or the ELBO. Approved in planning.)

- [ ] **Step 4: Add the `kmeans_centers` helper + its test**

```python
# append to experiments/v1_graf/graf_loader.py
def kmeans_centers(states: np.ndarray, n_rbf: int, *, width_scale: float = 1.0,
                   seed: int = 20260609):
    """Place n_rbf RBF centers by k-means on visited latent states (states: (T, xdim)),
    widths = width_scale * median nearest-center distance. Returns (centers, logwidths)
    as float32 numpy arrays for RBFDS.initialize(rbf_centers=..., rbf_logwidths=...)."""
    from sklearn.cluster import KMeans
    n = min(n_rbf, states.shape[0])
    km = KMeans(n_clusters=n, n_init=4, random_state=seed).fit(states)
    c = km.cluster_centers_.astype(np.float32)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn_dist = d.min(1)
    w = (width_scale * np.median(nn_dist)).astype(np.float32)
    logw = np.log(np.full(n, max(float(w), 1e-3), dtype=np.float32))
    return c, logw
```

```python
# append to test/test_v1_graf.py
from experiments.v1_graf.graf_loader import kmeans_centers
def test_kmeans_centers_shapes():
    rng = np.random.default_rng(20260609)
    states = rng.standard_normal((500, 3)).astype(np.float32)
    c, logw = kmeans_centers(states, n_rbf=20)
    assert c.shape == (20, 3) and logw.shape == (20,)
    assert np.isfinite(logw).all()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/test_module.py::test_init_srls_uses_preset_centers test/test_v1_graf.py::test_kmeans_centers_shapes -v`
Expected: PASS. Then `uv run pytest test/test_module.py test/test_model.py -v` (no regression).

- [ ] **Step 6: Commit**

```bash
git add vjf/module.py vjf/model.py experiments/v1_graf/graf_loader.py test/test_module.py test/test_v1_graf.py
git commit -m "rbfds: data-driven RBF center placement (init_srls preset centers) + kmeans helper"
```

---

## Task 5: Eval - leave-one-neuron-out predictive log-likelihood (U4)

**Files:**
- Create: `experiments/v1_graf/eval.py`
- Test: `test/test_v1_graf.py`

PLL (bits/spike, vLGP Eq. 39) given per-(trial,bin,neuron) predicted rates `lam` and observed counts `y`, vs a homogeneous-Poisson baseline `ybar` (population mean rate):

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_v1_graf.py
from experiments.v1_graf.eval import predictive_ll_bits_per_spike
def test_pll_perfect_vs_baseline():
    rng = np.random.default_rng(20260609)
    ybar = 0.1
    y = rng.poisson(ybar, size=(5, 100, 8)).astype(np.float32)
    # model that predicts the true generating rate beats the mean-rate baseline ~ 0
    lam_true = np.full_like(y, ybar)
    pll = predictive_ll_bits_per_spike(y, lam_true, ybar)
    assert abs(pll) < 0.05                       # equals baseline -> ~0 bits/spike
    lam_better = np.clip(y.mean(0, keepdims=True).repeat(5,0), 1e-3, None)  # PSTH predictor
    assert predictive_ll_bits_per_spike(y, lam_better, ybar) >= -0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_v1_graf.py::test_pll_perfect_vs_baseline -v`
Expected: FAIL (`eval` module undefined).

- [ ] **Step 3: Implement PLL**

```python
# experiments/v1_graf/eval.py
from __future__ import annotations
import numpy as np

def predictive_ll_bits_per_spike(y: np.ndarray, lam: np.ndarray, ybar: float) -> float:
    """Poisson predictive log-likelihood normalized to a homogeneous-Poisson baseline,
    in bits/spike (vLGP Eq. 39). y, lam: (trial, bin, neuron) counts and predicted rates."""
    lam = np.clip(lam, 1e-6, None)
    model_ll = (y * np.log(lam) - lam).sum()
    base_ll = (y * np.log(ybar) - ybar).sum()
    n_spk = y.sum()
    return float((model_ll - base_ll) / (n_spk * np.log(2.0) + 1e-12))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_v1_graf.py::test_pll_perfect_vs_baseline -v`
Expected: PASS.

- [ ] **Step 5: Implement leave-one-neuron-out inference + commit**

Add to `experiments/v1_graf/eval.py` the driver-facing routine that, given a trained model + readout and a test trial's counts, infers the latent from all-but-one neuron (projection encoder using `C` with that row removed) and predicts the held-out neuron's rate from the full `C` row. Uses `model.decoder.decode.weight/bias` and `model.transition` frozen.

```python
import torch
from vjf.distribution import Gaussian

@torch.no_grad()
def leave_one_neuron_rates(model, readout, trial_counts: np.ndarray) -> np.ndarray:
    """Return predicted rates (bin, neuron) for one test trial under leave-one-neuron-out:
    for each held-out neuron n, filter the trial from the OTHER neurons' projection and
    predict lam[:, n] = exp(C[n].x + b[n]). Model/readout frozen (no learning, no refresh)."""
    C = model.decoder.decode.weight.detach().cpu().numpy()      # (N, m)
    b = model.decoder.decode.bias.detach().cpu().numpy()        # (N,)
    n_bin, N = trial_counts.shape
    lam = np.zeros((n_bin, N), dtype=np.float64)
    for n in range(N):
        keep = np.arange(N) != n
        Csub = C[keep]; Cpinv = np.linalg.pinv(Csub).astype(np.float32)
        q = None
        for t in range(n_bin):
            y = trial_counts[t]
            g = readout.feature(y, update_mean=False)           # frozen feature
            x_enc = torch.as_tensor((Cpinv @ (g[keep] - readout.mean_b[keep])).astype(np.float32))
            qt, *_ = model.filter(y, None, q, sgd=False, update=False, verbose=True, y_enc=x_enc)
            q = qt
            xm = qt.mean.detach().cpu().numpy()[0]
            lam[t, n] = np.exp(C[n] @ xm + b[n])
    return lam
```

```bash
git add experiments/v1_graf/eval.py test/test_v1_graf.py
git commit -m "v1_graf: eval - leave-one-neuron-out PLL + per-trial inference"
```

---

## Task 6: Eval - orientation decoding, torus, forecast (U4)

**Files:**
- Modify: `experiments/v1_graf/eval.py`
- Test: `test/test_v1_graf.py`

- [ ] **Step 1: Write the failing test (decoding + torus shape)**

```python
# append to test/test_v1_graf.py
from experiments.v1_graf.eval import orientation_decode_acc, torus_embedding
def test_decode_and_torus_shapes():
    rng = np.random.default_rng(20260609)
    # latents separable by direction -> decodable
    dirs = np.repeat(np.arange(0,360,5.0), 4)
    lat = np.stack([np.cos(np.deg2rad(dirs)), np.sin(np.deg2rad(dirs)),
                    rng.standard_normal(dirs.size)*0.01], 1).astype(np.float32)
    acc = orientation_decode_acc(lat, dirs, n_splits=4, seed=20260609)
    assert acc > 0.5
    emb, axis = torus_embedding(lat, dirs)
    assert emb.shape[0] == 72 and emb.shape[1] == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/test_v1_graf.py -k "decode or torus" -v`
Expected: FAIL.

- [ ] **Step 3: Implement decode + torus + forecast**

```python
# append to experiments/v1_graf/eval.py
def orientation_decode_acc(latent_per_trial: np.ndarray, dirs: np.ndarray, *,
                           n_splits: int = 5, seed: int = 20260609) -> float:
    """Cross-validated direction-decoding accuracy from per-trial latent summaries
    (latent_per_trial: (n_trial, feat)). Multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
    y = np.round(dirs).astype(int)
    return float(cross_val_score(clf, latent_per_trial, y, cv=n_splits).mean())

def torus_embedding(latent: np.ndarray, dirs: np.ndarray):
    """Trial-averaged latent per direction, projected to its first 3 singular vectors
    (reproduces vLGP Fig 8). latent (n_trial, m). Returns (72, 3) and the direction axis."""
    axis = np.unique(dirs)
    avg = np.stack([latent[dirs == d].mean(0) for d in axis], 0)   # (72, m)
    u, s, vt = np.linalg.svd(avg - avg.mean(0), full_matrices=False)
    return (avg - avg.mean(0)) @ vt[:3].T, axis

@torch.no_grad()
def forecast_r2(model, x0: np.ndarray, true_path: np.ndarray, k: int) -> float:
    """Affine-aligned R^2 of a k-step free run of the learned flow from x0 vs true_path
    (true_path: (k, m)). Mirrors the synthetic forecast metric."""
    x, _ = model.forecast(torch.as_tensor(x0[None].astype(np.float32)), n_step=k)
    pred = x.detach().cpu().numpy()[1:, 0, :]
    A = np.concatenate([pred, np.ones((k, 1))], 1)
    W, *_ = np.linalg.lstsq(A, true_path, rcond=None)
    sse = ((true_path - A @ W) ** 2).sum(); tss = ((true_path - true_path.mean(0)) ** 2).sum() + 1e-12
    return float(1 - sse / tss)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest test/test_v1_graf.py -k "decode or torus" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/v1_graf/eval.py test/test_v1_graf.py
git commit -m "v1_graf: eval - orientation decoding, torus embedding, forecast R2"
```

---

## Task 7: M1 driver + smoke run (U6)

**Files:**
- Create: `experiments/v1_graf/run_m1.py`
- Test: `test/test_v1_graf.py` (a `--quick` smoke gated on the data file)

- [ ] **Step 1: Implement the driver**

`run_m1.py`: load `array_5`, bin at 10 ms, build the well-tuned mask (save it; M1 uses the 63-neuron set and logs N kept), 80/20 trial split per direction (seed 20260609), construct `VJF.make_model(ydim=N, xdim=L, udim=0, n_rbf=200, hidden_sizes=[100,100], likelihood='poisson', transition_flow='srrls', encoder='projection')` for `L in {3,4}`, an `OnlineReadout(N, L, smooth_tau=..., refresh_K=1000, link='log')` warm-started from the coverage trials, stream train trials through `online_filter_trials(..., warmup_trials=ceil(coverage), seed_centers=lambda s: kmeans_centers(s, 200))`, freeze, then eval on test trials: `leave_one_neuron_rates` -> `predictive_ll_bits_per_spike`; per-trial latent summary -> `orientation_decode_acc` + `torus_embedding`; `forecast_r2`; per-bin timing from `OnlineResult.elapsed_s`. Save everything to `experiments/v1_graf/results/m1_array5_bin10_L{L}.json` and a `provenance` block (commit, seeds, config, CPU). Accept `--quick` (few directions, few trials) for the smoke test.

Pattern to follow: `experiments/lc_poisson_stream/experiment.py` (config dict, provenance, chunked transition calls to avoid the O(T^2) OOM trap, streaming generation), and `examples/realtime_tutorial.py` (warm-start + `online_filter` wiring).

- [ ] **Step 2: Smoke test (gated on data presence)**

```python
# append to test/test_v1_graf.py
import os, pytest
DATA = os.path.join(os.path.dirname(__file__), "..", "experiments", "v1_graf", "data", "raw", "array_5.mat")
@pytest.mark.skipif(not os.path.exists(DATA), reason="Graf array_5 not downloaded")
def test_run_m1_quick():
    from experiments.v1_graf.run_m1 import main
    out = main(quick=True, latent_dim=3)
    assert out["pll_bits_per_spike"] > -1.0 and 0.0 <= out["decode_acc"] <= 1.0
    assert out["median_ms_per_bin"] > 0
```

- [ ] **Step 3: Run the smoke test**

Run: `uv run pytest test/test_v1_graf.py::test_run_m1_quick -v`
Expected: PASS (runs in seconds on a few directions/trials).

- [ ] **Step 4: Commit**

```bash
git add experiments/v1_graf/run_m1.py test/test_v1_graf.py
git commit -m "v1_graf: M1 driver (array_5 @ 10 ms) + quick smoke test"
```

---

## Task 8: Full M1 run + review

- [ ] **Step 1: Run the full M1 locally for a sanity pass** (`L=3`, 63 neurons, all trials)

Run: `uv run python experiments/v1_graf/run_m1.py --latent-dim 3`
Expected: a results JSON with PLL, decode acc, torus, forecast, timing; `diverge == 0`.

- [ ] **Step 2: If the local pass is healthy, run the L sweep (3 and 4) on GCP**

Use `/gcp_run` (git mode; push `feat/v1-graf` first). Two arms (L=3, L=4) fit well within the 24 h budget on one VM (CPU-bound; size by core speed, not GPU). Pull results to `experiments/v1_graf/results/`.

- [ ] **Step 3: Review results + decisions**

Run `/codex-review` on the diff and on the M1 result interpretation (PLL vs the vLGP-reported range, torus recovery, decode acc, forecast horizon, timing). Record findings + the go/no-go for M2 in `experiments/v1_graf/IMPL.md`.

- [ ] **Step 4: Commit the results summary + IMPL notes**

```bash
git add experiments/v1_graf/IMPL.md
git commit -m "v1_graf: M1 results summary + M2 go/no-go"
```

---

## Notes / deferred to follow-on plans
- U5 baselines (vLGP / GPFA / PLDS) + vanilla-VJF collapse arm -> `PLAN_baselines.md` once M1's sVJF numbers are in hand (vLGP is Yuan's code; GPFA/PLDS via existing implementations).
- M2 (5 ms, 1 ms) and M3 (arrays 1-4) -> `PLAN_M2.md`, `PLAN_M3.md`.
- Spike-history term: only if the PLL gap to vLGP demands it (stretch).
- The well-tuned mask + signal metric are computed at first run; if the data is huge, do the binning once and cache `(trial,bin,neuron)` arrays under `data/` (gitignored).
