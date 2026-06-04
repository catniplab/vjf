"""Online readout (loading matrix) estimation for the projection-input encoder.

Estimates the observation loading ``C`` and bias ``b`` ONLINE by incremental PCA
(CCIPCA) of a link-matched, causally-smoothed feature of the observations, decoupled
from the ELBO gradient (which collapses under sparse low-rate Poisson). The same
``(C, b)`` is used by the decoder and to form the recognition input projection
``pinv(C) (g~(y) - b)``.

See ``experiments/lc_poisson_stream/note_projection_encoder.tex`` for the math.
All estimation is NumPy/CPU on the observations; only the periodic write of ``(C, b)``
into a torch decoder touches torch.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _procrustes(c_new: np.ndarray, c_ref: np.ndarray) -> np.ndarray:
    """Orthogonal ``R`` (m x m) minimizing ||c_new R - c_ref||; returns ``c_new @ R``.

    Anchors the within-subspace rotation/sign of a refreshed loading to the previous
    one (the paper's "normalize C each iteration" for the ``C x = (C R)(R^-1 x)``
    identifiability)."""
    u, _, vt = np.linalg.svd(c_new.T @ c_ref)
    return c_new @ (u @ vt)


class OnlineReadout:
    """Streaming estimator of ``(C, b)`` + the input projection ``pinv(C)(g~(y)-b)``.

    Usage per bin (see the experiment driver)::

        g  = ro.feature(y_t)        # causal link-matched feature; updates running mean b
        ro.update(g)                # one incremental-PCA step
        x_enc = ro.project(g)       # pinv(C) (g - b), reused O(n m)
        ...                         # model.filter(y_t, y_enc=x_enc)
        ro.maybe_refresh(decoder, t)   # every K: Procrustes-anchored (C, b) -> decoder
    """

    def __init__(self, n_obs: int, latent_dim: int, *, smooth_tau: float = 8.0,
                 log_c: float = 1e-2, refresh_K: int = 1000, link: str = 'log'):
        if link not in ('log', 'identity'):
            raise ValueError(f"link must be 'log' or 'identity', got {link!r}")
        self.n, self.m = n_obs, latent_dim
        self.alpha = 1.0 / smooth_tau
        self.c, self.K, self.link = log_c, refresh_K, link
        self.nu = np.zeros(n_obs)            # causal EMA state
        self.mean_b = np.zeros(n_obs)        # running mean of the feature (= bias b)
        self.count = 0
        self.vecs = [np.zeros(n_obs) for _ in range(latent_dim)]  # eigvec*eigval estimates
        self.C = np.zeros((n_obs, latent_dim), dtype=np.float32)
        self.C_pinv = np.zeros((latent_dim, n_obs), dtype=np.float32)

    # --- features -------------------------------------------------------------
    def _feat(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.link == 'identity':
            return y
        self.nu = (1.0 - self.alpha) * self.nu + self.alpha * y    # causal EMA
        return np.log(self.nu + self.c)

    def _accumulate_mean(self, feat: np.ndarray) -> None:
        self.count += 1
        self.mean_b += (feat - self.mean_b) / self.count

    def _scaled_C(self) -> np.ndarray:
        """Loading from the current PCA vectors, scaled so the latent has unit variance
        (fold sqrt(eigenvalue) into the columns)."""
        cols = []
        for v in self.vecs:
            nv = np.linalg.norm(v) + 1e-12
            cols.append(v / nv * math.sqrt(nv))
        return np.stack(cols, 1).astype(np.float32)

    # --- API ------------------------------------------------------------------
    def warm_start(self, counts_window):
        """Causal feature over the initial window -> running mean + batch-PCA init of
        ``(C, b)``. Returns ``(C, b)`` (float32) to set the decoder; resets the EMA so
        streaming continues cleanly."""
        feats = []
        for y in counts_window:
            f = self._feat(y)
            self._accumulate_mean(f)
            feats.append(f)
        lc = np.asarray(feats) - self.mean_b
        w, v = np.linalg.eigh(lc.T @ lc)
        self.vecs = [(v[:, -1 - i] * w[-1 - i]).copy() for i in range(self.m)]
        self.C = self._scaled_C()
        self.C_pinv = np.linalg.pinv(self.C).astype(np.float32)
        self.nu = np.zeros(self.n)          # reset EMA for the streaming phase
        return self.C.copy(), self.mean_b.astype(np.float32).copy()

    def feature(self, y) -> np.ndarray:
        f = self._feat(y)
        self._accumulate_mean(f)
        return f

    def project(self, feat: np.ndarray) -> np.ndarray:
        """Recognition input ``pinv(C) (feat - b)`` (m,). O(n m); C_pinv cached."""
        return (self.C_pinv @ (np.asarray(feat) - self.mean_b)).astype(np.float32)

    def update(self, feat: np.ndarray) -> None:
        """One CCIPCA step on the centered feature (converges to batch PCA)."""
        u = (np.asarray(feat) - self.mean_b).copy()
        lr = 1.0 / max(self.count, 1)
        for i in range(self.m):
            v = self.vecs[i]
            vh = v / (np.linalg.norm(v) + 1e-12)
            self.vecs[i] = (1.0 - lr) * v + lr * (u @ vh) * u
            vh2 = self.vecs[i] / (np.linalg.norm(self.vecs[i]) + 1e-12)
            u = u - (u @ vh2) * vh2

    @torch.no_grad()
    def maybe_refresh(self, decoder, step: int) -> bool:
        """Every ``K`` steps, Procrustes-anchor the refreshed ``C`` to the current decoder
        and write ``(C, b)`` into it (and refresh the cached ``pinv(C)``). The projection
        encoder makes explicit flow re-rotation unnecessary (the anchored update keeps the
        latent frame ~fixed; the dynamics learner absorbs the small residual)."""
        if self.K <= 0 or step % self.K != 0:
            return False
        w = decoder.decode.weight
        c_aligned = _procrustes(self._scaled_C(), w.detach().cpu().numpy())
        self.C = c_aligned.astype(np.float32)
        self.C_pinv = np.linalg.pinv(self.C).astype(np.float32)
        w.copy_(torch.as_tensor(self.C, device=w.device))
        decoder.decode.bias.copy_(torch.as_tensor(self.mean_b.astype(np.float32), device=w.device))
        return True
