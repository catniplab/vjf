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


def _principal_angle(a, b):
    """Largest principal angle (radians) between the column spaces of a and b."""
    qa = np.linalg.qr(np.asarray(a))[0]
    qb = np.linalg.qr(np.asarray(b))[0]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))


def _rot_angle_deg(R):
    """Rotation angle (deg) of an orthogonal R; exact for 2x2, else via the trace."""
    R = np.asarray(R)
    if R.shape == (2, 2):
        return float(abs(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))
    c = (np.trace(R) - (R.shape[0] - 2)) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


@torch.no_grad()
def apply_latent_rotation(transition, M):
    """Re-express the RBF flow under a latent coordinate change ``x_new = M x_old`` (orthogonal
    ``M``, m x m), so the dynamics are unchanged up to that rotation: ``centroid[:, :m] @= M^T``
    and ``w_mean @= M^T`` (the srrls feature-space covariance is invariant to an output rotation,
    so ``w_chol`` is untouched). udim>0: only the first m (state) centroid columns rotate."""
    feat = transition.velocity.feature
    m = np.asarray(M).shape[0]
    Mt = torch.as_tensor(np.asarray(M).T, dtype=feat.centroid.dtype, device=feat.centroid.device)
    feat.centroid.data[:, :m] = feat.centroid.data[:, :m] @ Mt
    wm = transition.velocity.w_mean
    new_wm = wm @ Mt
    if isinstance(wm, torch.nn.Parameter):
        wm.data.copy_(new_wm)
    else:
        transition.velocity.w_mean = new_wm


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
                 log_c: float = 1e-2, refresh_K: int = 1000, link: str = 'log',
                 column_norm: str = 'eig'):
        if link not in ('log', 'identity'):
            raise ValueError(f"link must be 'log' or 'identity', got {link!r}")
        if column_norm not in ('eig', 'unit'):
            raise ValueError(f"column_norm must be 'eig' or 'unit', got {column_norm!r}")
        self.n, self.m = n_obs, latent_dim
        self.alpha = 1.0 / smooth_tau
        self.c, self.K, self.link = log_c, refresh_K, link
        # 'eig': columns scaled by sqrt(eigenvalue) -> unit-variance latent (original).
        # 'unit': unit-norm columns -> the scale gauge lives in the latent, not C; this
        # removes the per-refresh sqrt(eigenvalue) rescaling that ratchets the latent scale.
        self.column_norm = column_norm
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
        """Loading from the current PCA vectors. 'eig' folds sqrt(eigenvalue) into the
        columns (unit-variance latent); 'unit' returns unit-norm columns (the scale
        gauge then lives in the latent, avoiding the per-refresh rescaling)."""
        cols = []
        for v in self.vecs:
            nv = np.linalg.norm(v) + 1e-12
            cols.append(v / nv if self.column_norm == 'unit' else v / nv * math.sqrt(nv))
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
        # Seed vecs at the SAME scale the streaming update() converges to: ||v_i|| = lambda_i
        # (the variance). eigh(scatter) gives eigenvalues ~ T*lambda, so divide by T -> the
        # covariance. Without this, vecs start T-fold too large and migrate down over training,
        # shrinking C and inflating the latent (Var(x_enc) drifts from 1/T toward 1).
        w, v = np.linalg.eigh(lc.T @ lc / lc.shape[0])
        self.vecs = [(v[:, -1 - i] * w[-1 - i]).copy() for i in range(self.m)]
        self.C = self._scaled_C()
        self.C_pinv = np.linalg.pinv(self.C).astype(np.float32)
        self.nu = np.zeros(self.n)          # reset EMA for the streaming phase
        return self.C.copy(), self.mean_b.astype(np.float32).copy()

    def feature(self, y, update_mean: bool = True) -> np.ndarray:
        f = self._feat(y)
        if update_mean:
            self._accumulate_mean(f)
        return f

    def set_fixed(self, C, b):
        """Freeze the projection to a known (C, b) (e.g. oracle); the EMA still runs but
        the running mean and PCA state are not updated (use feature(update_mean=False))."""
        self.C = np.asarray(C, dtype=np.float32)
        self.C_pinv = np.linalg.pinv(self.C).astype(np.float32)
        self.mean_b = np.asarray(b, dtype=np.float64).ravel()

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
    def maybe_refresh(self, decoder, step: int, *, transition=None,
                      track_subspace: bool = False, oracle_C=None):
        """Every ``K`` steps, Procrustes-anchor the refreshed ``C`` to the current decoder and
        write ``(C, b)`` into it (refreshing cached ``pinv(C)``). Returns a dict of
        subspace-drift metrics (or ``None`` if no refresh fired).

        If ``track_subspace`` and a ``transition`` is given, also rotate the RBF flow by the
        orthogonal part of the old->new latent-frame map, so the dynamics travel with the factor
        subspace instead of chasing a moving alignment (E1 subspace-tracking fix). ``oracle_C``
        (if given) is used only to log the principal angle to the true loading."""
        if self.K <= 0 or step % self.K != 0:
            return None
        w = decoder.decode.weight
        C_prev = w.detach().cpu().numpy()
        C_raw = self._scaled_C()
        u, _, vt = np.linalg.svd(C_raw.T @ C_prev)
        rstar = u @ vt                                   # Procrustes alignment (new -> prev)
        c_aligned = (C_raw @ rstar).astype(np.float32)
        cpinv = np.linalg.pinv(c_aligned).astype(np.float32)
        t_map = cpinv @ C_prev                           # old-latent -> new-latent (m x m)
        uq, _, vtq = np.linalg.svd(t_map)
        q = uq @ vtq                                     # orthonormal part of the frame map

        metrics = {
            "step": int(step),
            "refresh_rot_deg": _rot_angle_deg(rstar),
            "angle_prev_deg": float(np.degrees(_principal_angle(c_aligned, C_prev))),
            "angle_oracle_deg": (float(np.degrees(_principal_angle(c_aligned, np.asarray(oracle_C))))
                                 if oracle_C is not None else None),
            "frame_rot_deg": _rot_angle_deg(q),
            "cond": float(np.linalg.cond(c_aligned)),
        }
        self.C = c_aligned
        self.C_pinv = cpinv
        w.copy_(torch.as_tensor(self.C, device=w.device))
        decoder.decode.bias.copy_(torch.as_tensor(self.mean_b.astype(np.float32), device=w.device))
        if track_subspace and transition is not None:
            apply_latent_rotation(transition, q)
        return metrics

    @torch.no_grad()
    def impose_rotation(self, decoder, Q, *, transition=None, track_subspace=False):
        """E1(c) positive control: rotate the (fixed/oracle) readout by an orthogonal ``Q``
        (``C <- C Q``) -- imposing a known factor rotation with no estimation error -- and, if
        ``track_subspace``, rotate the flow to follow it (which should make the dynamics
        invariant). Mirrors ``maybe_refresh`` but with an externally supplied rotation."""
        Q = np.asarray(Q, dtype=np.float32)
        w = decoder.decode.weight
        C_new = (w.detach().cpu().numpy() @ Q).astype(np.float32)
        self.C = C_new
        self.C_pinv = np.linalg.pinv(C_new).astype(np.float32)
        w.copy_(torch.as_tensor(C_new, device=w.device))
        if track_subspace and transition is not None:
            apply_latent_rotation(transition, Q.T)       # latent change x_new = Q^T x_old
