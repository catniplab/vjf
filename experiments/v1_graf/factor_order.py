"""Order VJF latent factors by the variance each explains in the decoded log-rate.

The VJF latent ``x`` is identifiable only up to an invertible linear transform
``(x, C) -> (M x, C M^{-1})`` (the decoded log-rate ``eta = C x + b`` is unchanged), so the
raw factor axes are not unique. This utility expresses the factors in the basis that orders
them by the variance each carries in the DECODED LOG-RATE ``eta = C x`` -- the
neural-data-relevant signal -- rather than in the (arbitrary-metric) latent itself.

With a centered reference latent path ``Xc`` (rows ``(x_t - xbar)^T``) and decoder loading
``C`` (N x L), the centered decoded log-rate is ``H = Xc C^T``; its SVD ``H = U S V_L^T``
(``V_L`` the top-L right singular vectors) gives

    W = C^T V_L,     z_t = W^T (x_t - xbar),

so factor ``i`` (``z_{t,i} = (C^T v_i)^T (x_t - xbar)``) is the latent projection explaining
the i-th largest share ``S_i^2 / sum_j S_j^2`` of the decoded-log-rate variance, with
``Z^T Z = S^2`` on the reference path. See report eq:rot.

IMPORTANT: ``W = C^T V_L`` is a decoder-induced linear projection (the decoded-log-rate
PC-score map), NOT an orthogonal rotation -- ``W^T W != I`` in general. When ``C`` has
eig-normalized columns (the scale-fixed CCIPCA loading, ``C = U Lambda^{1/2}``), the latent
is approximately whitened, so ``W`` is close to a reordering/rescaling of the existing
factors rather than a substantive recombination.
"""
from __future__ import annotations
import numpy as np


def decoded_variance_basis(x_ref: np.ndarray, C: np.ndarray, center: np.ndarray = None):
    """The decoded-log-rate PC-score map from a reference latent path.

    x_ref: (T, L) reference latent path (e.g. the trial-averaged latent) defining the order.
    C:     (N, L) decoder loading. center: (L,) latent center; default x_ref.mean(0).
    Returns (W (L, L), fve (L,), center (L,)) where ``W = C^T V_L`` and
    ``fve_i = S_i^2 / sum_j S_j^2`` is the fraction of decoded-log-rate variance per factor."""
    x_ref = np.asarray(x_ref, dtype=float)
    C = np.asarray(C, dtype=float)
    L = C.shape[1]
    center = x_ref.mean(0) if center is None else np.asarray(center, dtype=float)
    H = (x_ref - center) @ C.T                              # (T, N) centered decoded log-rate
    _, S, Vt = np.linalg.svd(H, full_matrices=False)
    W = C.T @ Vt[:L].T                                      # (L, L) = C^T V_L
    fve = (S[:L] ** 2) / (S[:L] ** 2).sum()
    return W, fve, center


def project(x: np.ndarray, W: np.ndarray, center: np.ndarray) -> np.ndarray:
    """z = (x - center) @ W  -- a latent path expressed in the decoded-variance-ordered basis."""
    return (np.asarray(x, dtype=float) - center) @ W
