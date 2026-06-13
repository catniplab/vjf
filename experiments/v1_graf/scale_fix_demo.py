"""Figure: the CCIPCA warm-start scale fix, on stationary data.

The streaming CCIPCA update converges to the covariance scale (||v_i|| -> lambda_i, the
variance). The original warm_start seeded the scatter scale (||v_i|| ~ T*lambda_i), T-fold
too large, so C shrank ~sqrt(T) over training and the latent inflated (Var(x_enc) drifted
from 1/T toward 1). The fix divides the warm-start scatter by the sample count.

We run the REAL OnlineReadout (fixed = covariance warm-start) and a 'bug' replica that
re-scales the warm-start vectors back to scatter scale (x T), on the same stationary
Gaussian stream, and track ||v|| and Var(x_enc) over epochs.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from vjf.readout import OnlineReadout
from experiments.v1_graf.figstyle import set_style, FW

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "report_m1", "figs")
SEED = 20260613


def _track(rescale_bug: bool, n=30, m=2, T_warm=1120, epochs=50):
    rng = np.random.default_rng(SEED)
    U = np.linalg.qr(rng.standard_normal((n, n)))[0][:, :m]
    lam = np.array([4.0, 1.0])                                   # true variances along PCs
    draw = lambda k: (rng.standard_normal((k, m)) * np.sqrt(lam)) @ U.T + 0.01 * rng.standard_normal((k, n))
    ro = OnlineReadout(n, m, link="identity", refresh_K=0)
    ro.warm_start(draw(T_warm))
    if rescale_bug:                                              # recreate the scatter-scale seed
        ro.vecs = [v * T_warm for v in ro.vecs]
        ro.C = ro._scaled_C(); ro.C_pinv = np.linalg.pinv(ro.C).astype(np.float32)
    vnorm, xvar = [], []
    def rec():
        C = ro._scaled_C(); g = draw(4000); xenc = (np.linalg.pinv(C) @ (g - ro.mean_b).T).T
        vnorm.append(float(np.linalg.norm(ro.vecs[0]))); xvar.append(float(xenc.var(0).mean()))
    rec()
    for _ in range(epochs):
        for g in draw(T_warm):
            ro.update(ro.feature(g))
        rec()
    return np.array(vnorm), np.array(xvar), lam[0]


def main():
    set_style()
    vb, xb, lam0 = _track(rescale_bug=True)
    vf, xf, _ = _track(rescale_bug=False)
    ep = np.arange(len(vb))
    fig, ax = plt.subplots(1, 2, figsize=(FW(0.8), 2.6))
    ax[0].plot(ep, vb, "-", color="C3", label="scatter seed (bug)")
    ax[0].plot(ep, vf, "-", color="C0", label="covariance seed (fix)")
    ax[0].axhline(lam0, color="0.6", lw=0.7, ls=":")
    ax[0].set_yscale("log"); ax[0].set_title(r"PCA vector norm $\|v_1\|$ (true $\lambda_1{=}4$)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel(r"$\|v_1\|$"); ax[0].legend(fontsize=7)
    ax[1].plot(ep, xb, "-", color="C3"); ax[1].plot(ep, xf, "-", color="C0")
    ax[1].axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax[1].set_title(r"encoder-input variance $\mathrm{Var}(x_{enc})$")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("variance")
    fig.suptitle("CCIPCA warm-start scale: scatter seed drifts, covariance seed is pinned", fontsize=9)
    out = os.path.join(FIGS, "scale_fix.png")
    fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
    print(f"bug:  ||v|| {vb[0]:.1f}->{vb[-1]:.1f}  Var {xb[0]:.4f}->{xb[-1]:.4f}")
    print(f"fix:  ||v|| {vf[0]:.2f}->{vf[-1]:.2f}  Var {xf[0]:.4f}->{xf[-1]:.4f}")
    print("fig ->", out)


if __name__ == "__main__":
    main()
