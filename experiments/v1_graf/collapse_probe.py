"""Why does continued training collapse the residual-init grow flow mid-run?

Instruments ONE adam/grow L=4 lr1e-4 run (the clearest collapse: high through ep8,
dead ep12-40, recovers ep50) and tracks, over dense checkpoints, the mechanistic
signals that distinguish the candidate causes:

- forecast R2 (the thing being explained)
- mean velocity magnitude over the visited latent (flow -> identity? => static free-run)
- flow-map Jacobian spectral radius at the cycle (>1 expansive/diverge, <1 contractive)
- flow-weight norm ||w_mean|| (gradient over-sharpening / blow-up)
- latent spread and n_basis

A static collapse (free-run frozen) shows up as vel_mag -> 0; a divergent one as
jac_radius >> 1 with large vel_mag.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.v1_graf.single_dir import prepare_single_dir_data, _train_eval, FIGS
from experiments.v1_graf.figstyle import set_style, FW

CHK = (1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50)


def probe_fn(model, paths, freerun):
    """Mechanistic flow diagnostics at one checkpoint (model weights intact)."""
    vel = model.transition.velocity
    X = torch.as_tensor(np.asarray(paths[0]), dtype=torch.float32)     # inferred test path (nt, L)
    with torch.no_grad():
        v = vel(X, sampling=False)                                     # velocity at visited states
        vel_mag = float(v.norm(dim=1).mean())
        w_norm = float(vel.w_mean.norm())
    idx = np.unique(np.linspace(0, X.shape[0] - 1, 15).astype(int))    # subsample of cycle points
    radii = []
    with torch.enable_grad():
        for i in idx:
            J = torch.autograd.functional.jacobian(
                lambda z: model.transition.forward(z.unsqueeze(0), sampling=False).squeeze(0),
                X[i])
            radii.append(float(torch.linalg.eigvals(J).abs().max()))
    fr = np.asarray(freerun)
    return dict(w_norm=w_norm, vel_mag=vel_mag, jac_radius=float(np.mean(radii)),
                jac_max=float(np.max(radii)), freerun_disp=float(np.linalg.norm(fr[-1] - fr[0])),
                latent_std=float(np.asarray(paths[0]).std(0).mean()))


def main(dyn_noise=0.0, dyn_noise_decay=1.0, tag=""):
    set_style()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    d = prepare_single_dir_data(None)
    res = _train_eval(d["train_trials"], d["test_trials"], d["test_counts"], d["ybar"],
                      max(CHK), 4, d["N"], flow="sgd", optimizer="adam", lr=1e-4,
                      grow=True, grow_weight_init="residual", snapshot_epochs=CHK,
                      dyn_noise=dyn_noise, dyn_noise_decay=dyn_noise_decay, probe_fn=probe_fn)
    s = res["snapshots"]
    ep = [x["epoch"] for x in s]
    get = lambda k: [x[k] for x in s]
    for x in s:
        print(f"ep {x['epoch']:5.1f}: fc={x['fc']:+.2f} vel={x['vel_mag']:.3f} "
              f"jac={x['jac_radius']:.3f}(max {x['jac_max']:.2f}) |w|={x['w_norm']:.2f} "
              f"lat_std={x['latent_std']:.2f} nb={x['n_basis']}")

    panels = [("forecast R2", get("fc"), 0.0), ("velocity magnitude", get("vel_mag"), None),
              ("Jacobian spectral radius", get("jac_radius"), 1.0),
              ("flow weight ||w||", get("w_norm"), None),
              ("latent spread (std)", get("latent_std"), None), ("n_basis", get("n_basis"), None)]
    fig, ax = plt.subplots(2, 3, figsize=(FW(1.0), 4.6))
    for a, (title, y, ref) in zip(ax.ravel(), panels):
        a.plot(ep, y, marker="o", ms=3, color="0.2")
        if ref is not None:
            a.axhline(ref, color="C3", lw=0.8, ls=":")
        a.set_title(title); a.set_xlabel("epoch")
    fig.suptitle(f"adam/grow L=4 lr1e-4 (dyn_noise={dyn_noise}): mechanistic signals over training "
                 "(red dotted = R2 0 / Jacobian unit circle)", fontsize=9)
    sfx = ("_" + tag) if tag else ""
    out = os.path.join(FIGS, f"collapse_probe{sfx}.png")
    fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
    print("fig ->", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn-noise", type=float, default=0.0)
    ap.add_argument("--dyn-noise-decay", type=float, default=1.0)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    main(dyn_noise=a.dyn_noise, dyn_noise_decay=a.dyn_noise_decay, tag=a.tag)
