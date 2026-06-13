# report.tex edit audit

## Pass 1 (2026-06-13/14): draft + codex-review + manuscript-revision

**Sources:** REPORT_M1.md, REPORT_single_dir_and_growth.md, IMPL.md, DESIGN.md, PLAN_M1.md,
and this session's results (single-dir SGD scan, collapse probe, noise sweep, the two
multi-direction scans).

**codex-review (gpt-5.5) findings:**
- [P2 reproducibility] report figures were untracked -> a fresh checkout could not build.
  FIXED: `Makefile`, the figures committed under `figs/`, slim staged data under `data/`,
  and `plot_multidir.py` / `scale_fix_demo.py` / `noise_sweep.py` regenerate them.
- No scientific-accuracy or overclaiming findings.

**manuscript-revision fixes applied:**
- Added the vLGP reference (`vlgp`); it was cited as "vLGP" without a bibliography entry.
- Trimmed changelog phrasing ("previously observed ...") to a direct statement of the
  inflation phenomenon.
- Dropped redundant/clipped on-figure titles (paper-figures); render each figure at the
  width it is included at (scale 1.0); captions carry the description.
- Section 7 + Table 1 + Fig 3 updated to the controlled refined comparison
  (srrls_base vs sgd_grow_noise vs sgd_grow_refined); the refinements help the dynamics
  arm but do not overtake the scale-fixed srrls_base on decode/PLL (the trade-off persists).

**Deferred:**
- Single-trial forecast in the multi-direction driver scores only test trial 0 (noisy);
  averaging over test trials would tighten Fig 3 right panel. Noted in the text.
- A separate collapse-probe figure (Jacobian spectral radius) could replace the prose-only
  treatment in Section 6 if the report is expanded.
