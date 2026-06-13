# V1 Graf sVJF tech report.
# TEXINPUTS must resolve catniplab.sty + math_preamble.tex from memmingCommons/latex/
# (set in your shell profile or paper/.envrc). Verify: kpsewhich catniplab.sty
#
# Figures live in figs/ and are regenerated from committed scripts + staged data:
#   figs/scale_fix.{png,pdf}              <- ../scale_fix_demo.py        (self-contained)
#   figs/noise_sweep.{png,pdf}            <- ../noise_sweep.py           (trains; stages results/noise_sweep.json)
#   figs/scan_multidir_summary.{png,pdf}  <- ../plot_multidir.py <scan_multidir*.json>  (reads a scan JSON)
# The scan JSON itself comes from ../scan_multidir.py on GCP (see ../IMPL.md).

DOC = report

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex
	latexmk -pdf -interaction=nonstopmode $(DOC).tex

clean:
	latexmk -c

.PHONY: all clean
