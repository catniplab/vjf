# TEXINPUTS must resolve catniplab.sty and math_preamble.tex from memmingCommons/latex/.
# Do not commit local copies of those files. Set TEXINPUTS in your shell profile or via the
# bundled .envrc (direnv), or export before running make. Verify: kpsewhich catniplab.sty

DOC = main

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex references.bib
	latexmk -pdf -interaction=nonstopmode $(DOC).tex

clean:
	latexmk -c

.PHONY: all clean
