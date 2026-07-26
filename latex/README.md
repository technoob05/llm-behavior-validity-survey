# LaTeX source

The submission source is organised around one canonical paper entry point:

- `paper.tex`: main manuscript
- `references.bib`: bibliography
- `paper.pdf`: current full manuscript
- `figures/core/`: figures used in the main paper
- `figures/appendix/`: appendix-only conceptual figures
- `figures/icons/paper/`: icon assets for other paper figures
- `slides/`: ACL presentation source

From the repository root, build the manuscript with:

```powershell
Set-Location latex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

The public repository intentionally excludes anonymous review copies, review
notes, and downloaded source trees from other papers.
