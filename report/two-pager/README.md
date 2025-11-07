Two‑pager (Finnish) – TAU thesis template

Fill metadata in `main.tex`:
- `\title{FI}{EN (optional)}`
- `\author{...}`, `\examiner[Tarkastaja]{...}`
- `\finishdate{YYYY}{MM}{DD}`
- `\facultyname{...}{...}`, `\programmename{...}{...}`

Write the text
- Put ~600-word Finnish body into `tex/text.tex`.
- Must be understandable standalone; images/tables allowed but not counted.
- Subheadings optional; if used, ≥2 paragraphs under each.
- Cite with `\cite{...}`; bibliography prints on its own page.

Bibliography
- Reuses `../tex/references.bib`. Ensure cited keys exist there.

Build
- From this directory: `make`
- Or manually: `pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex`

Clean
- `make clean`


