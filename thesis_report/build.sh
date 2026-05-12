#!/usr/bin/env bash
# Build thesis_report/report.pdf with XeLaTeX + biber (private TeX Live on /data).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/hpc/texlive/env.sh"
cd "$(dirname "$0")"

ENGINE=(xelatex -interaction=nonstopmode -halt-on-error)

"${ENGINE[@]}" report.tex
biber report
"${ENGINE[@]}" report.tex
"${ENGINE[@]}" report.tex

echo "Built: $(pwd)/report.pdf"
ls -la report.pdf
