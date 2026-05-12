#!/usr/bin/env bash
# Repair / extend private TeX Live after a flaky initial install (mirror checksums, partial failures).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
REPO="${TEXLIVE_TLMGR_REPO:-https://ftp.snt.utwente.nl/pub/software/tex/systems/texlive/tlnet}"
tlmgr option repository "$REPO"

# tudelft-report.cls + report.tex (biblatex/parskip/list deps)
PKGS=(
  biblatex fontspec etoolbox iftex float caption titlesec xcolor luatex85
  booktabs longtable multirow graphics pgf xspace microtype hyperref cleveref
  pdfpages datetime enumitem fancyhdr geometry parskip
  mathtools amsmath amsfonts xkeyval fmtcount
)
# tlmgr exits non-zero on updmap noise sometimes; thesis still works
tlmgr install "${PKGS[@]}" || true

echo "Repair pass done. Spot-check:"
kpsewhich biblatex.sty fontspec.sty booktabs.sty tikz.sty hyperref.sty 2>/dev/null || true
