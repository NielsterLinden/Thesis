#!/usr/bin/env bash
# Prefer private TeX Live on /data when present (thesis / LaTeX).
for y in 2026 2025; do
  TLBIN="/data/atlas/users/nterlind/texlive/${y}/bin/x86_64-linux"
  if [[ -d "$TLBIN" ]]; then
    export PATH="${TLBIN}:${PATH}"
    break
  fi
done

# Thesis fonts: Roboto Slab on /data + Arial→Liberation substitute (works when $HOME fontconfig is full).
THESIS_FC="/data/atlas/users/nterlind/texlive/fontconfig/thesis-fonts.conf"
if [[ -f "$THESIS_FC" ]]; then
  export FONTCONFIG_FILE="$THESIS_FC"
fi
