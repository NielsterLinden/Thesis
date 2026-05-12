#!/usr/bin/env bash
# One-time TeX Live (scheme-medium) under /data for XeLaTeX + biber thesis builds.
set -euo pipefail

ROOT="/data/atlas/users/nterlind/texlive"
WORKDIR="${ROOT}/install-work"
MIRROR="${TEXLIVE_MIRROR:-https://mirror.ctan.org/systems/texlive/tlnet}"
PROFILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/texlive.profile"

if [[ ! -f "$PROFILE" ]]; then
  echo "Missing profile: $PROFILE" >&2
  exit 1
fi

for y in 2026 2025; do
  if [[ -x "${ROOT}/${y}/bin/x86_64-linux/xelatex" ]]; then
    echo "TeX Live already present at ${ROOT}/${y} — remove that tree to reinstall."
    exit 0
  fi
done

mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "Downloading install-tl from $MIRROR ..."
curl -fsSL "${MIRROR}/install-tl-unx.tar.gz" -o install-tl-unx.tar.gz
rm -rf install-tl
tar -xzf install-tl-unx.tar.gz
cd "$(find . -maxdepth 1 -type d -name 'install-tl-*' | head -1)"

echo "Installing (scheme-medium, ~3–5 GiB). This takes several minutes."
./install-tl -no-gui -profile="$PROFILE"

# Default mirror.ctan.org can redirect to broken mirrors; prefer a stable HTTPS mirror for tlmgr.
TLMGR_REPO="${TEXLIVE_TLMGR_REPO:-https://ftp.snt.utwente.nl/pub/software/tex/systems/texlive/tlnet}"
if [[ -x "${ROOT}/2026/bin/x86_64-linux/tlmgr" ]]; then
  echo "Setting tlmgr default repository to ${TLMGR_REPO}"
  "${ROOT}/2026/bin/x86_64-linux/tlmgr" option repository "$TLMGR_REPO" || true
fi
