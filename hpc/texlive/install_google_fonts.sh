#!/usr/bin/env bash
# Install Roboto Slab (Apache 2.0) for TU thesis template; fonts on /data (not $HOME quota).
set -euo pipefail

FONTDIR="/data/atlas/users/nterlind/texlive/fonts/thesis"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

mkdir -p "$FONTDIR"
cd "$WORKDIR"
curl -fsSL "https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf" -o RobotoSlab-VF.ttf
cp RobotoSlab-VF.ttf "$FONTDIR/"

FCCD="${HOME}/.config/fontconfig/conf.d"
mkdir -p "$FCCD"
CONF="${FCCD}/99-thesis-texlive-fonts.conf"
if [[ ! -f "$CONF" ]]; then
  cat >"$CONF" <<EOF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>${FONTDIR}</dir>
</fontconfig>
EOF
  echo "Wrote fontconfig snippet: $CONF"
else
  echo "Fontconfig snippet already exists: $CONF"
fi

fc-cache -fv "$FONTDIR" 2>/dev/null || true
echo "Roboto Slab installed under $FONTDIR"
