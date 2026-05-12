# Non-interactive TeX Live install for thesis PDF builds (see hpc/texlive/README.md).
# TeX Live 2026 layout; if you install an older release, retarget TEXDIR / TEXMFSYS*.

selected_scheme scheme-medium
tlpdbopt_install_docfiles 0
tlpdbopt_install_srcfiles 0

TEXDIR /data/atlas/users/nterlind/texlive/2026
TEXMFLOCAL /data/atlas/users/nterlind/texlive/texmf-local
TEXMFSYSCONFIG /data/atlas/users/nterlind/texlive/2026/texmf-config
TEXMFSYSVAR /data/atlas/users/nterlind/texlive/2026/texmf-var
TEXMFVAR /data/atlas/users/nterlind/texlive/texmf-var-user
TEXMFCONFIG /data/atlas/users/nterlind/texlive/texmf-config-user
TEXMFHOME /data/atlas/users/nterlind/texlive/texmf-home

portable 0
