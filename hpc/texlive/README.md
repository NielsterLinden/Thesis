# TeX Live on `/data` (thesis PDF)

The cluster `texlive-scheme-basic` RPMs are not enough for [`thesis_report/`](../thesis_report/) (missing `biber`, `biblatex`, `siunitx`, etc.). Install a private **TeX Live** under `/data/atlas/users/nterlind/texlive/`.

If many packages failed during the first run (checksum / mirror issues), fix the default mirror and pull critical packages:

```bash
export TEXLIVE_TLMGR_REPO='https://ftp.snt.utwente.nl/pub/software/tex/systems/texlive/tlnet'
/project/atlas/users/nterlind/Thesis-Code/hpc/texlive/repair_texlive.sh
```

The installer may suggest `tlmgr update --all --reinstall-forcibly-removed`; if `fmtutil` errors on obscure formats, **XeLaTeX** thesis builds can still work once `biblatex` / `fontspec` resolve via `repair_texlive.sh`.

## Install (once)

On a login or interactive CPU node with network access:

```bash
chmod +x /project/atlas/users/nterlind/Thesis-Code/hpc/texlive/install_texlive.sh
/project/atlas/users/nterlind/Thesis-Code/hpc/texlive/install_texlive.sh
```

Optional: faster mirror, e.g.

```bash
export TEXLIVE_MIRROR='https://<your-preferred-mirror>/systems/texlive/tlnet'
```

If the installer expects a different year than `2026`, edit [`texlive.profile`](texlive.profile) `TEXDIR` / `TEXMFSYS*` paths to match the release folder the installer creates.

## Environment

After install:

```bash
source /project/atlas/users/nterlind/Thesis-Code/hpc/texlive/env.sh
```

Or append the contents of [`bashrc.snippet`](bashrc.snippet) to `~/.bashrc` so every session prefers this TeX over `/usr/bin`.

## Fonts (XeLaTeX)

The TU class expects **Arial** and **Roboto Slab**. Install Roboto Slab (open license):

```bash
/project/atlas/users/nterlind/Thesis-Code/hpc/texlive/install_google_fonts.sh
```

**Arial** is not redistributable here; install MS core fonts or copy licensed TTFs into `~/.local/share/fonts/` and run `fc-cache -fv`.

## Build the thesis

From the repo:

```bash
source /project/atlas/users/nterlind/Thesis-Code/hpc/texlive/env.sh
cd /project/atlas/users/nterlind/Thesis-Code/thesis_report
./build.sh
```

Or see [`thesis_report/build.sh`](../thesis_report/build.sh).
