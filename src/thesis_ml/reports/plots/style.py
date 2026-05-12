"""Thesis plotting style — apply once per process, not per figure.

Usage
-----
At module level in every plots/* file:

    from thesis_ml.reports.plots.style import apply_thesis_style, figure_size, axis_color
    apply_thesis_style()

Then in each plotting function:

    fig, ax = plt.subplots(figsize=figure_size("full"))
    color = axis_color("E")   # positional encoding group
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Axis-group colour palette  (Wong / IBM colorblind-safe, TU Delft cyan for recommended)
# ---------------------------------------------------------------------------

AXIS_GROUP_COLORS: dict[str, str] = {
    "baseline":     "#222222",   # near-black — reference / baseline models
    "recommended":  "#00A6D6",   # TU Delft cyan — recommended configuration
    "D":            "#0072B2",   # data treatment — dark blue
    "T":            "#56B4E9",   # tokenizer — sky blue
    "E":            "#009E73",   # positional encoding — green
    "B":            "#008891",   # physics biases — teal
    "A":            "#CC79A7",   # attention / block structure — pink
    "F":            "#E69F00",   # FFN — amber
    "C":            "#D55E00",   # head / pooling — vermillion
    "H":            "#666666",   # scaling — gray
}

# Default categorical cycle: D, T, E, B, A, F, C, H  (colorblind-safe order)
_CATEGORICAL_ORDER: list[str] = ["D", "T", "E", "B", "A", "F", "C", "H"]
CATEGORICAL_COLORS: list[str] = [AXIS_GROUP_COLORS[k] for k in _CATEGORICAL_ORDER]

# ---------------------------------------------------------------------------
# rcParams dict — apply with plt.rcParams.update(THESIS_RC)
# ---------------------------------------------------------------------------

THESIS_RC: dict[str, object] = {
    # Font
    "font.family":               "sans-serif",
    "font.sans-serif":           ["DejaVu Sans"],
    "mathtext.fontset":          "dejavusans",
    "font.size":                 9,
    "axes.labelsize":            9,
    "xtick.labelsize":           8,
    "ytick.labelsize":           8,
    "legend.fontsize":           8,
    "legend.title_fontsize":     9,
    "axes.titlesize":            10,
    # Spines
    "axes.spines.top":           False,
    "axes.spines.right":         False,
    # Grid
    "axes.grid":                 True,
    "grid.alpha":                0.25,
    "grid.linewidth":            0.5,
    "grid.linestyle":            "--",
    # Lines
    "lines.linewidth":           1.5,
    "lines.markersize":          4,
    "patch.linewidth":           0.8,
    # Legend
    "legend.framealpha":         0.8,
    "legend.edgecolor":          "0.8",
    "legend.borderpad":          0.4,
    # Save defaults
    "savefig.dpi":               300,
    "savefig.format":            "pdf",
    "savefig.bbox":              "tight",
    "savefig.pad_inches":        0.05,
    # Colour cycle: D → T → E → B → A → F → C → H
    "axes.prop_cycle":           plt.cycler("color", CATEGORICAL_COLORS),
}

# ---------------------------------------------------------------------------
# TU Delft A4 text-width constants  (text width ≈ 160 mm = 6.30 in)
# ---------------------------------------------------------------------------

_GOLDEN = 1.618033988749895

_WIDTH_INCHES: dict[str, float] = {
    "full":       6.30,    # full text width
    "two_thirds": 4.20,    # two-thirds text width
    "half":       3.00,    # half text width
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_thesis_style() -> None:
    """Apply THESIS_RC to the global matplotlib rcParams.

    Call once at module level in every plots/* file. Safe to call multiple
    times (idempotent — last write wins, which is always the same dict).
    """
    plt.rcParams.update(THESIS_RC)


def figure_size(width: str = "full", aspect: float | None = None) -> tuple[float, float]:
    """Return (width_in, height_in) for a thesis figure.

    Parameters
    ----------
    width : str
        One of ``"full"`` (6.30 in), ``"two_thirds"`` (4.20 in),
        or ``"half"`` (3.00 in).
    aspect : float | None
        Height = width / aspect. Defaults to the golden ratio (1.618).
        Pass ``aspect=1.0`` for square plots (confusion matrices, heatmaps).

    Returns
    -------
    tuple[float, float]
        ``(width_inches, height_inches)``
    """
    if width not in _WIDTH_INCHES:
        raise ValueError(
            f"width must be one of {list(_WIDTH_INCHES)}, got {width!r}"
        )
    w = _WIDTH_INCHES[width]
    h = w / (_GOLDEN if aspect is None else aspect)
    return (w, h)


def axis_color(group_id: str) -> str:
    """Return the hex colour for an axis-group identifier.

    Parameters
    ----------
    group_id : str
        Key in ``AXIS_GROUP_COLORS`` (e.g. ``"E"``, ``"baseline"``).

    Raises
    ------
    KeyError
        If ``group_id`` is not a recognised axis-group key.
    """
    if group_id not in AXIS_GROUP_COLORS:
        raise KeyError(
            f"Unknown axis group {group_id!r}. "
            f"Valid keys: {list(AXIS_GROUP_COLORS)}"
        )
    return AXIS_GROUP_COLORS[group_id]
