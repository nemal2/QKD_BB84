"""
bb84_ieee_style.py
===================
Single shared style sheet for every BB84 figure submitted to
IEEE Transactions on Quantum Engineering (TQE).

WHY THIS FILE EXISTS
---------------------
Up to now, each plotting script set its own fontsize="11", figsize=(9,5),
legend fontsize, tick style, etc., by hand. Every figure ended up slightly
different (different widths, different label sizes, duplicated threshold
lines). For a journal submission all figures must look like they came
from the same paper.

This module fixes that by being the *only* place that defines:

    - fonts / sizes        (via plt.rcParams, applied globally on import)
    - the colour palette    (C)
    - the two journal figure footprints (FIG_1COL, FIG_2COL, FIG_2PANEL)
    - the SECURE/WARNING/ABORT QBER threshold bands
    - tick styling (inward, boxed, with minor ticks)
    - a single `save()` helper

Every plotting file should start with:

    import bb84_ieee_style as style          # applies rcParams on import

and then use `style.FIG_1COL`, `style.C[...]`, `style.threshold_bands(ax, ...)`,
`style.box_ticks(ax)` and `style.save(fig, save_path)` instead of repeating
fontsize=... / figsize=... / colour hex codes inline.

No plot's *data* or *logic* changes by importing this — only the
typography, colour consistency and figure footprint are centralised.

University of Ruhuna – Dept. of Computer Engineering
MIT Licence – see LICENSE
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts & notebooks
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────────────────────────────
# 1. GLOBAL RC PARAMS — applied the moment this module is imported.
#    These numbers match the heatmap figure you already approved, so
#    every figure in the paper will look like it came from that one.
# ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    # Fonts
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",          # serif-matching math glyphs
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.titleweight":   "bold",
    "axes.labelsize":     8,
    "axes.labelweight":   "normal",
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "#888888",
    "figure.titlesize":   9,
    "figure.titleweight": "bold",

    # Resolution
    "figure.dpi":         150,             # on-screen preview
    "savefig.dpi":        300,             # exported PNG/PDF — journal quality

    # Boxed, inward ticks on all four spines (matches the heatmap axes)
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.bottom": True,
    "axes.spines.left":   True,
    "axes.linewidth":     0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.size":   2,
    "ytick.minor.size":   2,

    # Grid / lines
    "grid.linewidth":     0.45,
    "grid.alpha":         0.35,
    "grid.color":         "#aaaaaa",
    "lines.linewidth":    1.2,
    "lines.markersize":   4.5,

    # Saving
    "savefig.bbox":       "tight",
})

DPI = 300

# ──────────────────────────────────────────────────────────────────────
# 2. STANDARD FIGURE FOOTPRINTS (inches) — pick ONE of these per plot,
#    never a bespoke size. This is what makes every figure in the paper
#    sit at a consistent scale next to the others.
# ──────────────────────────────────────────────────────────────────────
FIG_1COL   = (4.8, 3.4)   # single-column: one axis, line/scatter/heatmap
FIG_2COL   = (9.6, 3.6)   # double-column / full-width: wide bar charts,
                          # multi-series comparisons, dual-axis plots
FIG_2PANEL = (9.6, 3.8)   # double-column: two side-by-side subplots (1x2)

# Small in-axes annotation text (value labels, contour clabels) —
# always this size, never ad-hoc fontsize=8 / 6.5 / 6.0 scattered around.
ANNOT_FS = 6.5

# ──────────────────────────────────────────────────────────────────────
# 3. COLOUR PALETTE — colour-blind-safe (Wong 2011) + IEEE-standard reds
#    for thresholds. One dictionary, reused by every figure.
# ──────────────────────────────────────────────────────────────────────
C = {
    "ideal":      "#009E73",
    "depolar":    "#E69F00",
    "amp_damp":   "#56B4E9",
    "phase_damp": "#CC79A7",
    "combined":   "#D55E00",
    "fibre":      "#0072B2",
    "eve":        "#F0E442",
    "secure":     "#2ca02c",
    "warning":    "#ff7f0e",
    "abort":      "#d62728",
    "qber":       "#0072B2",
    "rate":       "#D55E00",
    "theory":     "#555555",
    "rect":       "#0072B2",
    "diag":       "#D55E00",
    "agg":        "#333333",
}

THRESH_WARN  = 5.0    # %
THRESH_ABORT = 11.0   # %


# ──────────────────────────────────────────────────────────────────────
# 4. SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────

def threshold_bands(ax: plt.Axes, y_max: float = 30.0, labels: bool = False) -> None:
    """Draw SECURE / WARNING / ABORT shaded QBER regions + dashed lines.

    Call this once per QBER axis. Set ``labels=True`` only on the one
    call whose legend should show "Warning threshold (5%)" /
    "Abort threshold (11%)" — never call it more than once per axis,
    which previously caused duplicated dashed lines in some figures.
    """
    ax.axhspan(0,            THRESH_WARN,  alpha=0.06, color=C["secure"],  zorder=0)
    ax.axhspan(THRESH_WARN,  THRESH_ABORT, alpha=0.06, color=C["warning"], zorder=0)
    ax.axhspan(THRESH_ABORT, y_max,        alpha=0.06, color=C["abort"],   zorder=0)
    ax.axhline(THRESH_WARN,  ls="--", lw=1.0, color=C["warning"], alpha=0.8,
               label=f"Warning threshold ({THRESH_WARN:.0f}%)" if labels else None)
    ax.axhline(THRESH_ABORT, ls="--", lw=1.0, color=C["abort"],   alpha=0.8,
               label=f"Abort threshold ({THRESH_ABORT:.0f}%)" if labels else None)


def box_ticks(ax: plt.Axes, minor: bool = True) -> None:
    """Inward ticks on all four spines, with minor ticks — matches the
    heatmap axes exactly. Call once per axis after all plotting is done.
    """
    ax.tick_params(which="major", top=True, right=True, direction="in")
    if minor:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(which="minor", top=True, right=True, direction="in")


def save(fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
    """Save at journal DPI with a tight bbox, and print a confirmation."""
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"  [\u2713] Saved \u2192 {save_path}  ({DPI} dpi)")
    return fig
