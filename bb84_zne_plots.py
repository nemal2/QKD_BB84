"""
bb84_zne_plots.py
==================
Phase 5 (ZNE-QBER) visualisation module — matches bb84_ieee_style exactly,
same footprint/colour/threshold conventions as bb84_phase3_plots.py.

Seven plot functions, one per E11 sub-result:

    plot_zne_extrapolation()        - E11a: QBER vs f_scale, fitted lines -> f=0
    plot_bias_comparison()          - E11a: raw vs ZNE-linear bias, grouped bars
    plot_control_convergence()      - E11b: no-Eve intercept vs base p_dep, CI whiskers
    plot_discriminator_comparison() - E11c: D/R ratio vs ZNE error, single panel, twin y-axes
    plot_sensitivity_sweep()        - bias reduction vs base p_dep (robustness check)
    plot_threshold_crossing()       - E11e: QBER vs pEve with 5% crossing points marked
    plot_key_length_recovery()      - E11f: Shor-Preskill secure key length, raw vs ZNE
    plot_combined_noise_diagnostic()- E11d: linear vs quadratic curvature diagnostic

Style conventions applied throughout (per project house-style pass):
    - No in-axes titles / suptitles. Captions live outside the figure
      (e.g. in the paper or notebook markdown), not baked into the PNG.
    - A very light grid is drawn inside every axes box (low alpha, thin
      line, placed behind the data) via `_light_grid(ax)`.

University of Ruhuna - Dept. of Computer Engineering
MIT Licence - see LICENSE
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import bb84_ieee_style as style

_ALPHA_BAR = 0.82

# Colorblind-safe (Okabe–Ito) pair for the E11c twin-axis plot — chosen
# together deliberately for contrast, rather than pulled from style.C,
# since the two series share one panel and must stay distinguishable
# in print and in grayscale reproduction.
_COL_DR      = "#0072B2"   # blue   — D/R ratio (left axis)
_COL_ZNE_ERR = "#D55E00"   # vermillion — ZNE-linear error (right axis)

def _light_grid(ax, axis: str = "both") -> None:
    """Very light grid drawn behind the data, inside the axes box only."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color="grey", alpha=0.22, linewidth=0.5, zorder=0)


# ──────────────────────────────────────────────────────────────────────
# E11a  ZNE extrapolation — QBER vs f_scale with fitted lines to f=0
# ──────────────────────────────────────────────────────────────────────

def plot_zne_extrapolation(
    f_scales:      Sequence[float],
    p_eve_grid:    Sequence[float],
    mean_qbers:    Dict[float, Sequence[float]],   # p_eve -> qber(%) per f_scale
    fit_intercepts: Dict[float, float],             # p_eve -> a (linear fit intercept)
    fit_slopes:     Dict[float, float],              # p_eve -> b (linear fit slope)
    true_eve_fn=lambda p: p * 25.0,                  # theory: QBER_Eve = pEve/4 * 100
    save_path:     Optional[str] = None,
) -> plt.Figure:
    """
    Core E11a figure. One line per p_eve: measured points at each
    f_scale, dashed fitted line extending back to f=0, and a dotted
    horizontal marker at the true Eve-only QBER for that p_eve.
    This is the single most important figure in the ZNE section —
    it visually IS the extrapolation.

    Rendered as a square panel (rather than the wide 2-column
    footprint) with the x-axis ticked every 0.5 in f, since f_scales
    are naturally spaced in 0.5 steps.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    palette = [style.C["secure"], style.C["warning"], style.C["abort"],
               style.C["combined"], style.C["phase_damp"], style.C["fibre"]]

    f_arr = np.array(f_scales, dtype=float)
    f_line = np.linspace(0, max(f_arr) * 1.05, 60)

    all_y = []
    for p_eve, col in zip(p_eve_grid, palette):
        means = np.array(mean_qbers[p_eve])
        all_y.extend(means.tolist())
        ax.plot(f_arr, means, "o", color=col, ms=6, zorder=3,
                label=f"pEve={p_eve:.1f} (measured)")

        a, b = fit_intercepts[p_eve], fit_slopes[p_eve]
        ax.plot(f_line, a + b * f_line, "--", color=col, lw=1.3, alpha=0.65, zorder=2)

        true_q = true_eve_fn(p_eve)
        ax.plot(0, true_q, "*", color=col, ms=13, mec="black", mew=0.5, zorder=4)

    ax.axvline(0, color="grey", lw=0.9, zorder=1)
    ax.axvline(1.0, color="grey", lw=0.7, ls=":", alpha=0.6, zorder=1)
    ax.text(1.0, max(all_y) * 1.02, "f=1\n(as normally\nreported)",
            fontsize=style.ANNOT_FS, ha="center", va="bottom", color="#555555")

    ax.set_xlabel("Noise-Scaling Factor $f$")
    ax.set_ylabel("QBER (%)")
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    ax.set_xlim(left=-0.15)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.set_box_aspect(1)
    style.box_ticks(ax)
    _light_grid(ax)
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E11a  Bias comparison — raw vs ZNE-linear, grouped bars with CI
# ──────────────────────────────────────────────────────────────────────

def plot_bias_comparison(
    p_eve_grid: Sequence[float],
    raw_qbers:  Sequence[float],
    zne_qbers:  Sequence[float],
    zne_ci_low: Optional[Sequence[float]] = None,
    zne_ci_high: Optional[Sequence[float]] = None,
    true_eve_fn=lambda p: p * 25.0,
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart: true Eve QBER (reference line), raw QBER bars,
    ZNE-corrected bars with bootstrap CI error bars. This is the
    figure that makes "41% bias reduction" visually obvious at a
    glance rather than requiring the reader to parse a table.
    """
    n = len(p_eve_grid)
    x = np.arange(n)
    w = 0.35

    true_vals = [true_eve_fn(p) for p in p_eve_grid]

    fig, ax = plt.subplots(figsize=style.FIG_2COL)

    b1 = ax.bar(x - w/2, raw_qbers, w, label="Raw QBER (f=1)",
                color=style.C["warning"], alpha=_ALPHA_BAR, edgecolor="white", linewidth=0.6)

    if zne_ci_low is not None and zne_ci_high is not None:
        yerr = [
            [max(0, zq - lo) for zq, lo in zip(zne_qbers, zne_ci_low)],
            [max(0, hi - zq) for zq, hi in zip(zne_qbers, zne_ci_high)],
        ]
    else:
        yerr = None

    b2 = ax.bar(x + w/2, zne_qbers, w, label="ZNE-linear QBER (f=0)",
                color=style.C["secure"], alpha=_ALPHA_BAR, edgecolor="white", linewidth=0.6,
                yerr=yerr, capsize=3, ecolor="#333333")

    ax.plot(x, true_vals, "D", color="black", ms=7, zorder=5, label="True Eve QBER (theory)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.1f}" for p in p_eve_grid])
    ax.set_xlabel("Eve Intercept Probability (pEve)")
    ax.set_ylabel("QBER (%)")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    style.box_ticks(ax)
    _light_grid(ax, axis="y")
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E11b  Control — no-Eve intercept vs base p_dep, with CI whiskers
# ──────────────────────────────────────────────────────────────────────

def plot_control_convergence(
    p_dep_bases: Sequence[float],
    raw_f1:      Sequence[float],
    zne_mean:    Sequence[float],
    zne_ci_low:  Sequence[float],
    zne_ci_high: Sequence[float],
    pass_tolerance: float = 2.0,
    save_path:   Optional[str] = None,
) -> plt.Figure:
    """
    E11b positive-control figure: shows the ZNE-extrapolated no-Eve
    intercept staying near zero (within pass_tolerance) across a
    range of base noise levels, with bootstrap CI whiskers. This is
    the figure a reviewer looks at first to check the method isn't
    fabricating a false Eve signal.
    """
    fig, ax = plt.subplots(figsize=style.FIG_1COL)

    p_pct = [p * 100 for p in p_dep_bases]
    yerr = [
        [max(0, m - lo) for m, lo in zip(zne_mean, zne_ci_low)],
        [max(0, hi - m) for m, hi in zip(zne_mean, zne_ci_high)],
    ]

    ax.axhspan(0, pass_tolerance, alpha=0.08, color=style.C["secure"], zorder=0)
    ax.axhline(pass_tolerance, ls="--", lw=1.0, color=style.C["secure"], alpha=0.8,
               label=f"Pass tolerance ({pass_tolerance:.0f}pp)")
    ax.axhline(0, color="grey", lw=0.8, alpha=0.6)

    ax.plot(p_pct, raw_f1, "s--", color=style.C["warning"], lw=1.3, ms=6,
            alpha=0.8, label="Raw QBER (f=1)")
    ax.errorbar(p_pct, zne_mean, yerr=yerr, fmt="o-", color=style.C["secure"],
                lw=1.5, ms=7, capsize=4, label="ZNE-linear (f=0), 95% CI")

    ax.set_xlabel(r"Base Depolarising Probability $p_{\mathrm{dep}}$ (%)")
    ax.set_ylabel("QBER at $f=0$ (%)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(bottom=-0.5)
    style.box_ticks(ax)
    _light_grid(ax)
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E11c  Discriminator comparison — D/R ratio vs ZNE error, single
#       panel with twin y-axes (left = D/R ratio, right = ZNE error)
# ──────────────────────────────────────────────────────────────────────

def plot_discriminator_comparison(
    p_eve_grid:  Sequence[float],
    dr_ratios:   Sequence[float],
    zne_errors:  Sequence[float],
    save_path:   Optional[str] = None,
) -> plt.Figure:
    """
    ...
    """
    fig, ax1 = plt.subplots(figsize=style.FIG_1COL)

    col_dr  = _COL_DR
    col_err = _COL_ZNE_ERR

    l1, = ax1.plot(p_eve_grid, dr_ratios, "o-", color=col_dr, lw=1.8, ms=7,
                    label="Basis-resolved D/R ratio", zorder=3)
    ax1.axhline(1.0, ls=":", color="grey", lw=1.0, alpha=0.7, zorder=1)
    ax1.set_xlabel("Eve Intercept Probability (pEve)")
    ax1.set_ylabel("Diagonal / Rectilinear QBER Ratio", color=col_dr)
    ax1.tick_params(axis="y", labelcolor=col_dr)
    ax1.set_ylim(bottom=0)
    style.box_ticks(ax1)
    _light_grid(ax1)

    ax2 = ax1.twinx()
    l2, = ax2.plot(p_eve_grid, zne_errors, "s--", color=col_err, lw=1.6, ms=6,
                    label="ZNE-linear |error| (pp)", zorder=3)
    ax2.set_ylabel("|ZNE-linear QBER \u2212 True Eve QBER| (pp)", color=col_err)
    ax2.tick_params(axis="y", labelcolor=col_err)
    ax2.set_ylim(bottom=0)
    ax2.grid(False)

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()],
               loc="upper right", fontsize=6.5)

    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# Sensitivity sweep — bias reduction vs base p_dep (robustness check)
# ──────────────────────────────────────────────────────────────────────

def plot_sensitivity_sweep(
    p_dep_bases:    Sequence[float],
    bias_raw:       Sequence[float],
    bias_zne:       Sequence[float],
    save_path:      Optional[str] = None,
) -> plt.Figure:
    """
    Shows whether the bias-reduction result generalises beyond the
    single p_dep=0.03 masking point used in E11a, or was specific to
    it. Directly addresses "was this cherry-picked?" before a
    reviewer can ask.
    """
    fig, ax = plt.subplots(figsize=style.FIG_1COL)
    p_pct = [p * 100 for p in p_dep_bases]

    ax.plot(p_pct, bias_raw, "s--", color=style.C["warning"], lw=1.5, ms=7,
            label="Raw QBER mean |bias|")
    ax.plot(p_pct, bias_zne, "o-", color=style.C["secure"], lw=1.8, ms=7,
            label="ZNE-linear mean |bias|")
    ax.fill_between(p_pct, bias_zne, bias_raw,
                     where=[z <= r for z, r in zip(bias_zne, bias_raw)],
                     alpha=0.12, color=style.C["secure"], interpolate=True,
                     label="ZNE improvement region")

    ax.set_xlabel(r"Base Depolarising Probability $p_{\mathrm{dep}}$ (%)")
    ax.set_ylabel("Mean Absolute Bias vs. True Eve QBER (pp)")
    ax.legend(loc="best", fontsize=6.5)
    ax.set_ylim(bottom=0)
    style.box_ticks(ax)
    _light_grid(ax)
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E11e  Threshold crossing — QBER vs pEve with 5% crossing marked
# ──────────────────────────────────────────────────────────────────────

def plot_threshold_crossing(
    p_eve_grid: Sequence[float],
    raw_curve:  Sequence[float],
    zne_curve:  Sequence[float],
    p_thresh_raw: Optional[float],
    p_thresh_zne: Optional[float],
    p_thresh_true: float,
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Shows the raw and ZNE-corrected QBER curves against pEve, with
    vertical markers at the pEve where each crosses the 5% WARNING
    threshold, compared to the true crossing point. Directly
    visualises the "earlier/more accurate detection" claim.
    """
    fig, ax = plt.subplots(figsize=style.FIG_1COL)
    style.threshold_bands(ax, y_max=max(max(raw_curve), max(zne_curve)) * 1.2, labels=True)

    ax.plot(p_eve_grid, raw_curve, "s--", color=style.C["warning"], lw=1.5, ms=6,
            label="Raw QBER (f=1)")
    ax.plot(p_eve_grid, zne_curve, "o-", color=style.C["secure"], lw=1.8, ms=6,
            label="ZNE-linear QBER (f=0)")

    ax.axvline(p_thresh_true, color="black", ls=":", lw=1.2, alpha=0.8,
               label=f"True crossing (pEve={p_thresh_true:.2f})")
    if p_thresh_raw is not None:
        ax.axvline(p_thresh_raw, color=style.C["warning"], ls="--", lw=1.0, alpha=0.7)
    if p_thresh_zne is not None:
        ax.axvline(p_thresh_zne, color=style.C["secure"], ls="--", lw=1.0, alpha=0.7)

    ax.set_xlabel("Eve Intercept Probability (pEve)")
    ax.set_ylabel("QBER (%)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(bottom=0)
    style.box_ticks(ax)
    _light_grid(ax)
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E11f  Key-length recovery — Shor-Preskill, raw vs ZNE
# ──────────────────────────────────────────────────────────────────────

def plot_key_length_recovery(
    p_eve_grid: Sequence[float],
    l_raw:      Sequence[float],
    l_zne:      Sequence[float],
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    The "so what" figure: secure key bits recoverable under privacy
    amplification, raw vs ZNE-corrected QBER input, same sifted-key
    length. Most concrete, least abstract figure in the section —
    good candidate for the paper's final ZNE figure.
    """
    n = len(p_eve_grid)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=style.FIG_2COL)
    b1 = ax.bar(x - w/2, l_raw, w, label="Key length from raw QBER",
                color=style.C["warning"], alpha=_ALPHA_BAR, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w/2, l_zne, w, label="Key length from ZNE-corrected QBER",
                color=style.C["secure"], alpha=_ALPHA_BAR, edgecolor="white", linewidth=0.6)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=style.ANNOT_FS)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.1f}" for p in p_eve_grid])
    ax.set_xlabel("Eve Intercept Probability (pEve)")
    ax.set_ylabel("Shor\u2013Preskill Secure Key Length (bits)")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    style.box_ticks(ax)
    _light_grid(ax, axis="y")
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E11d  Combined-noise diagnostic — linear vs quadratic curvature
# ──────────────────────────────────────────────────────────────────────

def plot_combined_noise_diagnostic(
    p_eve_grid: Sequence[float],
    true_eve:   Sequence[float],
    zne_linear: Sequence[float],
    zne_quad:   Sequence[float],
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Limitation figure for E11d: shows both linear and quadratic
    extrapolation failing to track the true Eve contribution under
    combined T1+T2 noise, with the no-Eve point highlighted. This is
    meant to visually SUPPORT the "not universally applicable"
    limitation claim, not to sell a positive result.
    """
    fig, ax = plt.subplots(figsize=style.FIG_1COL)

    ax.plot(p_eve_grid, true_eve, "D-", color="black", lw=1.5, ms=6,
            label="True Eve QBER (theory)")
    ax.plot(p_eve_grid, zne_linear, "s--", color=style.C["combined"], lw=1.5, ms=6,
            label="ZNE-linear")
    ax.plot(p_eve_grid, zne_quad, "^:", color=style.C["abort"], lw=1.5, ms=6,
            label="ZNE-quadratic")

    ax.scatter([p_eve_grid[0]], [zne_linear[0]], s=140, facecolors="none",
               edgecolors=style.C["combined"], linewidths=1.5, zorder=5)
    ax.scatter([p_eve_grid[0]], [zne_quad[0]], s=140, facecolors="none",
               edgecolors=style.C["abort"], linewidths=1.5, zorder=5)
    ax.annotate("no-Eve\ncontrol point\n(should be ~0)",
                xy=(p_eve_grid[0], max(zne_linear[0], zne_quad[0])),
                xytext=(p_eve_grid[0] + 0.06, max(zne_linear[0], zne_quad[0]) + 2),
                fontsize=style.ANNOT_FS, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ax.set_xlabel("Eve Intercept Probability (pEve)")
    ax.set_ylabel("QBER (%)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(bottom=-1)
    style.box_ticks(ax)
    _light_grid(ax)
    plt.tight_layout()
    return style.save(fig, save_path)