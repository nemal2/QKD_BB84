"""
bb84_phase3_plots.py
====================
Phase 3 visualisation module for the BB84 QKD simulator — IEEE TQE style.

Six plot functions, matching the Phase 3 + Phase 4 experiment types:

    plot_noise_model_comparison()   - bar chart: all noise models side-by-side
    plot_depolar_sweep()            - QBER vs depolarising probability p
    plot_t1_sweep()                 - QBER vs T1 (amplitude damping)
    plot_t2_sweep()                 - QBER vs T2 (phase damping)
    plot_fibre_loss()               - QBER + key-rate vs channel distance (dual axis)
    plot_depolar_vs_eve_heatmap()   - QBER heatmap: depolarising p x Eve intercept p

Every function returns the matplotlib Figure object so callers can
inspect, save again, or embed it elsewhere. A ``save_path`` parameter
is accepted by every function for convenience.

STYLE
-----
All fonts, figure sizes, colours, tick styling and the SECURE/WARNING/
ABORT QBER bands come from ``bb84_ieee_style`` (see that file). Nothing
in this module sets its own fontsize, figsize or hex colour directly —
that is what keeps all six figures visually identical for the paper.
Only the data/content of each plot differs.

Usage example
-------------
>>> from bb84_phase3_plots import plot_fibre_loss
>>> fig = plot_fibre_loss(distances, qbers, key_rates,
...                        save_path='images/fibre_loss.png')

University of Ruhuna - Dept. of Computer Engineering
MIT Licence - see LICENSE
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np

import bb84_ieee_style as style

_ALPHA_BAR = 0.82


# ──────────────────────────────────────────────────────────────────────
# E1  Noise-model comparison  (bar chart)
# ──────────────────────────────────────────────────────────────────────

def plot_noise_model_comparison(
    labels:    List[str],
    qbers:     List[float],      # percentages
    key_rates: List[float],      # percentage of n_transmitted
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing QBER (%) and key-generation rate (%)
    across all noise models in a single figure.

    Parameters
    ----------
    labels    : model names (x-axis).
    qbers     : QBER in percent for each model.
    key_rates : key-generation rate (key bits / qubit) x 100 for each model.
    save_path : optional path to save the PNG.
    """
    n = len(labels)
    x = np.arange(n)
    w = 0.35

    colours_q = [
        style.C["ideal"], style.C["depolar"], style.C["amp_damp"],
        style.C["phase_damp"], style.C["combined"], style.C["fibre"], style.C["eve"],
    ][:n]
    colours_r = [c + "99" for c in colours_q]   # semi-transparent twin

    fig, ax1 = plt.subplots(figsize=style.FIG_2COL)
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w / 2, qbers,     w, label="QBER (%)",
                    color=colours_q, alpha=_ALPHA_BAR, edgecolor="white", linewidth=0.6)
    bars2 = ax2.bar(x + w / 2, key_rates, w, label="Key-gen rate (%)",
                    color=colours_r, alpha=_ALPHA_BAR, edgecolor="white", linewidth=0.6,
                    hatch="//")

    ax1.axhline(style.THRESH_WARN,  ls="--", lw=1.0, color=style.C["warning"], alpha=0.8)
    ax1.axhline(style.THRESH_ABORT, ls="--", lw=1.0, color=style.C["abort"],   alpha=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=style.ANNOT_FS, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=style.ANNOT_FS, color="#555555")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("QBER (%)")
    ax2.set_ylabel("Key-generation Rate (%)")
    ax1.set_title("Noise Model Comparison \u2014 QBER vs Key-Generation Rate")

    ax1.legend(handles=[bars1, bars2,
               plt.Line2D([0], [0], ls="--", color=style.C["warning"]),
               plt.Line2D([0], [0], ls="--", color=style.C["abort"])],
               labels=["QBER (%)", "Key-gen rate (%)",
                       f"Warning ({style.THRESH_WARN:.0f}%)",
                       f"Abort ({style.THRESH_ABORT:.0f}%)"],
               loc="upper left")

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    style.box_ticks(ax1)

    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E2  QBER vs depolarising probability
# ──────────────────────────────────────────────────────────────────────

def plot_depolar_sweep(
    p_values:     Sequence[float],
    qbers:        Sequence[float],          # percent
    ci_low:       Optional[Sequence[float]] = None,
    ci_high:      Optional[Sequence[float]] = None,
    theory_line:  bool = True,
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of QBER (%) vs depolarising probability p.

    Parameters
    ----------
    p_values    : depolarising probability values (0-1).
    qbers       : measured QBER in percent.
    ci_low/high : 95% Wilson CI bounds in percent (optional shading).
    theory_line : overlay the theoretical QBER ~ p x 2/3 x 100.
    save_path   : optional path to save.
    """
    p_arr = np.array(p_values)
    q_arr = np.array(qbers)

    fig, ax = plt.subplots(figsize=style.FIG_1COL)
    style.threshold_bands(ax, y_max=max(max(qbers) * 1.2, 35.0), labels=True)

    ax.plot(p_arr * 100, q_arr, "o-", color=style.C["depolar"],
            zorder=3, label="Measured QBER")

    if ci_low is not None and ci_high is not None:
        ax.fill_between(p_arr * 100, ci_low, ci_high,
                        alpha=0.18, color=style.C["depolar"], label="95% Wilson CI")

    if theory_line:
        q_theory = p_arr * (2 / 3) * 100
        ax.plot(p_arr * 100, q_theory, "--", color=style.C["theory"], alpha=0.9,
                label="Theory: QBER \u2248 p \u00d7 2/3")

    ax.set_xlabel("Depolarising Probability  p (%)")
    ax.set_ylabel("QBER (%)")
    ax.set_title("QBER vs Depolarising Probability")
    ax.legend(loc="best")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    style.box_ticks(ax)
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E3  QBER vs T1 (amplitude damping sweep)
# ──────────────────────────────────────────────────────────────────────

def plot_t1_sweep(
    t1_values_us: Sequence[float],
    qbers:        Sequence[float],          # percent
    ci_low:       Optional[Sequence[float]] = None,
    ci_high:      Optional[Sequence[float]] = None,
    gate_time_ns: float = 50.0,
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of QBER (%) vs T1 relaxation time (us).

    Also overlays the theoretical gamma = 1 - exp(-t_gate / T1) curve
    scaled to show the expected per-gate error contribution.
    """
    t1_arr = np.array(t1_values_us)
    q_arr  = np.array(qbers)

    fig, ax = plt.subplots(figsize=style.FIG_1COL)
    style.threshold_bands(ax, y_max=max(max(qbers) * 1.4, 15.0), labels=True)

    ax.plot(t1_arr, q_arr, "o-", color=style.C["amp_damp"],
            zorder=3, label="Measured QBER")

    if ci_low is not None and ci_high is not None:
        ax.fill_between(t1_arr, ci_low, ci_high,
                        alpha=0.18, color=style.C["amp_damp"], label="95% Wilson CI")

    # Overlay: expected QBER ~ gamma/2 x 100 (rough; amplitude damp is asymmetric)
    t1_fine = np.linspace(t1_arr.min(), t1_arr.max(), 300)
    gamma   = 1 - np.exp(-gate_time_ns / (t1_fine * 1000))  # t1 in ns
    q_exp   = gamma / 2 * 100
    ax.plot(t1_fine, q_exp, "--", color=style.C["theory"], alpha=0.9,
            label=r"Theory: QBER $\approx \gamma/2$  ($\gamma=1-e^{-t/T_1}$)")

    ax.set_xlabel("T1 Relaxation Time (\u00b5s)")
    ax.set_ylabel("QBER (%)")
    ax.set_title("QBER vs T1 Relaxation Time\n(Amplitude Damping)")
    ax.legend(loc="best")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    style.box_ticks(ax)
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E4  QBER vs T2 (phase damping sweep)
# ──────────────────────────────────────────────────────────────────────

def plot_t2_sweep(
    t2_values_us: Sequence[float],
    qbers:        Sequence[float],          # percent
    ci_low:       Optional[Sequence[float]] = None,
    ci_high:      Optional[Sequence[float]] = None,
    gate_time_ns: float = 50.0,
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of QBER (%) vs T2 dephasing time (us).
    """
    t2_arr = np.array(t2_values_us)
    q_arr  = np.array(qbers)

    fig, ax = plt.subplots(figsize=style.FIG_1COL)
    style.threshold_bands(ax, y_max=max(max(qbers) * 1.4, 15.0), labels=True)

    ax.plot(t2_arr, q_arr, "o-", color=style.C["phase_damp"],
            zorder=3, label="Measured QBER")

    if ci_low is not None and ci_high is not None:
        ax.fill_between(t2_arr, ci_low, ci_high,
                        alpha=0.18, color=style.C["phase_damp"], label="95% Wilson CI")

    # Overlay: expected QBER ~ lambda/4 x 100 (half of bits in diagonal basis)
    t2_fine = np.linspace(t2_arr.min(), t2_arr.max(), 300)
    lam     = 1 - np.exp(-gate_time_ns / (t2_fine * 1000))
    q_exp   = lam / 4 * 100
    ax.plot(t2_fine, q_exp, "--", color=style.C["theory"], alpha=0.9,
            label=r"Theory: QBER $\approx \lambda/4$  ($\lambda=1-e^{-t/T_2}$)")

    ax.set_xlabel("T2 Dephasing Time (\u00b5s)")
    ax.set_ylabel("QBER (%)")
    ax.set_title("QBER vs T2 Dephasing Time\n(Phase Damping)")
    ax.legend(loc="best")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    style.box_ticks(ax)
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E5  Fibre-loss dual-axis figure
# ──────────────────────────────────────────────────────────────────────

def plot_fibre_loss(
    distances:  Sequence[float],   # km
    qbers:      Sequence[float],   # percent
    key_rates:  Sequence[float],   # percent (key bits / n_transmitted x 100)
    survival:   Optional[Sequence[float]] = None,   # theoretical P_survive x 50
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Dual-axis plot: QBER (%) on the left axis, key-generation rate (%)
    on the right axis, both as functions of fibre-channel length (km).

    The theoretically predicted key rate R = P_survive x 50%
    (Beer-Lambert x 50% sifting) is overlaid as a dashed curve if
    ``survival`` is supplied.

    Parameters
    ----------
    distances  : channel lengths in km.
    qbers      : measured QBER in percent.
    key_rates  : key-generation rates (key bits / qubit x 100) in percent.
    survival   : optional theoretical R = P_survive x 50% curve values.
    save_path  : optional path to save.
    """
    d  = np.array(distances)
    q  = np.array(qbers)
    kr = np.array(key_rates)

    fig, ax1 = plt.subplots(figsize=style.FIG_1COL)
    ax2 = ax1.twinx()

    ax1.plot(d, q, "o-", color=style.C["qber"], zorder=4,
             label="QBER (%) [left axis]")
    ax1.set_ylim(-1, max(max(q) * 2, 15))
    ax1.set_ylabel("QBER (%)", color=style.C["qber"])
    ax1.tick_params(axis="y", labelcolor=style.C["qber"])
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    ax2.plot(d, kr, "s-", color=style.C["rate"], zorder=4,
             label="Key-gen rate (%) [right axis]")
    if survival is not None:
        ax2.plot(d, np.array(survival), "--", color=style.C["theory"], alpha=0.9,
                 label=r"Theory: $P_{\mathrm{survive}}\times$ 50% [right]")
    ax2.set_ylabel("Key-generation Rate (%)", color=style.C["rate"])
    ax2.tick_params(axis="y", labelcolor=style.C["rate"])
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    ax1.set_xlabel("Fibre Channel Length (km)")
    ax1.set_title("Fibre Loss \u2014 QBER and Key-Generation Rate vs Distance")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.axvline(15, ls=":", lw=1, color=style.C["theory"], alpha=0.6)
    ax1.text(16, max(q) * 0.8 + 0.2, "\u20133 dB\n(15 km)", fontsize=style.ANNOT_FS,
             color=style.C["theory"], va="top")

    style.box_ticks(ax1)
    plt.tight_layout()
    return style.save(fig, save_path)


# ──────────────────────────────────────────────────────────────────────
# E6  Depolarising noise x Eve intercept-probability heatmap
# ──────────────────────────────────────────────────────────────────────

def plot_depolar_vs_eve_heatmap(
    qber_matrix:    np.ndarray,
    depolar_values: List[float],
    eve_values:     List[float],
    save_path:      Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of QBER (%) as a function of depolarising noise probability
    and Eve's intercept probability, with 5%/11%/25% QBER contour lines.
    """
    from scipy.interpolate import RegularGridInterpolator

    fig, ax = plt.subplots(figsize=style.FIG_1COL)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.86, bottom=0.14)
    fig.suptitle("QBER (%) as a Function of Channel Noise\nand Eve Intercept Probability")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "qkd_heat",
        ["#f7fbff", "#fdd49e", "#fdae61", "#d7301f", "#7f0000"],
        N=512,
    )

    eve_pct = [e * 100 for e in eve_values]
    dp_pct  = [d * 100 for d in depolar_values]

    data_max = float(np.nanmax(qber_matrix))
    vmax     = max(30.0, np.ceil(data_max / 5.0) * 5.0)

    im = ax.imshow(
        qber_matrix,
        origin="lower",
        aspect="auto",
        extent=[eve_pct[0], eve_pct[-1], dp_pct[0], dp_pct[-1]],
        cmap=cmap,
        vmin=0, vmax=vmax,
        interpolation="bilinear",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("QBER (%)")
    cbar.set_ticks(np.arange(0, vmax + 1, 5))

    eve_fine = np.linspace(eve_pct[0], eve_pct[-1], 300)
    dp_fine  = np.linspace(dp_pct[0],  dp_pct[-1],  300)
    interp   = RegularGridInterpolator(
        (dp_pct, eve_pct), qber_matrix,
        method="linear", bounds_error=False, fill_value=None,
    )
    EE, DD = np.meshgrid(eve_fine, dp_fine)
    ZZ     = interp((DD, EE))

    cs5  = ax.contour(EE, DD, ZZ, levels=[style.THRESH_WARN],  colors=[style.C["warning"]],
                      linewidths=1.1, linestyles="--")
    cs11 = ax.contour(EE, DD, ZZ, levels=[style.THRESH_ABORT], colors=[style.C["abort"]],
                      linewidths=1.1, linestyles="--")
    cs25 = ax.contour(EE, DD, ZZ, levels=[25], colors=[style.C["theory"]],
                      linewidths=0.8, linestyles=":")
    ax.clabel(cs5,  fmt=" 5%%",  fontsize=style.ANNOT_FS, inline=True)
    ax.clabel(cs11, fmt="11%%",  fontsize=style.ANNOT_FS, inline=True)
    ax.clabel(cs25, fmt="25%%",  fontsize=style.ANNOT_FS, inline=True)

    ax.set_xlabel(r"Eve Intercept Probability  $p_{\mathrm{Eve}}$ (%)")
    ax.set_ylabel(r"Depolarising Noise  $p_{\mathrm{dep}}$ (%)")
    ax.set_xticks(eve_pct)
    ax.set_yticks(dp_pct)
    style.box_ticks(ax)

    return style.save(fig, save_path)