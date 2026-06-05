"""bb84_plots_extended.py
======================
Extended visualisation for BB84 QKD research report.
IEEE / research-paper quality — all plots 300 dpi.

Exports
-------
collect_sample_size_data()           Run simulations for QBER & CI width vs n
collect_intercept_sweep_data()       Run simulations sweeping Eve 0→100%
collect_sample_fraction_data()       Run simulations for QBER vs sample_frac
collect_depolar_eve_data()           Run simulations for depolar × Eve heatmap
collect_detection_power_data()       Run trials to measure Eve detection rate vs n

plot_sample_size_sensitivity()       QBER & CI width vs n_qubits
plot_qber_vs_intercept_enhanced()    Backward-compat wrapper → calls panels version
plot_qber_vs_intercept_panels()      *** REPLACES Plot B ***
                                     Saves one PNG per n value (e.g. _n40, _n100 …)
                                     y 0–100 %, security zone shading, MAD annotation
plot_qber_vs_intercept_overview()    Compact single-panel overlay for n≥400
plot_sample_fraction_effect()        QBER vs sample_frac at 4 intercept rates
plot_depolar_vs_eve_heatmap()        QBER heatmap: depolar_prob × eve_intercept
plot_depolar_vs_eve_lines()          Line plot companion to heatmap
plot_detection_power()               Empirical Eve-detection probability vs n
plot_ci_width_vs_n()                 95% CI width vs qubit count (log-log)
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.stats import sem as scipy_sem

from bb84.config import SimulationConfig, SimulationResult
from bb84.runner import run_simulation


# ──────────────────────────────────────────────────────────────────────────────
# IEEE / RESEARCH-PAPER GLOBAL STYLE
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.dpi":         150,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.bottom": True,
    "axes.spines.left":   True,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "grid.linewidth":     0.45,
    "grid.alpha":         0.35,
    "grid.color":         "#aaaaaa",
    "lines.linewidth":    1.2,
    "lines.markersize":   4.5,
})

# IEEE-palette: colorblind-safe, publication standard
_C = {
    "blue":    "#1f77b4",
    "orange":  "#d62728",
    "green":   "#2ca02c",
    "red":     "#d62728",
    "purple":  "#9467bd",
    "brown":   "#8c564b",
    "pink":    "#e377c2",
    "gray":    "#7f7f7f",
    "olive":   "#bcbd22",
    "teal":    "#17becf",
    "secure":  "#2ca02c",
    "warning": "#ff7f0e",
    "abort":   "#d62728",
    "theory":  "#555555",
}

_PANEL_COLORS = {
    40:  "#1f77b4",
    100: "#2ca02c",
    400: "#ff7f0e",
    600: "#d62728",
}
_DEFAULT_COLOR = "#9467bd"

_MARKERS   = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINESTYLE = ["-", "--", "-.", ":", (0,(3,1,1,1))]


def _status_color(r: SimulationResult) -> str:
    s = r.qber_result.security_status
    if "SECURE"  in s: return _C["secure"]
    if "WARNING" in s: return _C["warning"]
    return _C["abort"]


def _wrap(text: str, w: int = 70) -> str:
    return textwrap.fill(text, width=w, break_long_words=False)


def _apply_box_ticks(ax, major_len=4, minor_len=2):
    ax.tick_params(which="major", top=True, right=True,
                   length=major_len, width=0.8, direction="in")
    ax.tick_params(which="minor", top=True, right=True,
                   length=minor_len, width=0.5, direction="in")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))


def _save(fig, path, tight=True):
    if path:
        kw = {"dpi": 300, "bbox_inches": "tight"} if tight else {"dpi": 300}
        fig.savefig(path, **kw)
        print(f"  [✓] Saved → {path}  (300 dpi)")


# ──────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def collect_sample_size_data(
    n_values:        List[int],
    eve_intercept:   float = 1.0,
    sample_fraction: float = 0.15,
    seed:            int   = 42,
) -> List[dict]:
    """Run simulations for each n in n_values; return list of result dicts."""
    rows = []
    for n in n_values:
        cfg = SimulationConfig(
            n_qubits=n,
            eve_present=(eve_intercept > 0),
            eve_intercept_prob=eve_intercept,
            sample_fraction=sample_fraction,
            noise_enabled=False,
            seed=seed,
        )
        r = run_simulation(cfg, verbose=False)
        q = r.qber_result
        rows.append({
            "n":        n,
            "qber":     q.qber * 100,
            "ci_low":   q.confidence_low  * 100,
            "ci_high":  q.confidence_high * 100,
            "ci_width": (q.confidence_high - q.confidence_low) * 100,
            "status":   q.security_status,
            "sample_n": q.sample_size,
        })
        print(f"    n={n:5d}  QBER={rows[-1]['qber']:.1f}%  "
              f"CI=[{rows[-1]['ci_low']:.1f},{rows[-1]['ci_high']:.1f}]  "
              f"status={rows[-1]['status']}")
    return rows


def collect_intercept_sweep_data(
    n_qubits:        int   = 600,
    steps:           int   = 10,
    sample_fraction: float = 0.15,
    seed:            int   = 42,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Return (probs_pct, qbers, ci_low, ci_high) sweeping Eve 0→100%."""
    probs, qbers, ci_low, ci_hi = [], [], [], []
    for p in np.linspace(0, 1, steps + 1):
        cfg = SimulationConfig(
            n_qubits=n_qubits,
            eve_present=(p > 0),
            eve_intercept_prob=float(p),
            sample_fraction=sample_fraction,
            noise_enabled=False,
            seed=seed,
        )
        r = run_simulation(cfg, verbose=False)
        q = r.qber_result
        probs.append(float(p) * 100)
        qbers.append(q.qber * 100)
        ci_low.append(max(0.0, q.confidence_low  * 100))
        ci_hi.append(q.confidence_high * 100)
        print(f"    p={p:.2f}  QBER={qbers[-1]:.1f}%")
    return probs, qbers, ci_low, ci_hi


def collect_sample_fraction_data(
    intercept_rates: List[float],
    frac_values:     List[float],
    n_qubits:        int = 600,
    seed:            int = 42,
) -> Dict:
    """
    Returns nested dict: data[intercept_rate][frac] = {qber, ci_low, ci_high}
    """
    data = {}
    for p in intercept_rates:
        data[p] = {}
        for frac in frac_values:
            cfg = SimulationConfig(
                n_qubits=n_qubits,
                eve_present=(p > 0),
                eve_intercept_prob=float(p),
                sample_fraction=frac,
                noise_enabled=False,
                seed=seed,
            )
            r = run_simulation(cfg, verbose=False)
            q = r.qber_result
            data[p][frac] = {
                "qber":    q.qber * 100,
                "ci_low":  max(0.0, q.confidence_low  * 100),
                "ci_high": q.confidence_high * 100,
                "status":  q.security_status,
            }
            print(f"    p={p:.2f}  frac={frac:.2f}  QBER={data[p][frac]['qber']:.1f}%")
    return data


def collect_depolar_eve_data(
    depolar_values:  List[float],
    eve_values:      List[float],
    n_qubits:        int   = 600,
    sample_fraction: float = 0.15,
    seed:            int   = 42,
) -> np.ndarray:
    """
    Returns 2-D array qber[i, j] for depolar_values[i], eve_values[j].
    """
    arr = np.zeros((len(depolar_values), len(eve_values)))
    for i, dp in enumerate(depolar_values):
        for j, ep in enumerate(eve_values):
            cfg = SimulationConfig(
                n_qubits=n_qubits,
                eve_present=(ep > 0),
                eve_intercept_prob=float(ep),
                noise_enabled=(dp > 0),
                depolar_prob=float(dp),
                sample_fraction=sample_fraction,
                seed=seed,
            )
            r = run_simulation(cfg, verbose=False)
            arr[i, j] = r.qber_result.qber * 100
            print(f"    dp={dp:.3f}  ep={ep:.2f}  QBER={arr[i,j]:.1f}%")
    return arr


def collect_detection_power_data(
    n_values:      List[int],
    eve_intercept: float = 1.0,
    n_trials:      int   = 20,
    sample_frac:   float = 0.15,
) -> List[float]:
    """
    For each n in n_values, run n_trials simulations with different random
    seeds and record what fraction trigger an ABORT decision (QBER ≥ 11 %).

    Parameters
    ----------
    n_values      : qubit counts to sweep
    eve_intercept : Eve's intercept probability (1.0 = full intercept)
    n_trials      : number of independent seeds per n value
    sample_frac   : fraction of sifted key used for QBER estimation

    Returns
    -------
    List[float]
        Detection probability in percent for each entry in n_values.
        Pass directly to plot_detection_power().
    """
    print(f"\n  [collect_detection_power_data] "
          f"{n_trials} seeds × {len(n_values)} n-values …")
    detect_probs: List[float] = []

    for n in n_values:
        aborts = 0
        for seed in range(n_trials):
            cfg = SimulationConfig(
                n_qubits=n,
                eve_present=(eve_intercept > 0),
                eve_intercept_prob=float(eve_intercept),
                sample_fraction=sample_frac,
                noise_enabled=False,
                seed=seed,
            )
            r = run_simulation(cfg, verbose=False)
            status = r.qber_result.security_status
            if "ABORT" in status or "x" in status:
                aborts += 1

        prob = aborts / n_trials
        detect_probs.append(prob * 100)
        print(f"    n={n:5d}  detection={prob*100:.1f}%  "
              f"({aborts}/{n_trials} trials)")

    return detect_probs


# ──────────────────────────────────────────────────────────────────────────────
# PLOT A — SAMPLE SIZE SENSITIVITY
# ──────────────────────────────────────────────────────────────────────────────

def plot_sample_size_sensitivity(
    data_ideal:  List[dict],
    data_eve100: List[dict],
    n_values:    List[int],
    save_path:   Optional[str] = "fig_A_sample_size.png",
) -> None:
    """
    Single-panel: simulated QBER vs n, ideal + Eve-100%.
    Theoretical references as neutral dashed lines. y fixed 0-100%.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.4), constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.14)
    fig.suptitle(
        "Effect of Qubit Sample Size on QBER Estimation",
        fontsize=9, fontweight="bold", y=0.98,
    )

    qbers_ideal = [d["qber"] for d in data_ideal]
    qbers_eve   = [d["qber"] for d in data_eve100]

    ax.axhline(25.0, color="#888888", linestyle=(0,(5,3)), linewidth=0.9, zorder=1,
               label="Theory: Eve 100% (QBER = 25%)")
    ax.axhline(0.0,  color="#aaaaaa", linestyle=(0,(5,3)), linewidth=0.9, zorder=1,
               label="Theory: Ideal (QBER = 0%)")
    ax.axhline(11, color="#d62728", linestyle=":", linewidth=0.85, zorder=2,
               label="Abort threshold (11%)")
    ax.axhline(5,  color="#ff7f0e", linestyle=":", linewidth=0.85, zorder=2,
               label="Warning threshold (5%)")

    ax.plot(n_values, qbers_ideal,
            color="#2ca02c", marker="o", linestyle="-",
            markersize=5, linewidth=1.3, label="Ideal (no Eve)", zorder=4)
    ax.plot(n_values, qbers_eve,
            color="#1f77b4", marker="s", linestyle="-",
            markersize=5, linewidth=1.3, label="Eve 100% (simulated)", zorder=4)

    n_last = n_values[-1]
    q_last = qbers_eve[-1]
    ax.annotate(
        "Converges to\ntheory at large $n$",
        xy=(n_last, q_last),
        xytext=(n_values[-3], min(q_last + 6, 30)),
        fontsize=6, color="#333333", ha="center",
        arrowprops=dict(arrowstyle="->", color="#444444", lw=0.7),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Transmitted Qubits  $n$", fontsize=8)
    ax.set_ylabel("Estimated QBER (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_xlim(min(n_values) * 0.8, max(n_values) * 1.3)
    ax.set_xticks(n_values)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))

    leg = ax.legend(
        fontsize=6.2, loc="upper right",
        framealpha=0.93, edgecolor="#aaaaaa",
        ncol=1, handlelength=2.2, labelspacing=0.35,
        borderpad=0.5, handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(0.5)
    ax.grid(axis="y", zorder=0, which="major", alpha=0.4)
    _apply_box_ticks(ax)
    _save(fig, save_path)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOT A2 — CI WIDTH vs N
# ──────────────────────────────────────────────────────────────────────────────

def plot_ci_width_vs_n(
    data_ideal:      List[dict],
    data_eve100:     List[dict],
    n_values:        List[int],
    sample_fraction: float       = 0.15,
    save_path:       Optional[str] = "fig_A2_ci_width.png",
) -> None:
    """
    Log-log plot of 95% Wilson CI width vs qubit count n.
    Theoretical 1/sqrt(n) decay overlaid for reference.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.4), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.14)
    fig.suptitle(
        "95% Confidence Interval Width vs. Qubit Count",
        fontsize=9, fontweight="bold", y=0.98,
    )

    ci_w_ideal = [d["ci_width"] for d in data_ideal]
    ci_w_eve   = [d["ci_width"] for d in data_eve100]

    n_arr    = np.array(n_values, dtype=float)
    theory_w = 2 * 1.96 * np.sqrt(0.25 * 0.75 / (n_arr * sample_fraction)) * 100

    ax.plot(n_values, theory_w,
            color="#888888", linestyle=(0,(5,3)), linewidth=0.9,
            label=r"Theory: $2z_{0.975}\sqrt{p(1-p)/n_{s}}$, $p=0.25$",
            zorder=2)
    ax.plot(n_values, ci_w_ideal,
            color="#2ca02c", marker="o", linestyle="-",
            markersize=5, linewidth=1.3,
            label="Ideal (no Eve)", zorder=4)
    ax.plot(n_values, ci_w_eve,
            color="#1f77b4", marker="s", linestyle="-",
            markersize=5, linewidth=1.3,
            label="Eve 100% (simulated)", zorder=4)

    ax.annotate(
        "Slope $\\propto 1/\\sqrt{n}$",
        xy=(n_values[2], theory_w[2]),
        xytext=(n_values[2] * 1.6, theory_w[2] * 2.2),
        fontsize=6.5, color="#555555", ha="left",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.7),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Transmitted Qubits  $n$", fontsize=8)
    ax.set_ylabel("95% CI Width (%)", fontsize=8)
    ax.set_xlim(min(n_values) * 0.8, max(n_values) * 1.3)

    all_w = ci_w_ideal + ci_w_eve + list(theory_w)
    valid  = [w for w in all_w if w > 0]
    ax.set_ylim(min(valid) * 0.5, max(valid) * 2.5)

    ax.set_xticks(n_values)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    leg = ax.legend(
        fontsize=6.5, loc="upper right",
        framealpha=0.93, edgecolor="#aaaaaa",
        ncol=1, handlelength=2.2, labelspacing=0.4,
        borderpad=0.5,
    )
    leg.get_frame().set_linewidth(0.5)
    ax.grid(which="both", zorder=0, alpha=0.35)
    _apply_box_ticks(ax)
    _save(fig, save_path)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOT B — QBER vs EVE INTERCEPT  (2 × 2 per-n panels)   *** MAIN NEW VERSION
# ──────────────────────────────────────────────────────────────────────────────

def plot_qber_vs_intercept_panels(
    sweep_data:    Dict[int, Tuple],  # {n: (probs_pct, qbers, ci_low, ci_hi)}
    n_list:        List[int],
    save_path:     Optional[str] = "fig_B_qber_vs_intercept.png",
    save_overview: Optional[str] = None,   # optional compact overlay for n≥400
) -> None:
    """
    Saves one standalone PNG per qubit count n.

    The ``save_path`` value is used as the base filename; the n value is
    injected before the extension, e.g.:

        save_path = 'fig_B_qber_vs_intercept.png'
        → fig_B_qber_vs_intercept_n40.png
        → fig_B_qber_vs_intercept_n100.png
        → fig_B_qber_vs_intercept_n400.png
        → fig_B_qber_vs_intercept_n600.png

    Each figure shows
      • Secure / Warning / Abort background zone shading
      • Theory line  Q = 0.25 p  (grey dashed)
      • Simulated QBER with shaded 95 % CI ribbon
      • Abort (11 %) and Warning (5 %) threshold dotted lines
      • MAD annotation: mean absolute deviation from theory
    y-axis 0–100 % so no spike is clipped (especially n=40).
    """
    import os

    # Derive per-n filename from save_path
    def _make_path(base: Optional[str], n: int) -> Optional[str]:
        if base is None:
            return None
        root, ext = os.path.splitext(base)
        return f"{root}_n{n}{ext}"

    for idx, n in enumerate(n_list):
        col = _PANEL_COLORS.get(n, _DEFAULT_COLOR)
        mkr = _MARKERS[idx % len(_MARKERS)]

        probs, qbers, ci_low, ci_hi = sweep_data[n]
        probs  = np.asarray(probs,  dtype=float)
        qbers  = np.asarray(qbers,  dtype=float)
        ci_low = np.asarray(ci_low, dtype=float)
        ci_hi  = np.asarray(ci_hi,  dtype=float)

        fig, ax = plt.subplots(figsize=(3.8, 3.0), constrained_layout=False)
        fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.14)
        fig.suptitle(
            f"QBER vs. Eve Intercept Probability  ($n = {n}$ qubits)",
            fontsize=9, fontweight="bold", y=0.99,
        )

        # ── background security zone shading ─────────────────────────────────
        ax.axhspan(0,   5,   color="#2ca02c", alpha=0.07, zorder=0)
        ax.axhspan(5,   11,  color="#ff7f0e", alpha=0.08, zorder=0)
        ax.axhspan(11, 100,  color="#d62728", alpha=0.07, zorder=0)

        # ── zone labels ───────────────────────────────────────────────────────
        ax.text(1.0,  2.0, "Secure",  fontsize=5.5,
                color=_C["secure"],  style="italic")
        ax.text(1.0,  7.0, "Warning", fontsize=5.5,
                color=_C["warning"], style="italic")
        ax.text(1.0, 55.0, "Abort",   fontsize=5.5,
                color=_C["abort"],   style="italic")

        # ── theory line ───────────────────────────────────────────────────────
        theory = 0.25 * probs
        ax.plot(probs, theory,
                color=_C["theory"], linestyle="--", linewidth=0.9,
                zorder=2, label=r"Theory $Q=0.25\,p$")

        # ── 95 % CI ribbon ────────────────────────────────────────────────────
        ax.fill_between(
            probs,
            np.clip(ci_low, 0, 100),
            np.clip(ci_hi,  0, 100),
            color=col, alpha=0.20, zorder=2, label="95 % CI",
        )

        # ── simulated QBER line ───────────────────────────────────────────────
        ax.plot(probs, qbers,
                color=col, marker=mkr,
                linestyle="-", markersize=4, linewidth=1.2,
                zorder=3, label=f"Simulated QBER")

        # ── threshold lines ───────────────────────────────────────────────────
        ax.axhline(11, color=_C["abort"],   linestyle=":",
                   linewidth=0.85, zorder=2, label="Abort (11 %)")
        ax.axhline(5,  color=_C["warning"], linestyle=":",
                   linewidth=0.85, zorder=2, label="Warning (5 %)")

        # ── MAD annotation ────────────────────────────────────────────────────
        valid = probs > 0
        if valid.any():
            mad = float(np.mean(np.abs(qbers[valid] - theory[valid])))
            ax.text(
                0.97, 0.05,
                f"MAD = {mad:.1f} pp",
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color="#444444",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#cccccc", lw=0.5, alpha=0.88),
            )

        # ── axes + legend ─────────────────────────────────────────────────────
        ax.set_xlabel(
            "Eve Intercept Probability  $p_{\\mathrm{Eve}}$ (%)", fontsize=7.5)
        ax.set_ylabel("Estimated QBER (%)", fontsize=7.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(alpha=0.28, zorder=0)
        _apply_box_ticks(ax)

        leg = ax.legend(
            fontsize=6.2, loc="upper left",
            framealpha=0.90, edgecolor="#aaaaaa",
            handlelength=2.0, labelspacing=0.32,
            borderpad=0.45, handletextpad=0.4,
        )
        leg.get_frame().set_linewidth(0.4)

        out = _make_path(save_path, n)
        _save(fig, out)
        plt.show()
        plt.close(fig)

    if save_overview:
        plot_qber_vs_intercept_overview(sweep_data, n_list, save_overview)


def plot_qber_vs_intercept_overview(
    sweep_data: Dict[int, Tuple],
    n_list:     List[int],
    save_path:  Optional[str] = "fig_B_qber_overview.png",
) -> None:
    """
    Compact single-panel overlay — only n≥400 drawn, y capped at 35 %.
    Clean enough to include directly in a paper body as a summary figure.
    Small-n variance is deliberately excluded so the theory trend is legible.
    """
    large_n = [n for n in n_list if n >= 400] or n_list

    fig, ax = plt.subplots(figsize=(3.8, 3.0), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.14)
    fig.suptitle(
        "QBER vs. Eve Intercept — Large $n$ Summary",
        fontsize=9, fontweight="bold", y=0.99,
    )

    probs0 = np.asarray(list(sweep_data.values())[0][0])
    ax.plot(probs0, 0.25 * probs0,
            color=_C["theory"], linestyle="--", linewidth=0.9,
            label=r"Theory $Q=0.25\,p$", zorder=2)

    for idx, n in enumerate(large_n):
        col = _PANEL_COLORS.get(n, _DEFAULT_COLOR)
        mkr = _MARKERS[idx % len(_MARKERS)]
        probs, qbers, ci_low, ci_hi = sweep_data[n]
        probs  = np.asarray(probs)
        qbers  = np.asarray(qbers)
        ci_low = np.asarray(ci_low)
        ci_hi  = np.asarray(ci_hi)

        ax.fill_between(
            probs,
            np.clip(ci_low, 0, 35),
            np.clip(ci_hi,  0, 35),
            color=col, alpha=0.12, zorder=2,
        )
        ax.plot(probs, np.clip(qbers, 0, 35),
                color=col, marker=mkr, linestyle="-",
                markersize=4, linewidth=1.2,
                label=f"$n={n}$", zorder=3)

    ax.axhline(11, color=_C["abort"],   linestyle=":", linewidth=0.85,
               label="Abort 11 %", zorder=2)
    ax.axhline(5,  color=_C["warning"], linestyle=":", linewidth=0.85,
               label="Warning 5 %", zorder=2)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 35)
    ax.set_xlabel("Eve Intercept Probability  $p_{\\mathrm{Eve}}$ (%)", fontsize=7.5)
    ax.set_ylabel("Estimated QBER (%)", fontsize=7.5)
    leg = ax.legend(fontsize=6, loc="upper left",
                    framealpha=0.90, edgecolor="#aaaaaa")
    leg.get_frame().set_linewidth(0.4)
    ax.grid(alpha=0.30, zorder=0)
    _apply_box_ticks(ax)
    _save(fig, save_path)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# DEPRECATED WRAPPER — keeps old call-sites working
# ──────────────────────────────────────────────────────────────────────────────

def plot_qber_vs_intercept_enhanced(
    sweep_data: Dict[int, Tuple],
    n_list:     List[int],
    save_path:  Optional[str] = "fig_B_qber_vs_intercept.png",
) -> None:
    """
    Backward-compatible wrapper — existing runner scripts need no edits.
    Delegates to plot_qber_vs_intercept_panels(), which saves one PNG per n:
        fig_B_qber_vs_intercept_n40.png
        fig_B_qber_vs_intercept_n100.png
        fig_B_qber_vs_intercept_n400.png
        fig_B_qber_vs_intercept_n600.png
    """
    plot_qber_vs_intercept_panels(
        sweep_data=sweep_data,
        n_list=n_list,
        save_path=save_path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PLOT C — SAMPLE FRACTION EFFECT
# ──────────────────────────────────────────────────────────────────────────────

def plot_sample_fraction_effect(
    frac_data:       Dict,
    intercept_rates: List[float],
    frac_values:     List[float],
    save_path:       Optional[str] = "fig_C_sample_fraction.png",
) -> None:
    """
    Two-panel figure:
      Left  — QBER vs sample_fraction for each intercept rate
      Right — 95% CI width vs sample_fraction
    """
    fig = plt.figure(figsize=(7.0, 3.0), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.17,
                        wspace=0.32)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    fig.suptitle(
        "Effect of QBER Sample Fraction on Estimation Accuracy",
        fontsize=9, fontweight="bold", y=0.98,
    )

    colors = [_C["secure"], _C["warning"], "#1f77b4", _C["abort"]]
    labels = {0.0: "Ideal ($p=0$)", 0.3: "Eve 30 %",
              0.5: "Eve 50 %",      1.0: "Eve 100 %"}

    for idx, p in enumerate(intercept_rates):
        col   = colors[idx]
        qbers = [frac_data[p][f]["qber"]    for f in frac_values]
        ci_lo = [max(0.0, frac_data[p][f]["qber"] - frac_data[p][f]["ci_low"])  for f in frac_values]
        ci_hi = [max(0.0, frac_data[p][f]["ci_high"] - frac_data[p][f]["qber"]) for f in frac_values]
        ci_w  = [frac_data[p][f]["ci_high"] - frac_data[p][f]["ci_low"]         for f in frac_values]
        frac_pct = [f * 100 for f in frac_values]

        ax1.errorbar(frac_pct, qbers, yerr=[ci_lo, ci_hi],
                     color=col, marker=_MARKERS[idx],
                     linestyle="-",
                     capsize=3, capthick=0.8, elinewidth=0.8,
                     label=labels.get(p, f"Eve {p*100:.0f}%"), zorder=3)
        ax2.plot(frac_pct, ci_w,
                 color=col, marker=_MARKERS[idx],
                 linestyle="-",
                 label=labels.get(p, f"Eve {p*100:.0f}%"), zorder=3)

    for ax in (ax1, ax2):
        ax.set_xlabel("QBER Sample Fraction (%)")
        ax.grid(alpha=0.35, zorder=0)
        _apply_box_ticks(ax)
        ax.legend(fontsize=6.5, loc="best",
                  framealpha=0.90, edgecolor="#888888").get_frame().set_linewidth(0.5)

    # ── Threshold lines — small dotted, labelled, with pointer ───────────────
    ax1.axhline(11, color=_C["abort"],   linestyle=(0, (2, 2)), linewidth=0.9, zorder=2)
    ax1.annotate("Abort", xy=(25, 11), xytext=(25, 14),
                 color=_C["abort"], fontsize=5, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=_C["abort"],
                                 lw=0.7, mutation_scale=5))

    ax1.axhline(5,  color=_C["warning"], linestyle=(0, (2, 2)), linewidth=0.9, zorder=2)
    ax1.annotate("Warn", xy=(25, 5), xytext=(25, 8),
                 color=_C["warning"], fontsize=5, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=_C["warning"],
                                 lw=0.7, mutation_scale=5))

    ax1.set_ylabel("Estimated QBER (%)")
    ax1.set_ylim(0, 38)
    ax1.set_title("(a) QBER vs. Sample Fraction", fontsize=8, pad=4)

    ax2.set_ylabel("95 % CI Width (%)")
    ax2.set_title("(b) CI Width vs. Sample Fraction", fontsize=8, pad=4)

    _save(fig, save_path)
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# PLOT D — DEPOLARIZATION × EVE HEATMAP
# ──────────────────────────────────────────────────────────────────────────────

def plot_depolar_vs_eve_heatmap(
    qber_matrix:    np.ndarray,
    depolar_values: List[float],
    eve_values:     List[float],
    save_path:      Optional[str] = "fig_D_depolar_heatmap.png",
) -> None:
    """
    Heatmap of QBER (%) as a function of depolarising noise probability
    and Eve's intercept probability.
    """
    from scipy.interpolate import RegularGridInterpolator

    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.14)
    fig.suptitle(
        "QBER (%) as a Function of Channel Noise\nand Eve Intercept Probability",
        fontsize=9, fontweight="bold", y=1.0, linespacing=1.35,
    )

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
    cbar.set_label("QBER (%)", fontsize=8)
    cbar.set_ticks(np.arange(0, vmax + 1, 5))
    cbar.ax.tick_params(labelsize=6.5)

    eve_fine = np.linspace(eve_pct[0], eve_pct[-1], 300)
    dp_fine  = np.linspace(dp_pct[0],  dp_pct[-1],  300)
    interp   = RegularGridInterpolator(
        (dp_pct, eve_pct), qber_matrix,
        method="linear", bounds_error=False, fill_value=None,
    )
    EE, DD = np.meshgrid(eve_fine, dp_fine)
    ZZ     = interp((DD, EE))

    cs5  = ax.contour(EE, DD, ZZ, levels=[5],  colors=["#ff7f0e"],
                      linewidths=1.1, linestyles="--")
    cs11 = ax.contour(EE, DD, ZZ, levels=[11], colors=["#d62728"],
                      linewidths=1.1, linestyles="--")
    cs25 = ax.contour(EE, DD, ZZ, levels=[25], colors=["#555555"],
                      linewidths=0.8, linestyles=":")
    ax.clabel(cs5,  fmt=" 5 %%",  fontsize=6.5, inline=True)
    ax.clabel(cs11, fmt="11 %%",  fontsize=6.5, inline=True)
    ax.clabel(cs25, fmt="25 %%",  fontsize=6.0, inline=True)

    ax.set_xlabel("Eve Intercept Probability  $p_{\\mathrm{Eve}}$ (%)", fontsize=7.5)
    ax.set_ylabel("Depolarising Noise  $p_{\\mathrm{dep}}$ (%)", fontsize=7.5)
    ax.set_xticks([e * 100 for e in eve_values])
    ax.set_yticks([d * 100 for d in depolar_values])
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(which="major", top=True, right=True, length=4,
                   width=0.8, direction="in", labelsize=6.5)
    ax.tick_params(which="minor", top=True, right=True, length=2,
                   width=0.5, direction="in")

    _save(fig, save_path)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOT E — DEPOLAR × EVE LINE PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_depolar_vs_eve_lines(
    qber_matrix:    np.ndarray,
    depolar_values: List[float],
    eve_values:     List[float],
    save_path:      Optional[str] = "fig_E_depolar_lines.png",
) -> None:
    """
    Line plot: QBER vs Eve intercept for each depolarisation level.
    Companion to the heatmap — easier to read exact values.
    """
    eve_pct = np.asarray([e * 100 for e in eve_values])
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    data_max = float(np.nanmax(qber_matrix))
    y_ceil   = max(40.0, np.ceil(data_max / 10.0) * 10.0)

    fig, ax = plt.subplots(figsize=(3.8, 3.0), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.14)
    fig.suptitle(
        "QBER vs. Eve Intercept for Different\nDepolarising Noise Levels",
        fontsize=9, fontweight="bold", y=1.0, linespacing=1.35,
    )

    ax.axhspan(0,   5,      color="#2ca02c", alpha=0.07, zorder=0)
    ax.axhspan(5,   11,     color="#ff7f0e", alpha=0.08, zorder=0)
    ax.axhspan(11,  y_ceil, color="#d62728", alpha=0.06, zorder=0)

    ax.text(1.0,  2.0, "Secure",  fontsize=5.5, color=_C["secure"],  style="italic")
    ax.text(1.0,  7.0, "Warning", fontsize=5.5, color=_C["warning"], style="italic")
    ax.text(1.0, 13.5, "Abort",   fontsize=5.5, color=_C["abort"],   style="italic")

    theory_eve = np.linspace(0, 100, 300)
    ax.plot(theory_eve, 0.25 * theory_eve,
            color=_C["theory"], linestyle=(0,(4,2)), linewidth=0.9,
            label=r"Theory (no noise)", zorder=2)

    for idx, dp in enumerate(depolar_values):
        col   = palette[idx % len(palette)]
        ydata = qber_matrix[idx, :]
        label = f"$p_{{\\mathrm{{dep}}}}={dp:.3f}$"

        ax.plot(eve_pct, ydata,
                color=col, marker=_MARKERS[idx],
                linestyle=_LINESTYLE[idx % 4],
                markersize=4.5, linewidth=1.2,
                label=label, zorder=3)

        baseline = float(ydata[0])
        if baseline > 0.3:
            ax.annotate(
                f"{baseline:.1f} %",
                xy=(eve_pct[0], baseline),
                xytext=(5, baseline + 1.5),
                fontsize=5.5, color=col, ha="left",
                arrowprops=dict(arrowstyle="-", color=col,
                                lw=0.5, relpos=(0, 0.5)),
            )

    ax.axhline(11, color=_C["abort"],   linestyle=":", linewidth=0.9,
               label="Abort (11 %)", zorder=2)
    ax.axhline(5,  color=_C["warning"], linestyle=":", linewidth=0.9,
               label="Warning (5 %)", zorder=2)

    ax.set_xlabel("Eve Intercept Probability  $p_{\\mathrm{Eve}}$ (%)", fontsize=7.5)
    ax.set_ylabel("Estimated QBER (%)", fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, y_ceil)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.legend(fontsize=6, loc="upper left",
              framealpha=0.90, edgecolor="#888888",
              ncol=1).get_frame().set_linewidth(0.5)
    ax.grid(alpha=0.30, zorder=0)
    _apply_box_ticks(ax)
    _save(fig, save_path)
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOT F — DETECTION PROBABILITY vs n
# ──────────────────────────────────────────────────────────────────────────────

def plot_detection_power(
    n_values:      List[int],
    detect_probs:  List[float],
    n_trials:      int          = 20,
    sample_frac:   float        = 0.15,
    save_path:     Optional[str] = "fig_F_detection_power.png",
) -> None:
    """
    Plot empirical Eve-detection probability vs qubit count.

    Parameters
    ----------
    n_values      : qubit counts (x-axis), same order as detect_probs
    detect_probs  : detection probability in percent for each n,
                    as returned by collect_detection_power_data()
    n_trials      : number of trials used during collection
                    (used only for the figure subtitle)
    sample_frac   : sample fraction used during collection
                    (used only for the figure subtitle)
    save_path     : output PNG path, or None to skip saving
    """
    fig, ax = plt.subplots(figsize=(3.8, 3.0), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.14)
    fig.suptitle(
        "Eve Detection Probability vs. Qubit Count\n"
        f"(Eve 100 %, sample={sample_frac*100:.0f} %, {n_trials} trials)",
        fontsize=9, fontweight="bold", y=1.0, linespacing=1.35,
    )

    ax.semilogx(n_values, detect_probs,
                color=_C["abort"], marker="o", linestyle="-",
                markersize=5, label="Empirical detection rate", zorder=3)
    ax.axhline(95, color=_C["theory"], linestyle="--", linewidth=0.9,
               label="95 % reliability target", zorder=2)

    ax.set_xlabel("Number of Transmitted Qubits  $n$")
    ax.set_ylabel("Eve Detection Probability (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(min(n_values) * 0.8, max(n_values) * 1.3)
    ax.set_xticks(n_values)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.legend(fontsize=7, loc="lower right",
              framealpha=0.90, edgecolor="#888888").get_frame().set_linewidth(0.5)
    ax.grid(alpha=0.35, zorder=0)
    _apply_box_ticks(ax)
    _save(fig, save_path)
    plt.show()