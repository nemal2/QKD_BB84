"""
bb84_plots_extended.py
======================
Extended plotting and data-collection helpers used by the Phase 1/2
research notebook (bb84_phase3_experiments.ipynb).

Public API
----------
Data collectors  (return raw dicts / matrices)
    collect_sample_size_data()       – QBER vs n_qubits
    collect_intercept_sweep_data()   – QBER vs Eve intercept probability
    collect_sample_fraction_data()   – QBER vs QBER-sample fraction
    collect_depolar_eve_data()       – QBER[depolar × eve] 2-D matrix
    collect_detection_power_data()   – Eve-detection probability vs n

Plot functions  (return matplotlib Figure; accept save_path kwarg)
    plot_sample_size_sensitivity()   – Fig A main (QBER ± CI vs n)
    plot_ci_width_vs_n()             – Fig A2 (CI width vs n)
    plot_qber_vs_intercept_enhanced()– Fig B (QBER vs p_Eve, multi-n)
    plot_sample_fraction_effect()    – Fig C (QBER vs sample fraction)
    plot_depolar_vs_eve_heatmap()    – Fig D (heat map)
    plot_depolar_vs_eve_lines()      – Fig E (line overlay)
    plot_detection_power()           – Fig F (detection probability)

University of Ruhuna – Dept. of Computer Engineering
MIT Licence – see LICENSE
"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from bb84_config import SimulationConfig
from bb84_runner import run_simulation


# ── Style constants ────────────────────────────────────────────────────
_DPI   = 300
_FIG_W = 9
_FIG_H = 5

_C = {
    "ideal":    "#009E73",
    "eve100":   "#D55E00",
    "n40":      "#CC79A7",
    "n100":     "#56B4E9",
    "n400":     "#E69F00",
    "n600":     "#0072B2",
    "warn":     "#E69F00",
    "abort":    "#D55E00",
    "ci":       "#AAAAAA",
    "theory":   "#555555",
}

_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#E69F00", "#56B4E9", "#F0E442"]

_THRESH_WARN  = 5.0
_THRESH_ABORT = 11.0

# ──────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────

def _threshold_lines(ax: plt.Axes, x_max: float = 100.0) -> None:
    ax.axhline(_THRESH_WARN,  ls="--", lw=1.0, color=_C["warn"],  alpha=0.8,
               label=f"Warning threshold ({_THRESH_WARN:.0f}%)")
    ax.axhline(_THRESH_ABORT, ls="--", lw=1.0, color=_C["abort"], alpha=0.8,
               label=f"Abort threshold ({_THRESH_ABORT:.0f}%)")


def _save(fig: plt.Figure, path: Optional[str]) -> plt.Figure:
    if path:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        print(f"  [✓] Saved → {path}  ({_DPI} dpi)")
    return fig


def _run(n_qubits: int, seed: int,
         eve_intercept: float = 0.0,
         sample_fraction: float = 0.15,
         depolar_prob: float = 0.0,
         noise_enabled: bool = False) -> dict:
    """Thin wrapper: returns a summary dict from run_simulation."""
    cfg = SimulationConfig(
        n_qubits=n_qubits,
        seed=seed,
        eve_present=(eve_intercept > 0),
        eve_intercept_prob=eve_intercept,
        noise_enabled=noise_enabled,
        depolar_prob=depolar_prob,
        sample_fraction=sample_fraction,
    )
    r  = run_simulation(cfg, verbose=False)
    qr = r.qber_result
    return {
        "n":        n_qubits,
        "qber":     qr.qber * 100,
        "ci_low":   qr.confidence_low  * 100,
        "ci_high":  qr.confidence_high * 100,
        "ci_width": (qr.confidence_high - qr.confidence_low) * 100,
        "status":   qr.security_status,
        "key_len":  r.key_length,
    }


# ──────────────────────────────────────────────────────────────────────
# DATA COLLECTORS
# ──────────────────────────────────────────────────────────────────────

def collect_sample_size_data(
    n_values:        List[int],
    eve_intercept:   float = 0.0,
    sample_fraction: float = 0.15,
    seed:            int   = 42,
) -> List[dict]:
    """Return one dict per n_value with QBER, CI, and status."""
    results = []
    for n in n_values:
        d = _run(n, seed, eve_intercept=eve_intercept,
                 sample_fraction=sample_fraction)
        results.append(d)
        status = d["status"]
        print(f"    n={n:>5}  QBER={d['qber']:.1f}%  "
              f"CI=[{d['ci_low']:.1f},{d['ci_high']:.1f}]  status={status}")
    return results


def collect_intercept_sweep_data(
    n_qubits:        int,
    steps:           int   = 10,
    sample_fraction: float = 0.15,
    seed:            int   = 42,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Sweep Eve's intercept probability from 0 to 1.

    Returns (probs, qbers, ci_low, ci_high) — four parallel lists.
    """
    probs   = np.linspace(0, 1, steps + 1).tolist()
    qbers, ci_low, ci_high = [], [], []
    for p in probs:
        d = _run(n_qubits, seed, eve_intercept=p,
                 sample_fraction=sample_fraction)
        qbers.append(d["qber"])
        ci_low.append(d["ci_low"])
        ci_high.append(d["ci_high"])
        print(f"    p={p:.2f}  QBER={d['qber']:.1f}%")
    return probs, qbers, ci_low, ci_high


def collect_sample_fraction_data(
    intercept_rates: List[float],
    frac_values:     List[float],
    n_qubits:        int = 600,
    seed:            int = 42,
) -> Dict[float, Dict[float, dict]]:
    """
    Nested dict: data[intercept_rate][fraction] = summary dict.
    """
    data: Dict[float, Dict[float, dict]] = {}
    for p in intercept_rates:
        data[p] = {}
        for frac in frac_values:
            d = _run(n_qubits, seed, eve_intercept=p,
                     sample_fraction=frac)
            data[p][frac] = d
            print(f"    p={p:.2f}  frac={frac:.2f}  QBER={d['qber']:.1f}%")
    return data


def collect_depolar_eve_data(
    depolar_values:  List[float],
    eve_values:      List[float],
    n_qubits:        int   = 600,
    sample_fraction: float = 0.15,
    seed:            int   = 42,
) -> np.ndarray:
    """
    Run grid of (depolar_prob × eve_intercept) and return a 2-D
    numpy array of QBER percentages with shape
    (len(depolar_values), len(eve_values)).
    """
    matrix = np.zeros((len(depolar_values), len(eve_values)))
    for i, dp in enumerate(depolar_values):
        for j, ep in enumerate(eve_values):
            d = _run(n_qubits, seed,
                     eve_intercept=ep,
                     noise_enabled=(dp > 0),
                     depolar_prob=dp,
                     sample_fraction=sample_fraction)
            matrix[i, j] = d["qber"]
            print(f"    dp={dp:.3f}  ep={ep:.2f}  QBER={d['qber']:.1f}%")
    return matrix


def collect_detection_power_data(
    n_values:      List[int],
    eve_intercept: float = 1.0,
    n_trials:      int   = 20,
    sample_frac:   float = 0.15,
) -> List[float]:
    """
    For each n, run n_trials simulations with different seeds and count
    how often the protocol returns ABORT (QBER ≥ 11 %).

    Returns list of detection probabilities (%) — one per n value.
    """
    detect_probs = []
    for n in n_values:
        aborts = 0
        for trial in range(n_trials):
            d = _run(n, seed=trial,
                     eve_intercept=eve_intercept,
                     sample_fraction=sample_frac)
            if d["status"] == "ABORT x":
                aborts += 1
        prob = aborts / n_trials * 100
        detect_probs.append(prob)
        verdict = "Reliable" if prob >= 95 else ("Moderate" if prob >= 60 else "Unreliable")
        print(f"    n={n:>5}  detection={prob:.1f}%  [{verdict}]")
    return detect_probs


# ──────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def plot_sample_size_sensitivity(
    data_ideal:  List[dict],
    data_eve100: List[dict],
    n_values:    List[int],
    save_path:   Optional[str] = None,
) -> plt.Figure:
    """
    Fig A: QBER ± 95 % CI vs qubit count, ideal vs Eve-100 %.
    """
    n_arr = np.array(n_values)

    q_ideal  = np.array([d["qber"]     for d in data_ideal])
    lo_ideal = np.array([d["ci_low"]   for d in data_ideal])
    hi_ideal = np.array([d["ci_high"]  for d in data_ideal])

    q_eve    = np.array([d["qber"]     for d in data_eve100])
    lo_eve   = np.array([d["ci_low"]   for d in data_eve100])
    hi_eve   = np.array([d["ci_high"]  for d in data_eve100])

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _threshold_lines(ax)

    ax.plot(n_arr, q_ideal,  "o-", color=_C["ideal"],  lw=2, ms=6,
            label="Ideal channel (no Eve)")
    ax.fill_between(n_arr, lo_ideal, hi_ideal, alpha=0.15, color=_C["ideal"])

    ax.plot(n_arr, q_eve,    "s-", color=_C["eve100"], lw=2, ms=6,
            label="Eve full intercept (100 %)")
    ax.fill_between(n_arr, lo_eve,   hi_eve,   alpha=0.15, color=_C["eve100"])

    # Theory band for Eve: E[QBER] = 25%
    ax.axhline(25, ls=":", lw=1.2, color=_C["eve100"], alpha=0.5,
               label="Theoretical QBER (Eve 100%) = 25%")

    ax.set_xlabel("Number of Qubits Transmitted  (n)", fontsize=11)
    ax.set_ylabel("Estimated QBER (%)", fontsize=11)
    ax.set_title("Fig A: Effect of Qubit Count on QBER Estimation Reliability",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    return _save(fig, save_path)


def plot_ci_width_vs_n(
    data_ideal:      List[dict],
    data_eve100:     List[dict],
    n_values:        List[int],
    sample_fraction: float = 0.15,
    save_path:       Optional[str] = None,
) -> plt.Figure:
    """
    Fig A2: 95 % CI width vs n for both conditions, with
    theoretical 2z/√n_sample overlay.
    """
    n_arr  = np.array(n_values, dtype=float)
    w_ideal  = np.array([d["ci_width"] for d in data_ideal])
    w_eve    = np.array([d["ci_width"] for d in data_eve100])

    n_sample = n_arr * 0.5 * sample_fraction   # approx sample bits
    z = 1.96
    theory   = 2 * z / np.sqrt(np.maximum(n_sample, 1)) * 100  # %

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(n_arr, w_ideal,  "o-", color=_C["ideal"],   lw=2, ms=6,
            label="CI width – Ideal")
    ax.plot(n_arr, w_eve,    "s-", color=_C["eve100"],  lw=2, ms=6,
            label="CI width – Eve 100%")
    ax.plot(n_arr, theory,   "--", color=_C["theory"],  lw=1.4, alpha=0.8,
            label=r"Theory: $2z/\sqrt{n_{sample}}$  (z=1.96)")

    ax.set_xlabel("Number of Qubits Transmitted  (n)", fontsize=11)
    ax.set_ylabel("95 % Wilson CI Width (%)", fontsize=11)
    ax.set_title("Fig A2: CI Width vs Qubit Count", fontsize=12,
                 fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    return _save(fig, save_path)


def plot_qber_vs_intercept_enhanced(
    sweep_data: Dict[int, Tuple],
    n_list:     List[int],
    save_path:  Optional[str] = None,
) -> List[plt.Figure]:
    """
    Fig B: One plot per n value (QBER vs p_Eve).
    Returns a list of figures.  If save_path is given, files are named
    ``<base>_n<value>.png``.
    """
    figures = []
    base, ext = (os.path.splitext(save_path) if save_path else ("", ".png"))

    theory_p = np.linspace(0, 1, 100)
    theory_q = theory_p * 25            # % : QBER = 0.25 × p

    palette = _PALETTE

    for idx, n in enumerate(n_list):
        probs, qbers, ci_low, ci_high = sweep_data[n]
        p_arr = np.array(probs) * 100     # → percent for axis
        q_arr = np.array(qbers)

        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
        _threshold_lines(ax)

        ax.fill_between(p_arr, ci_low, ci_high, alpha=0.15,
                        color=palette[idx % len(palette)])
        ax.plot(p_arr, q_arr, "o-",
                color=palette[idx % len(palette)], lw=2, ms=6,
                label=f"Measured QBER  (n={n})")
        ax.plot(theory_p * 100, theory_q, "--", color=_C["theory"],
                lw=1.3, alpha=0.85, label="Theory: QBER = 0.25 × p_Eve")

        ax.set_xlabel("Eve Intercept Probability  p_Eve (%)", fontsize=11)
        ax.set_ylabel("QBER (%)", fontsize=11)
        ax.set_title(f"Fig B: QBER vs Eve Intercept Probability  (n = {n})",
                     fontsize=12, fontweight="bold", pad=10)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        plt.tight_layout()

        sp = f"{base}_n{n}{ext}" if save_path else None
        figures.append(_save(fig, sp))

    return figures


def plot_sample_fraction_effect(
    frac_data:       Dict[float, Dict[float, dict]],
    intercept_rates: List[float],
    frac_values:     List[float],
    save_path:       Optional[str] = None,
) -> plt.Figure:
    """
    Fig C: QBER (%) vs sample fraction, one line per intercept rate.
    """
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _threshold_lines(ax)

    for idx, p in enumerate(intercept_rates):
        q_arr = [frac_data[p][f]["qber"] for f in frac_values]
        ax.plot(
            [f * 100 for f in frac_values], q_arr,
            "o-", color=_PALETTE[idx % len(_PALETTE)], lw=2, ms=6,
            label=f"p_Eve = {int(p * 100)}%",
        )

    ax.set_xlabel("QBER Sample Fraction (%)", fontsize=11)
    ax.set_ylabel("Estimated QBER (%)", fontsize=11)
    ax.set_title("Fig C: Effect of Sample Fraction on QBER Estimation",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    return _save(fig, save_path)


def plot_depolar_vs_eve_heatmap(
    qber_matrix:    np.ndarray,
    depolar_values: List[float],
    eve_values:     List[float],
    save_path:      Optional[str] = None,
) -> plt.Figure:
    """
    Fig D: Heat map of QBER (%) over the (depolar_prob × eve_intercept) grid.
    """
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    im = ax.imshow(qber_matrix, aspect="auto", origin="lower",
                   cmap="RdYlGn_r", vmin=0, vmax=30,
                   extent=[-0.5, len(eve_values) - 0.5,
                            -0.5, len(depolar_values) - 0.5])

    ax.set_xticks(range(len(eve_values)))
    ax.set_xticklabels([f"{int(e * 100)}%" for e in eve_values], fontsize=9)
    ax.set_yticks(range(len(depolar_values)))
    ax.set_yticklabels([f"{p:.3f}" for p in depolar_values], fontsize=9)
    ax.set_xlabel("Eve Intercept Probability", fontsize=11)
    ax.set_ylabel("Depolarising Probability  p", fontsize=11)
    ax.set_title("Fig D: QBER (%) — Depolarising Noise × Eve Intercept\n"
                 "(mean across 3 seeds, n=600)", fontsize=12,
                 fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=ax, label="QBER (%)")
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Annotate cells
    for i in range(len(depolar_values)):
        for j in range(len(eve_values)):
            ax.text(j, i, f"{qber_matrix[i, j]:.1f}",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if qber_matrix[i, j] > 15 else "black")

    plt.tight_layout()
    return _save(fig, save_path)


def plot_depolar_vs_eve_lines(
    qber_matrix:    np.ndarray,
    depolar_values: List[float],
    eve_values:     List[float],
    save_path:      Optional[str] = None,
) -> plt.Figure:
    """
    Fig E: Line plot of QBER (%) vs Eve intercept probability,
    one line per depolar_prob level.
    """
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    _threshold_lines(ax)

    p_eve_pct = [e * 100 for e in eve_values]
    for i, dp in enumerate(depolar_values):
        ax.plot(p_eve_pct, qber_matrix[i], "o-",
                color=_PALETTE[i % len(_PALETTE)], lw=2, ms=6,
                label=f"p_depol = {dp:.3f}")

    # Theory overlay (no noise)
    ax.plot(p_eve_pct, [e * 25 for e in eve_values],
            "--", color=_C["theory"], lw=1.3, alpha=0.7,
            label="Theory: QBER = 0.25 × p_Eve")

    ax.set_xlabel("Eve Intercept Probability (%)", fontsize=11)
    ax.set_ylabel("QBER (%)", fontsize=11)
    ax.set_title("Fig E: QBER vs Eve Intercept for Different Depolarising Noise Levels",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    return _save(fig, save_path)


def plot_detection_power(
    n_values:     List[int],
    detect_probs: List[float],
    n_trials:     int   = 20,
    sample_frac:  float = 0.15,
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """
    Fig F: Eve-detection probability (%) vs qubit count n.
    """
    n_arr = np.array(n_values)
    d_arr = np.array(detect_probs)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    # Reference lines
    ax.axhline(95, ls="--", lw=1.0, color=_C["ideal"],  alpha=0.9,
               label="95 % reliability threshold")
    ax.axhline(60, ls="--", lw=1.0, color=_C["warn"],   alpha=0.9,
               label="60 % moderate threshold")

    ax.plot(n_arr, d_arr, "o-", color=_PALETTE[0], lw=2, ms=7, zorder=3,
            label=f"Empirical detection rate  ({n_trials} trials/n)")

    # Shade reliability regions
    ax.axhspan(95, 102, alpha=0.07, color=_C["ideal"],  label="Reliable zone (≥95%)")
    ax.axhspan(60,  95, alpha=0.07, color=_C["warn"],   label="Moderate zone (60–95%)")
    ax.axhspan(0,   60, alpha=0.07, color=_C["abort"],  label="Unreliable zone (<60%)")

    ax.set_xlabel("Number of Qubits Transmitted  (n)", fontsize=11)
    ax.set_ylabel("Eve Detection Probability (%)", fontsize=11)
    ax.set_title(f"Fig F: Eve Detection Reliability vs Qubit Count\n"
                 f"(Eve intercept 100%, sample_fraction={sample_frac:.0%}, "
                 f"{n_trials} trials/n)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.85)
    ax.set_ylim(-2, 104)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    return _save(fig, save_path)
