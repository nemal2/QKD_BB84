"""
bb84_plots.py  ·  Comprehensive Visualisation
══════════════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

All matplotlib/seaborn plots for the BB84 research platform.
Import only this module in notebook plot cells.

Exports
───────
plot_comparison()              — multi-scenario 3-panel bar chart
plot_qber_vs_intercept_rate()  — QBER sweep vs Eve intercept prob
plot_noise_model_comparison()  — QBER vs noise level (all models)
plot_skr_vs_distance()         — secret key rate vs fiber distance
plot_skr_vs_qber()             — SKR vs QBER (theory curves)
plot_pns_analysis()            — PNS photon statistics
plot_decoy_effectiveness()     — SKR gain from decoy state
plot_cascade_performance()     — Cascade EC error reduction
plot_2d_security_heatmap()     — SKR heatmap (noise × intercept)
plot_information_budget()      — I(A;B), χ(Eve), SKR per scenario
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from bb84_config import SimulationConfig, SimulationResult
from bb84_analysis import (
    sweep_qber_vs_noise,
    sweep_skr_vs_distance,
    sweep_skr_vs_qber,
    sweep_pns_vulnerability,
    sweep_decoy_effectiveness,
    full_parameter_sweep,
)


# ══════════════════════════════════════════════════════════════════════
# STYLE SETUP
# ══════════════════════════════════════════════════════════════════════

PALETTE = {
    "secure":  "#27ae60",
    "warning": "#e67e22",
    "abort":   "#e74c3c",
    "blue":    "#2980b9",
    "purple":  "#8e44ad",
    "teal":    "#16a085",
    "dark":    "#2c3e50",
    "grey":    "#95a5a6",
}

LEGEND_PATCHES = [
    mpatches.Patch(color=PALETTE["secure"],  label="Secure  (QBER < 5 %)"),
    mpatches.Patch(color=PALETTE["warning"], label="Warning (5 – 11 %)"),
    mpatches.Patch(color=PALETTE["abort"],   label="Abort   (QBER ≥ 11 %)"),
]

TITLE_SUFFIX = "\nUniversity of Ruhuna — Dept. of Computer Engineering"


def _status_color(r: SimulationResult) -> str:
    s = r.qber_result.security_status
    if "SECURE"  in s: return PALETTE["secure"]
    if "WARNING" in s: return PALETTE["warning"]
    return PALETTE["abort"]


def _label_bars(ax, bars, values, fmt="{:.1f}%", va_offset=0.5):
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + va_offset,
                fmt.format(val),
                ha="center", va="bottom", fontsize=8, fontweight="bold")


def _save_and_show(fig, save_path):
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# PLOT 1 — SCENARIO COMPARISON  (3-panel bar chart)
# ══════════════════════════════════════════════════════════════════════

def plot_comparison(
    scenarios: List[Tuple[str, SimulationConfig]],
    results:   List[SimulationResult],
    save_path: Optional[str] = "qkd_comparison.png",
) -> None:
    """3-panel bar chart: QBER, final key length, key agreement rate."""
    labels   = [s[0] for s in scenarios]
    qbers    = [r.qber_result.qber * 100      for r in results]
    ci_lo    = [r.qber_result.confidence_low  * 100 for r in results]
    ci_hi    = [r.qber_result.confidence_high * 100 for r in results]
    key_lens = [r.key_length                  for r in results]
    agree    = [r.key_agreement_rate * 100    for r in results]
    colors   = [_status_color(r)              for r in results]

    yerr = [[q - lo for q, lo in zip(qbers, ci_lo)],
            [hi - q for q, hi in zip(qbers, ci_hi)]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("BB84 QKD Research Platform — Scenario Comparison" + TITLE_SUFFIX,
                 fontsize=11, fontweight="bold", y=1.04)

    # Panel 1: QBER
    ax = axes[0]
    bars = ax.bar(x, qbers, color=colors, alpha=0.85, edgecolor="black",
                  linewidth=0.8, zorder=3)
    ax.errorbar(x, qbers, yerr=yerr, fmt="none", color="black",
                capsize=5, linewidth=1.5, zorder=4)
    ax.axhline(11, color=PALETTE["abort"],   ls="--", lw=1.5, label="Abort (11 %)")
    ax.axhline(5,  color=PALETTE["warning"], ls="--", lw=1.5, label="Warning (5 %)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("QBER (%)"); ax.set_title("Quantum Bit Error Rate", fontweight="bold")
    ax.set_ylim(0, max(35, max(qbers) + 6))
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3, zorder=0)
    _label_bars(ax, bars, qbers)

    # Panel 2: Key length
    ax = axes[1]
    bars2 = ax.bar(x, key_lens, color=colors, alpha=0.85, edgecolor="black",
                   linewidth=0.8, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Bits"); ax.set_title("Final Key Length", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    _label_bars(ax, bars2, key_lens, fmt="{:.0f}", va_offset=1)

    # Panel 3: Agreement
    ax = axes[2]
    bars3 = ax.bar(x, agree, color=colors, alpha=0.85, edgecolor="black",
                   linewidth=0.8, zorder=3)
    ax.axhline(100, color=PALETTE["secure"], ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Agreement (%)"); ax.set_title("Alice–Bob Key Agreement",
                                                  fontweight="bold")
    ax.set_ylim(0, 110); ax.grid(axis="y", alpha=0.3, zorder=0)
    _label_bars(ax, bars3, agree)

    fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))
    _save_and_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
# PLOT 2 — QBER vs EVE INTERCEPT RATE
# ══════════════════════════════════════════════════════════════════════

def plot_qber_vs_intercept_rate(
    n_qubits:  int = 400,
    steps:     int = 10,
    save_path: str = "qkd_qber_vs_eve.png",
) -> None:
    """Sweep Eve's intercept probability and plot QBER vs theoretical."""
    from bb84_runner import run_simulation
    probs = np.linspace(0, 1, steps + 1)
    qbers, ci_lo, ci_hi = [], [], []

    print("  Sweeping Eve intercept probability...")
    for p in probs:
        cfg = SimulationConfig(n_qubits=n_qubits, eve_present=(p > 0),
                               eve_intercept_prob=float(p), seed=42)
        r = run_simulation(cfg, verbose=False)
        qbers.append(r.qber_result.qber * 100)
        ci_lo.append(r.qber_result.confidence_low  * 100)
        ci_hi.append(r.qber_result.confidence_high * 100)
        print(f"    p={p:.2f}  QBER={qbers[-1]:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(probs * 100, qbers, "b-o", lw=2, ms=6, label="Simulated QBER")
    ax.fill_between(probs * 100, ci_lo, ci_hi, alpha=0.2, color="blue",
                    label="95 % Confidence Interval")
    ax.plot(probs * 100, [p * 25 for p in probs], "r--", lw=1.5,
            label="Theoretical  QBER = 0.25 × p")
    ax.axhline(11, color=PALETTE["abort"],   ls=":", lw=1.5, label="Abort (11 %)")
    ax.axhline(5,  color=PALETTE["warning"], ls=":", lw=1.5, label="Warning (5 %)")
    ax.set_xlabel("Eve Intercept Probability (%)", fontsize=11)
    ax.set_ylabel("QBER (%)", fontsize=11)
    ax.set_title("QBER vs Eve Intercept Rate" + TITLE_SUFFIX, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    _save_and_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
# PLOT 3 — NOISE MODEL COMPARISON  (Phase 3)
# ══════════════════════════════════════════════════════════════════════

def plot_noise_model_comparison(
    save_path: str = "qkd_noise_models.png",
) -> None:
    """Compare QBER contribution from different noise models."""
    # Pure analytical (no simulation needed for clarity)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 3 — Channel Noise Model Comparison" + TITLE_SUFFIX,
                 fontsize=11, fontweight="bold", y=1.04)

    # ── Panel 1: Depolarizing ─────────────────────────────────────────
    ax   = axes[0]
    ps   = np.linspace(0, 0.20, 200)
    qb   = ps * 3 / 4 * 100
    ax.plot(ps * 100, qb, color=PALETTE["blue"], lw=2)
    ax.axhline(11, color=PALETTE["abort"],   ls="--", lw=1.5)
    ax.axhline(5,  color=PALETTE["warning"], ls="--", lw=1.5)
    ax.fill_between(ps * 100, 0, qb, alpha=0.1, color=PALETTE["blue"])
    ax.set_xlabel("Depolarizing prob p (%)"); ax.set_ylabel("QBER (%)")
    ax.set_title("Depolarizing Noise", fontweight="bold"); ax.grid(alpha=0.3)

    # ── Panel 2: T1/T2 Damping ────────────────────────────────────────
    ax = axes[1]
    t1s  = np.logspace(3, 6, 100)   # T1 from 1 µs to 1 ms in ns
    gate = 50.0
    qb_amp  = (1 - np.exp(-gate / t1s)) / 2 * 100
    qb_pha  = (1 - np.exp(-gate / (t1s * 0.5))) / 4 * 100   # T2 ≈ T1/2
    ax.semilogx(t1s / 1e3, qb_amp, color=PALETTE["purple"], lw=2,
                label="Amplitude damping (T1)")
    ax.semilogx(t1s / 1e3, qb_pha, color=PALETTE["teal"],   lw=2,
                label="Phase damping  (T2 ≈ T1/2)")
    ax.axhline(11, color=PALETTE["abort"],   ls="--", lw=1.5)
    ax.axhline(5,  color=PALETTE["warning"], ls="--", lw=1.5)
    ax.set_xlabel("T1 (µs)"); ax.set_ylabel("QBER (%)")
    ax.set_title("Damping Noise Models", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # ── Panel 3: Fiber Loss ────────────────────────────────────────────
    ax   = axes[2]
    from bb84_noise import fiber_transmittance
    dists = np.linspace(0, 150, 200)
    eta   = np.array([fiber_transmittance(L) for L in dists])
    Y0    = 1.7e-6
    mu    = 0.5
    Q     = np.maximum(1 - np.exp(-mu * eta), Y0)
    e_mu  = np.minimum(0.5 * Y0 / Q + 0.005, 0.5) * 100
    ax.plot(dists, e_mu, color=PALETTE["teal"], lw=2, label="QBER")
    ax.plot(dists, (1 - eta) * 100, color=PALETTE["grey"], lw=2, ls="--",
            label="Loss rate")
    ax.axhline(11, color=PALETTE["abort"],   ls=":", lw=1.5)
    ax.axhline(5,  color=PALETTE["warning"], ls=":", lw=1.5)
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Rate (%)")
    ax.set_title("Fiber Loss (SMF-28, 0.2 dB/km)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    _save_and_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
# PLOT 4 — SECRET KEY RATE vs DISTANCE  (Phase 3 + Phase 4)
# ══════════════════════════════════════════════════════════════════════

def plot_skr_vs_distance(
    mu:        float = 0.5,
    f_EC:      float = 1.16,
    save_path: str   = "qkd_skr_vs_distance.png",
) -> None:
    """Plot secret key rate (three formulas) vs fiber distance."""
    data = sweep_skr_vs_distance(mu=mu, f_EC=f_EC)
    dists = data["distances"]
    max_d = data["max_distance_km"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Secret Key Rate vs Fiber Distance" + TITLE_SUFFIX,
                 fontsize=11, fontweight="bold", y=1.04)

    # ── Panel 1: SKR (log scale) ──────────────────────────────────────
    for key, label, color, ls in [
        ("skr_shor_preskill", "Shor-Preskill (ideal)",  PALETTE["blue"],   "-"),
        ("skr_realistic",     f"Realistic (f={f_EC})", PALETTE["purple"], "--"),
        ("skr_gllp",          "GLLP w/ decoy state",   PALETTE["teal"],   "-."),
    ]:
        y = np.where(data[key] > 1e-10, data[key], np.nan)
        ax1.semilogy(dists, y, color=color, ls=ls, lw=2, label=label)

    ax1.axvline(max_d, color=PALETTE["abort"], ls=":", lw=2,
                label=f"Max secure dist ≈ {max_d:.0f} km")
    ax1.set_xlabel("Fiber Distance (km)", fontsize=11)
    ax1.set_ylabel("Secret Key Rate (bits/pulse)", fontsize=11)
    ax1.set_title("SKR vs Distance (log scale)", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, which="both")
    ax1.set_xlim(0, max(dists)); ax1.set_ylim(1e-7, 1)

    # ── Panel 2: QBER and transmittance ──────────────────────────────
    ax2.plot(dists, data["eta"] * 100, color=PALETTE["blue"], lw=2,
             label="Transmittance η (%)")
    ax2.plot(dists, data["qbers"],       color=PALETTE["abort"],  lw=2, ls="--",
             label="QBER (%)")
    ax2.axhline(11, color=PALETTE["abort"],   ls=":", lw=1.5, alpha=0.7)
    ax2.axhline(5,  color=PALETTE["warning"], ls=":", lw=1.5, alpha=0.7)
    ax2.set_xlabel("Fiber Distance (km)", fontsize=11)
    ax2.set_ylabel("Rate (%)", fontsize=11)
    ax2.set_title("Transmittance & QBER vs Distance", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_xlim(0, max(dists))

    _save_and_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
# PLOT 5 — SKR vs QBER  (theory curves)
# ══════════════════════════════════════════════════════════════════════

def plot_skr_vs_qber(
    save_path: str = "qkd_skr_vs_qber.png",
) -> None:
    """Analytical SKR vs QBER for different EC efficiencies."""
    data    = sweep_skr_vs_qber()
    qbers   = data["qbers"]
    sp      = data["skr_shor_preskill"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(qbers, sp, color=PALETTE["blue"], lw=2.5, label="Shor-Preskill (ideal)")

    for f, color in [(1.0, PALETTE["secure"]), (1.16, PALETTE["teal"]),
                     (1.3,  PALETTE["warning"]), (1.5, PALETTE["abort"])]:
        k = f"skr_f{f:.2f}".replace(".", "p")
        if k in data:
            ax.plot(qbers, data[k], lw=2, color=color, ls="--",
                    label=f"Realistic f = {f:.2f}")

    ax.axvline(11, color=PALETTE["abort"],   ls=":", lw=2, alpha=0.7,
               label="Abort threshold (11 %)")
    ax.axvline(5,  color=PALETTE["warning"], ls=":", lw=2, alpha=0.7,
               label="Warning threshold (5 %)")
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(qbers, 0, sp, where=sp > 0, alpha=0.07, color=PALETTE["blue"])

    ax.set_xlabel("QBER (%)", fontsize=11)
    ax.set_ylabel("Secret Key Rate (bits/sifted bit)", fontsize=11)
    ax.set_title("Secret Key Rate vs QBER — BB84" + TITLE_SUFFIX, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, 20); ax.set_ylim(-0.05, 1.05)
    _save_and_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
# PLOT 6 — PNS VULNERABILITY ANALYSIS  (Phase 4)
# ══════════════════════════════════════════════════════════════════════

def plot_pns_analysis(
    save_path: str = "qkd_pns_analysis.png",
) -> None:
    """Photon-number statistics and PNS vulnerability vs μ."""
    data = sweep_pns_vulnerability()
    mu   = data["mu_vals"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase 4 — PNS Attack: Photon Number Analysis" + TITLE_SUFFIX,
                 fontsize=11, fontweight="bold", y=1.04)

    # ── Panel 1: Poisson probabilities ────────────────────────────────
    ax1.stackplot(mu,
                  data["p_vacuum"] * 100,
                  data["p_single"] * 100,
                  data["p_multi"]  * 100,
                  labels=["Vacuum n=0", "Single-photon n=1", "Multi-photon n≥2"],
                  colors=[PALETTE["grey"], PALETTE["secure"], PALETTE["abort"]],
                  alpha=0.8)
    ax1.axvline(0.1, color="black", ls="--", lw=1.5, label="μ=0.1 (practical)")
    ax1.axvline(0.5, color="black", ls=":",  lw=1.5, label="μ=0.5 (typical)")
    ax1.set_xlabel("Mean photon number μ", fontsize=11)
    ax1.set_ylabel("Probability (%)", fontsize=11)
    ax1.set_title("Poisson Photon-Number Distribution", fontweight="bold")
    ax1.legend(fontsize=8, loc="center right"); ax1.grid(alpha=0.3)
    ax1.set_xlim(0, max(mu))

    # ── Panel 2: PNS vulnerability ────────────────────────────────────
    ax2.plot(mu, data["eve_free_info"] * 100, color=PALETTE["abort"], lw=2.5,
             label="Eve's free info (multi-photon %)")
    ax2.plot(mu, data["safe_fraction"] * 100, color=PALETTE["secure"], lw=2.5,
             ls="--", label="Single-photon fraction (secure %)")
    ax2.axvline(0.1, color="black", ls="--", lw=1.5, alpha=0.7)
    ax2.axvline(data["optimal_mu"], color=PALETTE["blue"], ls="-.", lw=2,
                label=f"Optimal μ ≈ {data['optimal_mu']:.2f}")
    ax2.set_xlabel("Mean photon number μ", fontsize=11)
    ax2.set_ylabel("Fraction (%)", fontsize=11)
    ax2.set_title("PNS Vulnerability vs μ", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.set_xlim(0, max(mu))

    _save_and_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
# PLOT 7 — DECOY STATE EFFECTIVENESS  (Phase 4)
# ══════════════════════════════════════════════════════════════════════

def plot_decoy_effectiveness(
    save_path: str = "qkd_decoy_state.png",
) -> None:
    """Compare SKR with and without decoy state over distance."""
    data  = sweep_decoy_effectiveness()
    dists = data["distances"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase 4 — Decoy State Protocol Effectiveness" + TITLE_SUFFIX,
                 fontsize=11, fontweight="bold", y=1.04)

    # ── Panel 1: SKR comparison ───────────────────────────────────────
    nd = np.where(data["skr_no_decoy"]   > 1e-10, data["skr_no_decoy"],   np.nan)
    wd = np.where(data["skr_with_decoy"] > 1e-10, data["skr_with_decoy"], np.nan)
    ax1.semilogy(dists, nd, color=PALETTE["abort"],  lw=2, label="No decoy state")
    ax1.semilogy(dists, wd, color=PALETTE["secure"], lw=2, label="With decoy state (3 intensities)")
    ax1.set_xlabel("Distance (km)", fontsize=11); ax1.set_ylabel("SKR (bits/pulse)", fontsize=11)
    ax1.set_title("SKR With vs Without Decoy State", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, which="both")
    ax1.set_xlim(0, max(dists))

    # ── Panel 2: Gain factor ──────────────────────────────────────────
    ax2.plot(dists, data["gain_factor"], color=PALETTE["blue"], lw=2.5)
    ax2.axhline(1, color="black", lw=0.8, ls="--")
    ax2.fill_between(dists, 1, data["gain_factor"],
                     where=data["gain_factor"] >= 1,
                     alpha=0.15, color=PALETTE["secure"], label="SKR gain from decoy")
    ax2.set_xlabel("Distance (km)", fontsize=11)
    ax2.set_ylabel("SKR gain factor (with/without decoy)", fontsize=11)
    ax2.set_title("SKR Improvement from Decoy State", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_xlim(0, max(dists)); ax2.set_ylim(0)

    _save_and_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
# PLOT 8 — 2D SECURITY HEATMAP  (novel research)
# ══════════════════════════════════════════════════════════════════════

def plot_2d_security_heatmap(
    noise_levels:    Optional[List[float]] = None,
    intercept_probs: Optional[List[float]] = None,
    n_qubits:        int   = 250,
    save_path:       str   = "qkd_security_heatmap.png",
) -> None:
    """
    2D heatmap: SKR over the (noise level, intercept probability) plane.
    Shows the joint security boundary — the core novel research result.
    """
    import numpy as np

    data = full_parameter_sweep(
        noise_levels    = noise_levels,
        intercept_probs = intercept_probs,
        n_qubits        = n_qubits,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Novel Research — 2D Security Parameter Space" + TITLE_SUFFIX,
                 fontsize=11, fontweight="bold", y=1.04)

    NP = data["noise_levels"] * 100
    EP = data["intercept_probs"] * 100

    # ── Panel 1: SKR heatmap ──────────────────────────────────────────
    im1 = ax1.pcolormesh(EP, NP, data["skr_grid"],
                          cmap="RdYlGn", vmin=0, vmax=0.5)
    cb1 = fig.colorbar(im1, ax=ax1)
    cb1.set_label("Secret Key Rate (bits/sifted bit)", fontsize=9)
    if data["secure_boundary"]:
        bx = [p * 100 for _, p in data["secure_boundary"]]
        by = [n * 100 for n, _ in data["secure_boundary"]]
        ax1.plot(bx, by, "w--", lw=2, label="SKR = 0 boundary")
        ax1.legend(fontsize=8)
    ax1.set_xlabel("Eve Intercept Probability (%)", fontsize=11)
    ax1.set_ylabel("Depolarizing Noise Level (%)", fontsize=11)
    ax1.set_title("Secret Key Rate Heatmap\n(Secure = green, Insecure = red)",
                  fontweight="bold")

    # ── Panel 2: QBER heatmap ─────────────────────────────────────────
    im2 = ax2.pcolormesh(EP, NP, data["qber_grid"],
                          cmap="YlOrRd", vmin=0, vmax=35)
    cb2 = fig.colorbar(im2, ax=ax2)
    cb2.set_label("QBER (%)", fontsize=9)
    # Draw security threshold contours
    CS  = ax2.contour(EP, NP, data["qber_grid"],
                       levels=[5, 11], colors=["orange", "red"],
                       linewidths=2, linestyles=["--", "-"])
    ax2.clabel(CS, fmt="%d%%", fontsize=9)
    ax2.set_xlabel("Eve Intercept Probability (%)", fontsize=11)
    ax2.set_ylabel("Depolarizing Noise Level (%)", fontsize=11)
    ax2.set_title("QBER Heatmap\n(contours at 5 % and 11 % thresholds)",
                  fontweight="bold")

    _save_and_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
# PLOT 9 — INFORMATION BUDGET  (Phase 5)
# ══════════════════════════════════════════════════════════════════════

def plot_information_budget(
    scenarios: List[Tuple[str, SimulationConfig]],
    results:   List[SimulationResult],
    save_path: Optional[str] = "qkd_info_budget.png",
) -> None:
    """
    Stacked bar: I(A;B), χ(Eve), and secure key rate per scenario.
    Visualises the information budget of each run.
    """
    labels = [s[0] for s in scenarios]
    x      = np.arange(len(labels))

    iab  = []
    chi  = []
    skr  = []
    for r in results:
        sa = r.security_analysis
        if sa:
            iab.append(sa.mutual_info_alice_bob)
            chi.append(sa.holevo_bound_eve)
            skr.append(max(0.0, sa.secret_key_rate_realistic))
        else:
            qber = r.qber_result.qber
            from bb84_postprocessing import (
                binary_entropy, mutual_information_alice_bob,
                secret_key_rate_realistic
            )
            iab.append(mutual_information_alice_bob(qber))
            chi.append(r.eve_interception_rate * 0.5)
            skr.append(max(0.0, secret_key_rate_realistic(qber)))

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Information Budget — I(A;B), χ(Eve), SKR" + TITLE_SUFFIX,
                 fontsize=11, fontweight="bold", y=1.04)

    w = 0.26
    ax.bar(x - w, iab, w, label="I(A;B) mutual info",  color=PALETTE["blue"],   alpha=0.85)
    ax.bar(x,     chi, w, label="χ(Eve) Holevo bound",  color=PALETTE["abort"],  alpha=0.85)
    ax.bar(x + w, skr, w, label="SKR (realistic f=1.16)", color=PALETTE["secure"], alpha=0.85)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("bits per sifted key bit", fontsize=10)
    ax.set_title("Information Budget per Scenario", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    _save_and_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
# PLOT 10 — CASCADE EC PERFORMANCE  (Phase 5)
# ══════════════════════════════════════════════════════════════════════

def plot_cascade_performance(
    save_path: str = "qkd_cascade_ec.png",
) -> None:
    """
    Visualise Cascade error correction: errors remaining vs passes,
    and information leakage vs QBER.
    """
    from bb84_postprocessing import CascadeErrorCorrection
    import random

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase 5 — Cascade Error Correction Performance" + TITLE_SUFFIX,
                 fontsize=11, fontweight="bold", y=1.04)

    # ── Panel 1: errors vs passes at different QBER levels ───────────
    n      = 500
    qbers_test = [0.03, 0.06, 0.08, 0.10]
    colors_ec  = [PALETTE["secure"], PALETTE["teal"],
                  PALETTE["warning"], PALETTE["abort"]]

    for qber, col in zip(qbers_test, colors_ec):
        rng = random.Random(42)
        alice_k = [rng.randint(0, 1) for _ in range(n)]
        bob_k   = [b ^ (rng.random() < qber) for b in alice_k]

        pass_errors = []
        for p in range(1, 6):
            casc  = CascadeErrorCorrection(qber_estimate=qber, passes=p, seed=42)
            cr    = casc.correct(alice_k, bob_k)
            pass_errors.append(cr.errors_remaining)

        ax1.plot(range(1, 6), pass_errors, "o-", color=col, lw=2, ms=7,
                 label=f"QBER = {qber * 100:.0f} %")

    ax1.set_xlabel("Cascade Passes", fontsize=11)
    ax1.set_ylabel("Remaining Errors", fontsize=11)
    ax1.set_title("Error Reduction vs Passes", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.set_xticks(range(1, 6))

    # ── Panel 2: leakage vs QBER ──────────────────────────────────────
    qb_range = np.linspace(0.01, 0.12, 30)
    leakages = []
    rng2     = random.Random(42)

    for qber in qb_range:
        alice_k = [rng2.randint(0, 1) for _ in range(n)]
        bob_k   = [b ^ (rng2.random() < qber) for b in alice_k]
        casc    = CascadeErrorCorrection(qber_estimate=qber, passes=4, seed=42)
        cr      = casc.correct(alice_k, bob_k)
        leakages.append(cr.bits_leaked / n * 100)

    # Theoretical: f_EC × H(QBER) × 100
    theoretical = np.array([1.16 * binary_entropy(e) * 100 for e in qb_range])

    ax2.plot(qb_range * 100, leakages,     color=PALETTE["blue"],  lw=2.5,
             label="Cascade leakage (simulated)")
    ax2.plot(qb_range * 100, theoretical, color=PALETTE["grey"],  lw=2, ls="--",
             label="f_EC × H(e) theoretical")
    ax2.set_xlabel("QBER (%)", fontsize=11)
    ax2.set_ylabel("Leakage (% of key)", fontsize=11)
    ax2.set_title("Information Leakage vs QBER", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 12)

    _save_and_show(fig, save_path)
