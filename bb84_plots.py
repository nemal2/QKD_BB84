"""
bb84_plots.py
=============
All visualisation for the BB84 QKD simulator.

Exports
-------
plot_comparison()             2-panel bar chart for multi-scenario runs
plot_qber_vs_intercept_rate() sweep Eve intercept prob 0→100 %, plot QBER
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bb84_config import SimulationConfig, SimulationResult
from bb84_runner import run_simulation


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _status_color(r: SimulationResult) -> str:
    s = r.qber_result.security_status
    if "SECURE"  in s: return "#2ecc71"
    if "WARNING" in s: return "#e67e22"
    return "#e74c3c"


_LEGEND_PATCHES = [
    mpatches.Patch(color="#2ecc71", label="Secure  (QBER < 5 %)"),
    mpatches.Patch(color="#e67e22", label="Warning (5 – 11 %)"),
    mpatches.Patch(color="#e74c3c", label="Abort   (QBER ≥ 11 %)"),
]


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1 — SCENARIO COMPARISON  (2-panel)
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    scenarios: List[Tuple[str, SimulationConfig]],
    results:   List[SimulationResult],
    save_path: Optional[str] = "qkd_comparison.png",
) -> None:
    """
    Two-panel bar chart:
      Panel 1  QBER (%) with 95 % CI error bars + security thresholds
      Panel 2  Alice–Bob key agreement rate (%)

    Parameters
    ──────────
    scenarios  : list of (name, config) pairs — names used as axis labels
    results    : corresponding SimulationResult objects (same order)
    save_path  : PNG output path; pass None to skip saving
    """
    labels = [s[0] for s in scenarios]
    qbers  = [r.qber_result.qber * 100            for r in results]
    ci_low = [r.qber_result.confidence_low * 100  for r in results]
    ci_high= [r.qber_result.confidence_high * 100 for r in results]
    agree  = [r.key_agreement_rate * 100          for r in results]
    colors = [_status_color(r)                    for r in results]

    yerr_low  = [q - lo for q, lo in zip(qbers, ci_low)]
    yerr_high = [hi - q for q, hi in zip(qbers, ci_high)]

    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "BB84 QKD Simulation — Scenario Comparison\n"
        "University of Ruhuna, Dept. of Computer Engineering",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # ── Panel 1: QBER ─────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, qbers, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.8, zorder=3)
    ax.errorbar(x, qbers, yerr=[yerr_low, yerr_high],
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)
    ax.axhline(11, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="Abort threshold (11 %)")
    ax.axhline(5,  color="#e67e22", linestyle="--", linewidth=1.5,
               label="Warning threshold (5 %)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("QBER (%)")
    ax.set_title("Quantum Bit Error Rate", fontweight="bold")
    ax.set_ylim(0, max(35, max(qbers) + 6))
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars, qbers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f} %", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    # ── Panel 2: Key Agreement Rate ───────────────────────────────────────
    ax = axes[1]
    bars2 = ax.bar(x, agree, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.8, zorder=3)
    ax.axhline(100, color="#2ecc71", linestyle="--", linewidth=1.2,
               label="Perfect agreement (100 %)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Agreement (%)")
    ax.set_title("Alice–Bob Key Agreement\n(before error correction)",
                 fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars2, agree):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f} %", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    fig.legend(handles=_LEGEND_PATCHES, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved → {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2 — QBER vs EVE INTERCEPT RATE  (research sweep)
# ──────────────────────────────────────────────────────────────────────────────

def plot_qber_vs_intercept_rate(
    n_qubits:  int  = 500,
    steps:     int  = 10,
    save_path: str  = "qkd_qber_vs_eve.png",
) -> None:
    """
    Sweep Eve's intercept probability from 0 % → 100 % and plot QBER.

    Theory predicts QBER = 0.25 × p_intercept for intercept-resend.
    This experiment verifies that relationship empirically.
    """
    print("\n  [Experiment] QBER vs Eve Intercept Rate sweep...")

    probs   = np.linspace(0, 1, steps + 1)
    qbers:  List[float] = []
    ci_low: List[float] = []
    ci_hi:  List[float] = []

    for p in probs:
        cfg = SimulationConfig(
            n_qubits=n_qubits,
            eve_present=(p > 0),
            eve_intercept_prob=float(p),
            noise_enabled=False,
            seed=42,
        )
        r = run_simulation(cfg, verbose=False)
        qbers.append(r.qber_result.qber * 100)
        ci_low.append(r.qber_result.confidence_low  * 100)
        ci_hi.append(r.qber_result.confidence_high * 100)
        print(f"    p = {p:.2f}  →  QBER = {qbers[-1]:.1f} %")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(probs * 100, qbers, "b-o", linewidth=2,
            markersize=6, label="Simulated QBER")
    ax.fill_between(probs * 100, ci_low, ci_hi,
                    alpha=0.2, color="blue", label="95 % Confidence Interval")

    theoretical = [0.25 * p * 100 for p in probs]
    ax.plot(probs * 100, theoretical, "r--", linewidth=1.5,
            label="Theoretical  QBER = 0.25 × p")

    ax.axhline(11, color="#e74c3c", linestyle=":", linewidth=1.5,
               label="Abort threshold (11 %)")
    ax.axhline(5,  color="#e67e22", linestyle=":", linewidth=1.5,
               label="Warning threshold (5 %)")

    ax.set_xlabel("Eve's Intercept Probability (%)", fontsize=11)
    ax.set_ylabel("QBER (%)", fontsize=11)
    ax.set_title(
        "QBER vs Eve Intercept Rate\n(Intercept-Resend Attack, BB84)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved → {save_path}")
    plt.show()