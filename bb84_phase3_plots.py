"""
bb84_phase3_plots.py  ·  Phase 3 — Noise Visualisation
══════════════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Exports
───────
plot_noise_model_comparison()    – bar chart: QBER across all noise models
plot_qber_vs_depolar_prob()      – sweep depolarizing p → QBER
plot_qber_vs_t1()                – sweep T1 → QBER  (amplitude damping)
plot_qber_vs_t2()                – sweep T2 → QBER  (phase damping)
plot_fiber_loss_analysis()       – QBER + key rate vs fibre distance

All functions return the matplotlib Figure so you can customise further
in the notebook:
    fig = plot_qber_vs_t1(...)
    fig.axes[0].set_title("My custom title")
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bb84_config import SimulationConfig, SimulationResult
from bb84_runner import run_simulation
from bb84_noise  import NoiseModelType


# ──────────────────────────────────────────────────────────────────────
# SHARED STYLE HELPERS
# ──────────────────────────────────────────────────────────────────────

_COLORS = {
    NoiseModelType.DEPOLARIZING:   "#3498db",   # blue
    NoiseModelType.AMPLITUDE_DAMP: "#e67e22",   # orange
    NoiseModelType.PHASE_DAMP:     "#9b59b6",   # purple
    NoiseModelType.COMBINED:       "#e74c3c",   # red
    NoiseModelType.FIBER_LOSS:     "#2ecc71",   # green
    "ideal":                       "#95a5a6",   # grey
    "eve":                         "#c0392b",   # dark red
}

_STATUS_COLOR = {
    "SECURE ✓":  "#2ecc71",
    "WARNING ⚠": "#e67e22",
    "ABORT ✗":   "#e74c3c",
}


def _status_color(r: SimulationResult) -> str:
    for key, color in _STATUS_COLOR.items():
        if key in r.qber_result.security_status:
            return color
    return "#95a5a6"


def _add_thresholds(ax, xmin=None, xmax=None):
    """Draw ABORT (11%) and WARNING (5%) horizontal threshold lines."""
    kw = dict(linestyle="--", linewidth=1.4, zorder=2)
    ax.axhline(11, color="#e74c3c", label="Abort threshold (11%)", **kw)
    ax.axhline(5,  color="#e67e22", label="Warning threshold (5%)", **kw)


def _add_threshold_bands(ax):
    """Fill background bands for SECURE / WARNING / ABORT regions."""
    ax.axhspan(0,  5,  alpha=0.06, color="#2ecc71", zorder=0)
    ax.axhspan(5,  11, alpha=0.06, color="#e67e22", zorder=0)
    ax.axhspan(11, ax.get_ylim()[1] if ax.get_ylim()[1] > 11 else 35,
               alpha=0.06, color="#e74c3c", zorder=0)


# ──────────────────────────────────────────────────────────────────────
# PLOT 1 — NOISE MODEL COMPARISON  (bar chart)
# ──────────────────────────────────────────────────────────────────────

def plot_noise_model_comparison(
    n_qubits:  int  = 600,
    t1_ns:     float = 10_000,
    t2_ns:     float = 8_000,
    gate_ns:   float = 50.0,
    depolar_p: float = 0.05,
    fiber_km:  float = 50.0,
    save_path: Optional[str] = "qkd_noise_comparison.png",
) -> plt.Figure:
    """
    Run one simulation per noise model and produce a 3-panel comparison.

    Panels
    ──────
    1. QBER (%) with 95 % CI and security thresholds
    2. Final key length (bits)  — shows how noise reduces usable key
    3. Key agreement rate (%)   — shows raw error rate before Phase 5 EC

    The ideal and Eve=100% baselines are included for reference.

    Educational purpose
    ───────────────────
    Students can visually compare how different physical noise mechanisms
    (T1 relaxation, T2 dephasing, fibre loss) affect security metrics
    differently even at the same gate-error rate.
    """

    scenarios: List[Tuple[str, SimulationConfig, str]] = [
        ("Ideal",
         SimulationConfig(n_qubits=n_qubits, label="Ideal"),
         _COLORS["ideal"]),

        ("Eve 100%\n(intercept-resend)",
         SimulationConfig(n_qubits=n_qubits, eve_present=True,
                          eve_intercept_prob=1.0, label="Eve 100%"),
         _COLORS["eve"]),

        (f"Depolarizing\n(p={depolar_p})",
         SimulationConfig(n_qubits=n_qubits, noise_enabled=True,
                          noise_model=NoiseModelType.DEPOLARIZING,
                          depolar_prob=depolar_p, label="Depolar"),
         _COLORS[NoiseModelType.DEPOLARIZING]),

        (f"Amplitude\nDamping\n(T1={t1_ns/1000:.0f}µs)",
         SimulationConfig(n_qubits=n_qubits, noise_enabled=True,
                          noise_model=NoiseModelType.AMPLITUDE_DAMP,
                          t1_ns=t1_ns, gate_time_ns=gate_ns, label="Amp.Damp"),
         _COLORS[NoiseModelType.AMPLITUDE_DAMP]),

        (f"Phase\nDamping\n(T2={t2_ns/1000:.0f}µs)",
         SimulationConfig(n_qubits=n_qubits, noise_enabled=True,
                          noise_model=NoiseModelType.PHASE_DAMP,
                          t2_ns=t2_ns, gate_time_ns=gate_ns, label="Phase.Damp"),
         _COLORS[NoiseModelType.PHASE_DAMP]),

        (f"Combined\nT1+T2",
         SimulationConfig(n_qubits=n_qubits, noise_enabled=True,
                          noise_model=NoiseModelType.COMBINED,
                          t1_ns=t1_ns, t2_ns=t2_ns, gate_time_ns=gate_ns,
                          label="Combined"),
         _COLORS[NoiseModelType.COMBINED]),

        (f"Fiber Loss\n({fiber_km:.0f} km)",
         SimulationConfig(n_qubits=n_qubits + 200, noise_enabled=True,
                          noise_model=NoiseModelType.FIBER_LOSS,
                          channel_length_km=fiber_km, label="Fiber"),
         _COLORS[NoiseModelType.FIBER_LOSS]),
    ]

    print("  Running noise model comparison...")
    results = []
    for name, cfg, _ in scenarios:
        r = run_simulation(cfg, verbose=False)
        results.append(r)
        print(f"    {cfg.label:<12}  QBER={r.qber_result.qber*100:.1f}%  "
              f"Key={r.key_length}b  {r.qber_result.security_status}")

    labels   = [s[0] for s in scenarios]
    colors   = [s[2] for s in scenarios]
    qbers    = [r.qber_result.qber * 100 for r in results]
    ci_low   = [(r.qber_result.qber - r.qber_result.confidence_low) * 100 for r in results]
    ci_high  = [(r.qber_result.confidence_high - r.qber_result.qber) * 100 for r in results]
    key_lens = [r.key_length for r in results]
    agree    = [r.key_agreement_rate * 100 for r in results]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "BB84 Phase 3 — Noise Model Comparison\n"
        "University of Ruhuna  ·  Dept. of Computer Engineering",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # ── Panel 1: QBER ─────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, qbers, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.7, zorder=3)
    ax.errorbar(x, qbers, yerr=[ci_low, ci_high],
                fmt="none", color="black", capsize=4, linewidth=1.3, zorder=4)
    _add_thresholds(ax)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("QBER (%)"); ax.set_ylim(0, max(35, max(qbers) + 5))
    ax.set_title("Quantum Bit Error Rate", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25, zorder=0)
    for bar, val in zip(bars, qbers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # ── Panel 2: Final Key Length ──────────────────────────────────────
    ax = axes[1]
    bars2 = ax.bar(x, key_lens, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.7, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Bits"); ax.set_title("Final Key Length", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for bar, val in zip(bars2, key_lens):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # ── Panel 3: Key Agreement ─────────────────────────────────────────
    ax = axes[2]
    bars3 = ax.bar(x, agree, color=colors, alpha=0.85,
                   edgecolor="black", linewidth=0.7, zorder=3)
    ax.axhline(100, color="#2ecc71", linestyle="--", linewidth=1.2,
               label="Perfect (100%)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Agreement (%)"); ax.set_ylim(0, 110)
    ax.set_title("Alice–Bob Key Agreement\n(before error correction)",
                 fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25, zorder=0)
    for bar, val in zip(bars3, agree):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Secure  (QBER < 5%)"),
        mpatches.Patch(color="#e67e22", label="Warning (5–11%)"),
        mpatches.Patch(color="#e74c3c", label="Abort   (QBER ≥ 11%)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved → {save_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────
# PLOT 2 — QBER vs DEPOLARIZING PROBABILITY  (sweep)
# ──────────────────────────────────────────────────────────────────────

def plot_qber_vs_depolar_prob(
    n_qubits:  int   = 400,
    steps:     int   = 12,
    p_max:     float = 0.15,
    save_path: Optional[str] = "qkd_qber_vs_depolar.png",
) -> plt.Figure:
    """
    Sweep depolarizing probability p from 0 → p_max and plot QBER.

    Theory: for a depolarizing channel acting on one qubit before
    measurement, the expected QBER increase is p * (2/3) because
    Y errors cause errors in both bases.
    """
    print("  Sweeping depolarizing probability...")
    probs  = np.linspace(0, p_max, steps + 1)
    qbers, ci_l, ci_h = [], [], []

    for p in probs:
        cfg = SimulationConfig(
            n_qubits=n_qubits,
            noise_enabled=(p > 0),
            noise_model=NoiseModelType.DEPOLARIZING,
            depolar_prob=float(p), seed=42,
        )
        r = run_simulation(cfg, verbose=False)
        qbers.append(r.qber_result.qber * 100)
        ci_l.append(r.qber_result.confidence_low  * 100)
        ci_h.append(r.qber_result.confidence_high * 100)
        print(f"    p={p:.3f}  →  QBER={qbers[-1]:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(probs * 100, qbers, "o-", color=_COLORS[NoiseModelType.DEPOLARIZING],
            linewidth=2, markersize=6, label="Simulated QBER", zorder=3)
    ax.fill_between(probs * 100, ci_l, ci_h,
                    alpha=0.2, color=_COLORS[NoiseModelType.DEPOLARIZING],
                    label="95% CI")
    _add_thresholds(ax)
    ax.set_xlabel("Depolarizing probability p (%)", fontsize=11)
    ax.set_ylabel("QBER (%)", fontsize=11)
    ax.set_title("QBER vs Depolarizing Noise\n(Lindblad: L ∈ {X, Y, Z}, equal weight)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.set_ylim(0, None)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved → {save_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────
# PLOT 3 — QBER vs T1  (amplitude damping sweep)
# ──────────────────────────────────────────────────────────────────────

def plot_qber_vs_t1(
    n_qubits:    int   = 400,
    gate_ns:     float = 50.0,
    t1_min_us:   float = 1.0,
    t1_max_us:   float = 200.0,
    steps:       int   = 12,
    save_path:   Optional[str] = "qkd_qber_vs_t1.png",
) -> plt.Figure:
    """
    Sweep T1 (relaxation time) from t1_min_us → t1_max_us µs.

    Physical interpretation
    ───────────────────────
    Short T1  → qubit decays from |1⟩ to |0⟩ quickly
              → high QBER (|1⟩ states become |0⟩ errors)
    Long  T1  → qubit is stable during gate operations
              → QBER approaches ideal (0%)

    This is the dominant noise source in superconducting qubits.
    Typical values: T1 = 10 – 500 µs (state of the art: > 1 ms)

    Lindblad operator: L = √(γ) |0⟩⟨1|  where γ = 1 − exp(−t/T1)
    """
    print("  Sweeping T1 (amplitude damping)...")
    t1_values_us = np.linspace(t1_min_us, t1_max_us, steps + 1)
    qbers, ci_l, ci_h = [], [], []

    for t1_us in t1_values_us:
        cfg = SimulationConfig(
            n_qubits=n_qubits,
            noise_enabled=True,
            noise_model=NoiseModelType.AMPLITUDE_DAMP,
            t1_ns=t1_us * 1000,
            gate_time_ns=gate_ns, seed=42,
        )
        r = run_simulation(cfg, verbose=False)
        qbers.append(r.qber_result.qber * 100)
        ci_l.append(r.qber_result.confidence_low  * 100)
        ci_h.append(r.qber_result.confidence_high * 100)
        print(f"    T1={t1_us:.0f}µs  γ={1 - np.exp(-gate_ns/(t1_us*1000)):.5f}"
              f"  →  QBER={qbers[-1]:.1f}%")

    # Compute γ for secondary x-axis annotation
    gammas = [1 - np.exp(-gate_ns / (t * 1000)) for t in t1_values_us]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t1_values_us, qbers, "o-",
            color=_COLORS[NoiseModelType.AMPLITUDE_DAMP],
            linewidth=2, markersize=6, label="Simulated QBER", zorder=3)
    ax.fill_between(t1_values_us, ci_l, ci_h,
                    alpha=0.2, color=_COLORS[NoiseModelType.AMPLITUDE_DAMP],
                    label="95% CI")
    _add_thresholds(ax)
    ax.set_xlabel(f"T1 relaxation time (µs)  [gate time = {gate_ns:.0f} ns]",
                  fontsize=11)
    ax.set_ylabel("QBER (%)", fontsize=11)
    ax.set_title(
        "QBER vs T1 (Amplitude Damping)\n"
        "Lindblad L = √γ |0⟩⟨1|,   γ = 1 − exp(−t_gate/T1)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.set_ylim(0, None)
    ax.invert_xaxis()   # short T1 = more noise = left side is "worst"

    # Annotate γ value at a few points
    for i in [0, steps // 2, steps]:
        ax.annotate(
            f"γ≈{gammas[i]:.4f}",
            xy=(t1_values_us[i], qbers[i]),
            xytext=(8, 6), textcoords="offset points",
            fontsize=7, color=_COLORS[NoiseModelType.AMPLITUDE_DAMP],
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved → {save_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────
# PLOT 4 — QBER vs T2  (phase damping sweep)
# ──────────────────────────────────────────────────────────────────────

def plot_qber_vs_t2(
    n_qubits:    int   = 400,
    gate_ns:     float = 50.0,
    t2_min_us:   float = 0.5,
    t2_max_us:   float = 100.0,
    steps:       int   = 12,
    save_path:   Optional[str] = "qkd_qber_vs_t2.png",
) -> plt.Figure:
    """
    Sweep T2 (dephasing time) from t2_min_us → t2_max_us µs.

    Physical interpretation
    ───────────────────────
    Short T2  → qubit loses phase coherence quickly
              → diagonal-basis (|+⟩/|−⟩) states become mixed
              → QBER rises, especially affecting the X-basis qubits
    Long  T2  → coherence maintained → low QBER

    Note: T2 ≤ 2·T1 always (Bloch sphere constraint).
    Phase damping does NOT change qubit populations (no bit-flips).
    This is why QBER from phase damping is lower than from amplitude damping
    at equal noise rates — an important educational comparison point.

    Lindblad operator: L = √λ |1⟩⟨1|  where λ = 1 − exp(−t/T2)
    """
    print("  Sweeping T2 (phase damping)...")
    t2_values_us = np.linspace(t2_min_us, t2_max_us, steps + 1)
    qbers, ci_l, ci_h = [], [], []

    for t2_us in t2_values_us:
        cfg = SimulationConfig(
            n_qubits=n_qubits,
            noise_enabled=True,
            noise_model=NoiseModelType.PHASE_DAMP,
            t2_ns=t2_us * 1000,
            gate_time_ns=gate_ns, seed=42,
        )
        r = run_simulation(cfg, verbose=False)
        qbers.append(r.qber_result.qber * 100)
        ci_l.append(r.qber_result.confidence_low  * 100)
        ci_h.append(r.qber_result.confidence_high * 100)
        print(f"    T2={t2_us:.1f}µs  λ={1 - np.exp(-gate_ns/(t2_us*1000)):.5f}"
              f"  →  QBER={qbers[-1]:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t2_values_us, qbers, "o-",
            color=_COLORS[NoiseModelType.PHASE_DAMP],
            linewidth=2, markersize=6, label="Simulated QBER", zorder=3)
    ax.fill_between(t2_values_us, ci_l, ci_h,
                    alpha=0.2, color=_COLORS[NoiseModelType.PHASE_DAMP],
                    label="95% CI")
    _add_thresholds(ax)
    ax.set_xlabel(f"T2 dephasing time (µs)  [gate time = {gate_ns:.0f} ns]",
                  fontsize=11)
    ax.set_ylabel("QBER (%)", fontsize=11)
    ax.set_title(
        "QBER vs T2 (Phase Damping / Dephasing)\n"
        "Lindblad L = √λ |1⟩⟨1|,   λ = 1 − exp(−t_gate/T2)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.set_ylim(0, None)
    ax.invert_xaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved → {save_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────
# PLOT 5 — FIBER LOSS ANALYSIS  (QBER + key rate vs distance)
# ──────────────────────────────────────────────────────────────────────

def plot_fiber_loss_analysis(
    n_qubits:   int  = 800,
    km_max:     float = 150.0,
    steps:      int  = 10,
    save_path:  Optional[str] = "qkd_fiber_loss.png",
) -> plt.Figure:
    """
    Sweep fibre distance 0 → km_max km and plot QBER + key generation rate.

    Key insight for students
    ────────────────────────
    Fibre loss does NOT increase QBER — it simply reduces the number of
    photons Bob receives.  So QBER stays near 0% while the key rate
    (bits / transmitted qubit) drops exponentially with distance.

    This is fundamentally different from gate noise (T1/T2/depolarizing)
    which directly corrupts bit values and raises QBER.

    Real-world context
    ──────────────────
    • Standard SMF-28 fibre: α = 0.2 dB/km at 1550 nm
    • Practical QKD limit without repeaters: ~100–150 km
    • Beyond this, key rate → 0 (too few photons reach Bob)
    • Solution: quantum repeaters (active research area)
    """
    print("  Sweeping fibre channel length...")
    import math

    distances  = np.linspace(0, km_max, steps + 1)
    qbers, ci_l, ci_h, key_rates, sift_rates = [], [], [], [], []

    for km in distances:
        cfg = SimulationConfig(
            n_qubits=n_qubits,
            noise_enabled=(km > 0),
            noise_model=NoiseModelType.FIBER_LOSS,
            channel_length_km=float(km), seed=42,
        )
        r = run_simulation(cfg, verbose=False)
        qbers.append(r.qber_result.qber * 100)
        ci_l.append(r.qber_result.confidence_low  * 100)
        ci_h.append(r.qber_result.confidence_high * 100)
        key_rates.append(r.key_generation_rate * 100)   # as %
        sift_rates.append(r.sifted_key_rate * 100)
        survive = 10 ** (-0.2 * km / 10) if km > 0 else 1.0
        print(f"    {km:5.0f} km  P_survive={survive:.3f}  "
              f"QBER={qbers[-1]:.1f}%  KeyRate={key_rates[-1]:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "BB84 Fibre Loss Analysis  (α = 0.2 dB/km, SMF-28)\n"
        "University of Ruhuna  ·  Dept. of Computer Engineering",
        fontsize=11, fontweight="bold", y=1.02,
    )

    # ── Panel 1: QBER vs distance ──────────────────────────────────────
    ax1.plot(distances, qbers, "o-",
             color=_COLORS[NoiseModelType.FIBER_LOSS],
             linewidth=2, markersize=6, label="Simulated QBER", zorder=3)
    ax1.fill_between(distances, ci_l, ci_h,
                     alpha=0.2, color=_COLORS[NoiseModelType.FIBER_LOSS],
                     label="95% CI")
    _add_thresholds(ax1)
    ax1.set_xlabel("Channel length (km)", fontsize=11)
    ax1.set_ylabel("QBER (%)", fontsize=11)
    ax1.set_title("QBER vs Distance\n"
                  "(fibre loss does NOT raise QBER)", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)
    ax1.set_ylim(0, max(15, max(qbers) + 3))

    # ── Panel 2: Key rate vs distance ─────────────────────────────────
    ax2.plot(distances, key_rates, "s-",
             color="#3498db", linewidth=2, markersize=6,
             label="Key generation rate", zorder=3)

    # Theoretical survival curve (just for reference)
    theo_survive = [10 ** (-0.2 * km / 10) * 50 if km > 0 else 50
                    for km in distances]
    ax2.plot(distances, theo_survive, "r--", linewidth=1.5,
             label="Theoretical (50% × P_survive)")

    ax2.set_xlabel("Channel length (km)", fontsize=11)
    ax2.set_ylabel("Key generation rate (%  of transmitted qubits)", fontsize=10)
    ax2.set_title("Key Rate vs Distance\n"
                  "(exponential decay — practical QKD limit ~100–150 km)",
                  fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.25)
    ax2.set_ylim(0, None)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved → {save_path}")
    plt.show()
    return fig
