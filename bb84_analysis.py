"""
bb84_analysis.py  ·  Security Analysis & Research Sweeps
══════════════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Provides all information-theoretic analysis, parameter sweeps, and
research-grade metric computation for the BB84 platform.

Exports
───────
compute_security_analysis()   — full SecurityAnalysis for one result
sweep_qber_vs_noise()         — QBER as function of noise level
sweep_skr_vs_distance()       — secret key rate vs fiber length
sweep_skr_vs_qber()           — SKR vs QBER for different formulas
sweep_pns_vulnerability()     — Eve's free info vs μ
sweep_decoy_effectiveness()   — compare SKR with/without decoy
full_parameter_sweep()        — 2D grid of SKR(noise, intercept)
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from bb84_config import SecurityAnalysis, SimulationConfig, SimulationResult
from bb84_postprocessing import (
    binary_entropy,
    holevo_bound_eve_intercept_resend,
    mutual_information_alice_bob,
    secret_key_rate_realistic,
    secret_key_rate_shor_preskill,
)
from bb84_noise import fiber_transmittance, max_secure_distance
from bb84_attacks import DecoyStateProtocol, p_multi, p_single, p_vacuum


# ══════════════════════════════════════════════════════════════════════
# SECURITY ANALYSIS FOR A SINGLE RUN
# ══════════════════════════════════════════════════════════════════════

def compute_security_analysis(
    result:   SimulationResult,
    f_EC:     float = 1.16,
    verbose:  bool  = True,
) -> SecurityAnalysis:
    """
    Compute full information-theoretic security metrics for a simulation result.

    Populates and returns a SecurityAnalysis object, and also stores it
    in result.security_analysis.

    Parameters
    ──────────
    result  : SimulationResult from run_simulation()
    f_EC    : error correction efficiency (default 1.16 for good codes)
    verbose : print the analysis table
    """
    qber = result.qber_result.qber
    h_e  = binary_entropy(qber)

    # ── Information-theoretic metrics ─────────────────────────────────
    iab  = mutual_information_alice_bob(qber)
    chi  = holevo_bound_eve_intercept_resend(
               result.eve_interception_rate, qber
           )

    # ── Secret key rates ──────────────────────────────────────────────
    skr_sp  = secret_key_rate_shor_preskill(qber)
    skr_rl  = secret_key_rate_realistic(qber, f_EC)

    # ── GLLP rate (if decoy enabled) ─────────────────────────────────
    skr_gllp = 0.0
    if result.config.decoy_state_enabled:
        dsp = DecoyStateProtocol(
            mu_signal=result.config.mu_signal,
            mu_decoy =result.config.mu_decoy,
            mu_vacuum=result.config.mu_vacuum,
        )
        eta = fiber_transmittance(result.config.channel_length_km)
        da  = dsp.full_analysis(eta, qber, f_EC, seed=result.config.seed)
        skr_gllp = da["skr_gllp"]

    # ── Eve's information estimate ────────────────────────────────────
    # Conservative upper bound on Eve's mutual information with raw key
    eve_info = result.eve_interception_rate * (1.0 - binary_entropy(0.25))

    # ── Privacy amplification compression ────────────────────────────
    pa_comp = 0.0
    final_bits = 0
    if result.post_processing and result.post_processing.privacy_amp_applied:
        m = result.post_processing.bits_after_privacy_amp
        n = result.post_processing.bits_after_ecc or result.key_length
        pa_comp    = 1.0 - (m / n if n > 0 else 0)
        final_bits = m
    else:
        final_bits = result.key_length

    analysis = SecurityAnalysis(
        binary_entropy_qber=h_e,
        mutual_info_alice_bob=iab,
        holevo_bound_eve=chi,
        secret_key_rate_ideal=skr_sp,
        secret_key_rate_realistic=skr_rl,
        secret_key_rate_gllp=skr_gllp,
        estimated_eve_info=eve_info,
        privacy_amp_compression=pa_comp,
        final_secret_bits=final_bits,
    )

    result.security_analysis = analysis

    if verbose:
        _print_security_table(analysis, result)

    return analysis


def _print_security_table(analysis: SecurityAnalysis, result: SimulationResult):
    line = "─" * 62
    print(f"\n{line}")
    print("  SECURITY ANALYSIS")
    print(line)
    e = result.qber_result.qber
    print(f"  QBER                     : {e * 100:.2f} %")
    print(f"  H(QBER)                  : {analysis.binary_entropy_qber:.4f} bits")
    print(f"  I(A;B)                   : {analysis.mutual_info_alice_bob:.4f} bits/symbol")
    print(f"  χ(Eve) Holevo bound      : {analysis.holevo_bound_eve:.4f} bits")
    print(f"  SKR — Shor-Preskill      : {analysis.secret_key_rate_ideal:.4f} bits/qubit")
    print(f"  SKR — Realistic (f=1.16) : {analysis.secret_key_rate_realistic:.4f} bits/qubit")
    if result.config.decoy_state_enabled:
        print(f"  SKR — GLLP (decoy)       : {analysis.secret_key_rate_gllp:.4f} bits/pulse")
    print(f"  Eve's info (upper bound) : {analysis.estimated_eve_info:.4f} bits")
    if analysis.privacy_amp_compression > 0:
        print(f"  PA compression ratio     : {analysis.privacy_amp_compression * 100:.1f} %")
    print(f"  Final secret key         : {analysis.final_secret_bits} bits")
    print(line)


# ══════════════════════════════════════════════════════════════════════
# SWEEP 1 — QBER vs NOISE LEVEL
# ══════════════════════════════════════════════════════════════════════

def sweep_qber_vs_noise(
    noise_model:  str   = "depolarizing",
    n_steps:      int   = 12,
    n_qubits:     int   = 500,
    seed:         int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep noise level from near-zero to high and record QBER.

    Returns
    ───────
    noise_vals  : array of noise parameter values
    qbers       : simulated QBER at each level
    ci_low      : 95 % CI lower bound
    ci_high     : 95 % CI upper bound
    """
    from bb84_runner import run_simulation

    if noise_model == "depolarizing":
        noise_vals = np.linspace(0, 0.15, n_steps)
        param_key  = "depolar_prob"
    elif noise_model == "amplitude_damp":
        noise_vals = np.linspace(1e3, 1e5, n_steps)   # T1 in ns
        param_key  = "t1_ns"
    elif noise_model == "phase_damp":
        noise_vals = np.linspace(1e3, 1e5, n_steps)   # T2 in ns
        param_key  = "t2_ns"
    elif noise_model == "fiber_loss":
        noise_vals = np.linspace(0, 100, n_steps)      # km
        param_key  = "channel_length_km"
    else:
        raise ValueError(f"Unknown noise_model: {noise_model}")

    qbers, ci_lo, ci_hi = [], [], []

    for val in noise_vals:
        kwargs = dict(
            n_qubits      = n_qubits,
            eve_present   = False,
            noise_enabled = True,
            noise_model   = noise_model,
            seed          = seed,
            label         = f"{noise_model}={val:.2f}",
        )
        kwargs[param_key] = float(val)
        cfg = SimulationConfig(**kwargs)
        r   = run_simulation(cfg, verbose=False)
        qbers.append(r.qber_result.qber * 100)
        ci_lo.append(r.qber_result.confidence_low  * 100)
        ci_hi.append(r.qber_result.confidence_high * 100)

    return (np.array(noise_vals), np.array(qbers),
            np.array(ci_lo),      np.array(ci_hi))


# ══════════════════════════════════════════════════════════════════════
# SWEEP 2 — SECRET KEY RATE vs FIBER DISTANCE
# ══════════════════════════════════════════════════════════════════════

def sweep_skr_vs_distance(
    distances_km: Optional[List[float]] = None,
    mu:           float = 0.5,
    f_EC:         float = 1.16,
    n_qubits:     int   = 500,
    seed:         int   = 42,
) -> dict:
    """
    Compute theoretical and simulated SKR vs fiber distance.

    Returns
    ───────
    dict with keys:
      distances, eta, skr_shor_preskill, skr_realistic,
      skr_gllp (with decoy), qbers, max_distance_km
    """
    from bb84_runner import run_simulation

    if distances_km is None:
        distances_km = list(np.linspace(0, 120, 25))

    results = {
        "distances":           np.array(distances_km),
        "eta":                 [],
        "skr_shor_preskill":   [],
        "skr_realistic":       [],
        "skr_gllp":            [],
        "qbers":               [],
    }

    dsp = DecoyStateProtocol(mu_signal=mu, mu_decoy=mu * 0.2, mu_vacuum=0.0)

    for L in distances_km:
        eta   = fiber_transmittance(L)
        Y0    = 1.7e-6
        Q_mu  = max(1 - math.exp(-mu * eta), Y0)
        E_mu  = min(0.5 * Y0 / Q_mu + 0.005, 0.5)  # ~0.5% background QBER

        skr_sp = Q_mu * secret_key_rate_shor_preskill(E_mu)
        skr_rl = Q_mu * secret_key_rate_realistic(E_mu, f_EC)

        da  = dsp.full_analysis(eta, E_mu, f_EC)
        skr_g = da["skr_gllp"]

        results["eta"].append(eta)
        results["qbers"].append(E_mu * 100)
        results["skr_shor_preskill"].append(max(0.0, skr_sp))
        results["skr_realistic"].append(max(0.0, skr_rl))
        results["skr_gllp"].append(max(0.0, skr_g))

    results["max_distance_km"] = max_secure_distance(mu)
    for k in ("eta", "qbers", "skr_shor_preskill", "skr_realistic", "skr_gllp"):
        results[k] = np.array(results[k])

    return results


# ══════════════════════════════════════════════════════════════════════
# SWEEP 3 — SKR vs QBER  (formula comparison)
# ══════════════════════════════════════════════════════════════════════

def sweep_skr_vs_qber(
    qber_range: Optional[np.ndarray] = None,
    f_EC_values: Optional[List[float]] = None,
) -> dict:
    """
    Analytical SKR vs QBER for different EC efficiencies.

    No simulation required — pure information theory.

    Returns
    ───────
    dict with keys:
      qbers, h_qber, skr_shor_preskill,
      skr_realistic_{f} for each f in f_EC_values
    """
    if qber_range is None:
        qber_range = np.linspace(0.001, 0.20, 200)
    if f_EC_values is None:
        f_EC_values = [1.0, 1.16, 1.3, 1.5]

    h_vals = np.array([binary_entropy(e) for e in qber_range])
    skr_sp = np.array([secret_key_rate_shor_preskill(e) for e in qber_range])

    result = {
        "qbers":              qber_range * 100,
        "h_qber":             h_vals,
        "skr_shor_preskill":  skr_sp,
    }

    for f in f_EC_values:
        key = f"skr_f{f:.2f}".replace(".", "p")
        result[key] = np.array([secret_key_rate_realistic(e, f) for e in qber_range])

    result["qber_threshold_sp"] = _find_threshold(qber_range, skr_sp)
    for f in f_EC_values:
        k = f"skr_f{f:.2f}".replace(".", "p")
        result[f"qber_threshold_f{f:.2f}".replace(".", "p")] = (
            _find_threshold(qber_range, result[k])
        )

    return result


# ══════════════════════════════════════════════════════════════════════
# SWEEP 4 — PNS VULNERABILITY vs μ
# ══════════════════════════════════════════════════════════════════════

def sweep_pns_vulnerability(
    mu_range: Optional[np.ndarray] = None,
) -> dict:
    """
    Analytical PNS vulnerability vs mean photon number μ.

    Shows Eve's free-information fraction (multi-photon rate) and
    the safe single-photon fraction as functions of μ.

    Returns
    ───────
    dict with keys: mu_vals, p_vacuum, p_single, p_multi,
                    eve_free_info, safe_fraction
    """
    if mu_range is None:
        mu_range = np.linspace(0.01, 1.5, 150)

    return {
        "mu_vals":       mu_range,
        "p_vacuum":      np.array([p_vacuum(m) for m in mu_range]),
        "p_single":      np.array([p_single(m) for m in mu_range]),
        "p_multi":       np.array([p_multi(m) for m in mu_range]),
        "eve_free_info": np.array([p_multi(m) for m in mu_range]),
        "safe_fraction": np.array([p_single(m) / (1 - p_vacuum(m) + 1e-12)
                                   for m in mu_range]),
        "optimal_mu":    _find_optimal_mu(mu_range),
    }


# ══════════════════════════════════════════════════════════════════════
# SWEEP 5 — DECOY STATE EFFECTIVENESS
# ══════════════════════════════════════════════════════════════════════

def sweep_decoy_effectiveness(
    distances_km:  Optional[List[float]] = None,
    mu_signal:     float = 0.5,
    mu_decoy:      float = 0.1,
    f_EC:          float = 1.16,
) -> dict:
    """
    Compare SKR with and without decoy state protocol over distance.

    Returns
    ───────
    dict with keys:
      distances, skr_no_decoy, skr_with_decoy, gain_factor
    """
    if distances_km is None:
        distances_km = list(np.linspace(0, 120, 30))

    dsp = DecoyStateProtocol(mu_signal, mu_decoy, 0.0)

    skr_no_decoy   = []
    skr_with_decoy = []

    for L in distances_km:
        eta  = fiber_transmittance(L)
        Y0   = 1.7e-6
        Q_mu = max(1 - math.exp(-mu_signal * eta), Y0)
        E_mu = min(0.5 * Y0 / Q_mu + 0.005, 0.5)

        # Without decoy: cannot bound Q1 tightly → use WCP SKR formula
        skr_nd = max(0.0, Q_mu * secret_key_rate_realistic(E_mu, f_EC))

        # With decoy: GLLP formula using bounded Q1, e1
        da     = dsp.full_analysis(eta, E_mu, f_EC)
        skr_d  = da["skr_gllp"]

        skr_no_decoy.append(skr_nd)
        skr_with_decoy.append(skr_d)

    nd  = np.array(skr_no_decoy)
    wd  = np.array(skr_with_decoy)
    gain = np.where(nd > 1e-12, wd / nd, 0.0)

    return {
        "distances":       np.array(distances_km),
        "skr_no_decoy":    nd,
        "skr_with_decoy":  wd,
        "gain_factor":     gain,
    }


# ══════════════════════════════════════════════════════════════════════
# SWEEP 6 — 2D PARAMETER HEATMAP  (novel research experiment)
# ══════════════════════════════════════════════════════════════════════

def full_parameter_sweep(
    noise_levels:       Optional[List[float]] = None,
    intercept_probs:    Optional[List[float]] = None,
    n_qubits:           int   = 300,
    f_EC:               float = 1.16,
    seed:               int   = 42,
) -> dict:
    """
    2D sweep: SKR as a function of (noise level, intercept probability).

    Produces data for a heatmap showing the joint security parameter space.
    This is the core 'novel research' experiment — it maps the boundary
    between secure and insecure regions in the (noise, Eve) plane.

    Returns
    ───────
    dict with keys:
      noise_levels, intercept_probs,
      skr_grid  : 2D array shape (n_noise, n_intercept)
      qber_grid : 2D array shape (n_noise, n_intercept)
      secure_boundary : list of (noise, intercept) at SKR=0
    """
    from bb84_runner import run_simulation

    if noise_levels is None:
        noise_levels = list(np.linspace(0.0, 0.10, 8))
    if intercept_probs is None:
        intercept_probs = list(np.linspace(0.0, 1.0, 8))

    n_n = len(noise_levels)
    n_p = len(intercept_probs)
    skr_grid  = np.zeros((n_n, n_p))
    qber_grid = np.zeros((n_n, n_p))

    total = n_n * n_p
    done  = 0

    print(f"\n  [Analysis] 2D parameter sweep  ({total} simulations)...")

    for i, dp in enumerate(noise_levels):
        for j, p_eve in enumerate(intercept_probs):
            cfg = SimulationConfig(
                n_qubits       = n_qubits,
                eve_present    = p_eve > 0,
                eve_intercept_prob = p_eve,
                noise_enabled  = dp > 0,
                depolar_prob   = dp,
                seed           = seed,
                label          = f"dp={dp:.3f}_eve={p_eve:.2f}",
            )
            r = run_simulation(cfg, verbose=False)
            qber = r.qber_result.qber
            skr  = secret_key_rate_realistic(qber, f_EC)

            skr_grid[i, j]  = max(0.0, skr)
            qber_grid[i, j] = qber * 100

            done += 1
            if done % 8 == 0 or done == total:
                print(f"        ↳ {done}/{total} done", end="\r")

    print(f"        ↳ {total}/{total} done ✓")

    # Find the SKR=0 boundary (security edge)
    boundary = []
    for i, dp in enumerate(noise_levels):
        for j, p_eve in enumerate(intercept_probs):
            if j > 0 and skr_grid[i, j] == 0 and skr_grid[i, j - 1] > 0:
                boundary.append((dp, intercept_probs[j - 1]))
                break

    return {
        "noise_levels":    np.array(noise_levels),
        "intercept_probs": np.array(intercept_probs),
        "skr_grid":        skr_grid,
        "qber_grid":       qber_grid,
        "secure_boundary": boundary,
    }


# ══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════

def _find_threshold(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """Find x where y crosses 0 (linear interpolation)."""
    for i in range(len(y_arr) - 1):
        if y_arr[i] > 0 >= y_arr[i + 1]:
            dx = x_arr[i + 1] - x_arr[i]
            dy = y_arr[i + 1] - y_arr[i]
            return float(x_arr[i] - y_arr[i] * dx / dy) if dy != 0 else x_arr[i]
    return float(x_arr[-1])


def _find_optimal_mu(mu_range: np.ndarray) -> float:
    """
    Rough estimate of optimal μ: maximise single-photon fraction.
    p_single(μ) = μ·e^(−μ) is maximised at μ = 1, but in practice
    we want μ < 1 to reduce multi-photon events.
    """
    singles = np.array([p_single(m) / (1 - p_vacuum(m) + 1e-12)
                        for m in mu_range])
    return float(mu_range[np.argmax(singles)])
