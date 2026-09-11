"""
bb84_zne2d.py
=============
Two-axis (decoupled) Zero-Noise Extrapolation for the Combined T1+T2
noise model -- closes the RQ5 gap reported in the paper's ZNE-QBER
section (Section 3, "scope boundary"), where scaling gamma and lambda
by a SINGLE shared factor f left the no-Eve intercept at a
non-negligible ~2.7 pp instead of ~0.

Root cause
----------
`thermal_relaxation_error` (Combined T1+T2) is a genuine TWO-parameter
channel: gamma is driven by T1, lambda is driven by T2, and they are
coupled only by the physical constraint T2 <= 2*T1. Scaling both by a
single shared f (as bb84_phase4_experiments.ipynb's E11d cell does)
asks a 1-D line through (QBER at f=1, f=1.5, f=2.0, f=2.5) to
extrapolate to a point, (f=0), that is only guaranteed noise-free when
BOTH axes are independently zero. If the true surface QBER(f1, f2) is
not exactly linear along the diagonal f1=f2=f, a 1-D fit along that
diagonal picks up curvature as bias.

Fix
---
Scale gamma by f1 and lambda by f2 INDEPENDENTLY, sweep the full
(f1 x f2) grid, and fit a weighted PLANE (not a line) to
QBER(f1, f2) ~= a + b*f1 + c*f2, evaluated at (f1, f2) = (0, 0). This
is the direct generalisation of `bb84_zne.linear_extrapolate` to two
independent physical noise axes, and requires no new physical
capability beyond what `bb84_zne.py` already assumes (independently
dialling one continuous noise parameter).

University of Ruhuna - Dept. of Computer Engineering
MIT Licence - see LICENSE
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from bb84_config import SimulationConfig
from bb84_noise import NoiseModelType
from bb84_runner import run_simulation
from bb84_zne import scale_amplitude_damping, scale_phase_damping


# ──────────────────────────────────────────────────────────────────────
# 1. DECOUPLED SCALED CONFIG BUILDER
# ──────────────────────────────────────────────────────────────────────

def build_combined_scaled_config_2d(
    f1: float,
    f2: float,
    n_qubits: int,
    seed: int,
    p_eve: float,
    base_t1_ns: float = 500.0,
    base_t2_ns: float = 200.0,
    gate_time_ns: float = 50.0,
    sample_fraction: float = 0.15,
) -> SimulationConfig:
    """
    Build a Combined-T1+T2 SimulationConfig with gamma scaled by f1 and
    lambda scaled by f2 INDEPENDENTLY (contrast with the notebook's
    original build_combined_scaled_config(), which used one shared f
    for both -- that single-axis coupling is exactly what produced the
    non-zero no-Eve intercept reported as RQ5's scope boundary).
    """
    t1 = scale_amplitude_damping(base_t1_ns, gate_time_ns, f1)
    t2 = scale_phase_damping(base_t2_ns, gate_time_ns, f2)
    return SimulationConfig(
        n_qubits=n_qubits,
        seed=seed,
        noise_model=NoiseModelType.COMBINED,
        t1_ns=t1,
        t2_ns=t2,
        gate_time_ns=gate_time_ns,
        eve_present=(p_eve > 0),
        eve_intercept_prob=p_eve,
        sample_fraction=sample_fraction,
        label=f"2D-ZNE f1={f1:.2f} f2={f2:.2f} pEve={p_eve:.1f}",
    )


# ──────────────────────────────────────────────────────────────────────
# 2. PLANAR (2-D) EXTRAPOLATION
# ──────────────────────────────────────────────────────────────────────

def planar_extrapolate(
    f1s: List[float],
    f2s: List[float],
    qbers: List[float],
    weights: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """
    Weighted least-squares plane fit QBER(f1,f2) ~= a + b*f1 + c*f2,
    evaluated at (f1,f2)=(0,0). Direct 2-variable generalisation of
    bb84_zne.linear_extrapolate's 1-variable weighted LS fit.

    weights, if given, should be inverse-variance weights (e.g.
    1 / wilson_ci_halfwidth**2), matching bb84_zne's convention.
    Returns (intercept_a, slope_f1_b, slope_f2_c).
    """
    f1 = np.asarray(f1s, dtype=float)
    f2 = np.asarray(f2s, dtype=float)
    q = np.asarray(qbers, dtype=float)
    if weights is None:
        w = np.ones_like(f1)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.where(w <= 0, 1e-6, w)

    X = np.column_stack([np.ones_like(f1), f1, f2])
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    qw = q * sw

    coeffs, *_ = np.linalg.lstsq(Xw, qw, rcond=None)
    a, b, c = (float(v) for v in coeffs)
    return a, b, c


def zne_estimate_planar(f1s, f2s, qbers, weights=None) -> float:
    a, _, _ = planar_extrapolate(f1s, f2s, qbers, weights)
    return max(0.0, a)


def bilinear_extrapolate(
    f1s: List[float],
    f2s: List[float],
    qbers: List[float],
    weights: Optional[List[float]] = None,
) -> Tuple[float, float, float, float]:
    """
    Weighted least-squares fit QBER(f1,f2) ~= a + b*f1 + c*f2 + d*f1*f2,
    evaluated at (0,0). The added f1*f2 cross term captures the
    T1/T2 INTERACTION that thermal_relaxation_error's Kraus map has and
    a pure plane (planar_extrapolate) cannot represent -- empirically,
    this is what actually drives the no-Eve intercept back toward zero
    (see the notebook's RQ5-2D section): a plane alone leaves a
    residual ~2pp bias, adding one cross-coefficient removes nearly all
    of it. Needs a genuine 2-D grid (>=2 distinct values on each axis,
    >=4 points total) to identify all 4 coefficients; falls back to
    planar_extrapolate if the grid is too sparse.

    Returns (a, b, c, d).
    """
    f1 = np.asarray(f1s, dtype=float)
    f2 = np.asarray(f2s, dtype=float)
    q = np.asarray(qbers, dtype=float)
    if weights is None:
        w = np.ones_like(f1)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.where(w <= 0, 1e-6, w)

    if len(set(f1.tolist())) < 2 or len(set(f2.tolist())) < 2 or len(f1) < 4:
        a, b, c = planar_extrapolate(f1s, f2s, qbers, weights)
        return a, b, c, 0.0

    X = np.column_stack([np.ones_like(f1), f1, f2, f1 * f2])
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    qw = q * sw
    coeffs, *_ = np.linalg.lstsq(Xw, qw, rcond=None)
    a, b, c, d = (float(v) for v in coeffs)
    return a, b, c, d


def zne_estimate_bilinear(f1s, f2s, qbers, weights=None) -> float:
    a, _, _, _ = bilinear_extrapolate(f1s, f2s, qbers, weights)
    return max(0.0, a)


# ──────────────────────────────────────────────────────────────────────
# 3. FULL 2-D SWEEP COLLECTOR
# ──────────────────────────────────────────────────────────────────────

def run_zne_sweep_2d(
    f1_scales: List[float],
    f2_scales: List[float],
    p_eve_grid: List[float],
    seeds: List[int],
    n_qubits: int = 2000,
    base_t1_ns: float = 500.0,
    base_t2_ns: float = 200.0,
    gate_time_ns: float = 50.0,
    sample_fraction: float = 0.15,
) -> Dict:
    """
    Runs the full (f1 x f2) grid x p_eve x seeds, computing raw QBER +
    Wilson CI at every point, then a weighted planar fit per p_eve.

    Returns a dict:
        raw[p_eve][(f1,f2)]      = list of (qber_pct, ci_low, ci_high) per seed
        zne_planar[p_eve]        = extrapolated QBER (%) at (f1,f2)=(0,0)
        plane_coeffs[p_eve]      = (a, b, c) fitted plane coefficients
        qber_at_f1f2_1[p_eve]    = mean raw QBER (%) at (f1,f2)=(1,1), i.e.
                                    what the ORIGINAL single-axis E11d cell
                                    would have reported as its f=1 point
    """
    raw: Dict[float, Dict[Tuple[float, float], list]] = {}

    for p_eve in p_eve_grid:
        raw[p_eve] = {}
        for f1 in f1_scales:
            for f2 in f2_scales:
                vals = []
                for seed in seeds:
                    cfg = build_combined_scaled_config_2d(
                        f1, f2, n_qubits, seed, p_eve,
                        base_t1_ns, base_t2_ns, gate_time_ns, sample_fraction,
                    )
                    r = run_simulation(cfg, verbose=False)
                    qr = r.qber_result
                    vals.append((qr.qber * 100, qr.confidence_low * 100, qr.confidence_high * 100))
                raw[p_eve][(f1, f2)] = vals

    zne_planar: Dict[float, float] = {}
    zne_bilinear: Dict[float, float] = {}
    plane_coeffs: Dict[float, Tuple[float, float, float]] = {}
    bilinear_coeffs: Dict[float, Tuple[float, float, float, float]] = {}
    qber_at_11: Dict[float, float] = {}

    for p_eve in p_eve_grid:
        f1_list, f2_list, mean_list, w_list = [], [], [], []
        for (f1, f2), vals in raw[p_eve].items():
            qbers_here = [v[0] for v in vals]
            halfwidths = [max(1e-3, (v[2] - v[1]) / 2) for v in vals]
            f1_list.append(f1)
            f2_list.append(f2)
            mean_list.append(float(np.mean(qbers_here)))
            w_list.append(1.0 / (float(np.mean(halfwidths)) ** 2))

        a, b, c = planar_extrapolate(f1_list, f2_list, mean_list, w_list)
        zne_planar[p_eve] = max(0.0, a)
        plane_coeffs[p_eve] = (a, b, c)

        a2, b2, c2, d2 = bilinear_extrapolate(f1_list, f2_list, mean_list, w_list)
        zne_bilinear[p_eve] = max(0.0, a2)
        bilinear_coeffs[p_eve] = (a2, b2, c2, d2)

        if (1.0, 1.0) in raw[p_eve]:
            qber_at_11[p_eve] = float(np.mean([v[0] for v in raw[p_eve][(1.0, 1.0)]]))

    return {
        "raw": raw,
        "zne_planar": zne_planar,
        "zne_bilinear": zne_bilinear,
        "plane_coeffs": plane_coeffs,
        "bilinear_coeffs": bilinear_coeffs,
        "qber_at_f1f2_1": qber_at_11,
        "f1_scales": f1_scales,
        "f2_scales": f2_scales,
        "p_eve_grid": p_eve_grid,
    }
