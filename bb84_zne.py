"""
bb84_zne.py
===========
Phase 5: Zero-Noise Extrapolation (ZNE) utilities for QBER correction.

Provides:
    scale_depolar(p, f_scale)              -> scaled depolar_prob
    scale_amplitude_damping(t1_ns, gate_time_ns, f_scale) -> scaled t1_ns
    scale_phase_damping(t2_ns, gate_time_ns, f_scale)     -> scaled t2_ns
    linear_extrapolate(f_scales, qbers, weights=None)     -> (intercept, slope)
    exponential_extrapolate(f_scales, qbers)              -> (A, B, c) fit
    run_zne_sweep(...)                                    -> full E11 data collector

University of Ruhuna - Dept. of Computer Engineering
MIT Licence - see LICENSE
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
from scipy.optimize import curve_fit

from bb84_config import SimulationConfig
from bb84_noise import NoiseModelType
from bb84_runner import run_simulation


# ──────────────────────────────────────────────────────────────────────
# 1. NOISE-SCALING / INVERSION  (Section 4.1 of the design doc)
# ──────────────────────────────────────────────────────────────────────

def scale_depolar(p: float, f_scale: float) -> float:
    """Depolarising probability is already linear in the Kraus weight."""
    return min(max(p * f_scale, 0.0), 1.0)


def scale_amplitude_damping(t1_ns: float, gate_time_ns: float, f_scale: float) -> float:
    """
    Scale the *derived* damping probability gamma, not T1 directly
    (gamma is nonlinear in T1). Returns an effective t1_ns that
    reproduces gamma_scaled = gamma * f_scale.
    """
    gamma = 1.0 - math.exp(-gate_time_ns / t1_ns)
    gamma_scaled = min(max(gamma * f_scale, 0.0), 0.999999)
    if gamma_scaled <= 0.0:
        return 1e12  # effectively infinite T1 (no damping)
    return -gate_time_ns / math.log(1.0 - gamma_scaled)


def scale_phase_damping(t2_ns: float, gate_time_ns: float, f_scale: float) -> float:
    """Same inversion logic as amplitude damping, applied to lambda."""
    lam = 1.0 - math.exp(-gate_time_ns / t2_ns)
    lam_scaled = min(max(lam * f_scale, 0.0), 0.999999)
    if lam_scaled <= 0.0:
        return 1e12
    return -gate_time_ns / math.log(1.0 - lam_scaled)


# ──────────────────────────────────────────────────────────────────────
# 2. EXTRAPOLATION  (Section 3.2 / 3.4)
# ──────────────────────────────────────────────────────────────────────

def linear_extrapolate(
    f_scales: List[float],
    qbers:    List[float],
    weights:  Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Weighted least-squares fit QBER(f) ~ a + b*f, evaluated at f=0.

    weights, if given, should be inverse-variance weights (e.g.
    1 / wilson_ci_halfwidth**2). Returns (intercept_a, slope_b).
    """
    f = np.asarray(f_scales, dtype=float)
    q = np.asarray(qbers, dtype=float)
    if weights is None:
        w = np.ones_like(f)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.where(w <= 0, 1e-6, w)  # guard against zero-width CI

    # Weighted linear regression: minimise sum(w * (q - a - b*f)^2)
    W  = np.sum(w)
    Wf = np.sum(w * f)
    Wq = np.sum(w * q)
    Wff = np.sum(w * f * f)
    Wfq = np.sum(w * f * q)

    denom = (W * Wff - Wf ** 2)
    if abs(denom) < 1e-12:
        # degenerate (all f identical) — fall back to mean
        return float(np.average(q, weights=w)), 0.0

    b = (W * Wfq - Wf * Wq) / denom
    a = (Wq - b * Wf) / W
    return float(a), float(b)


# ── REPLACE exponential_extrapolate and zne_estimate_exponential again ──

def exponential_extrapolate(
    f_scales: List[float],
    qbers:    List[float],
) -> Dict[str, float]:
    """
    Fit QBER(f) ~ A - B*exp(-c*f). Returns a dict with:
        A, B, c              : fitted parameters
        estimate_raw          : A - B, BEFORE clamping to >=0
        estimate              : max(0, A - B)
        se_estimate            : propagated 1-sigma std error on (A-B),
                                  via the covariance matrix (NaN if unavailable)
        converged              : bool — see criteria below

    converged=True requires ALL of:
      (1) curve_fit exits without exception,
      (2) c is not pinned at its lower bound (structural degeneracy),
      (3) the covariance matrix is finite and well-conditioned,
      (4) se_estimate / max(1.0, |estimate_raw|) < 0.5 — i.e. the
          extrapolated intercept is not swamped by its own uncertainty.

    Criterion (4) is what the previous version was missing: with only
    4 points and 3 free parameters, curve_fit can report a "successful"
    fit whose intercept estimate has enormous uncertainty (A and B both
    poorly constrained, but A-B happening to land near zero). That is
    NOT a converged estimate in any useful sense, even though the
    optimizer did not raise an exception.
    """
    f = np.asarray(f_scales, dtype=float)
    q = np.asarray(qbers, dtype=float)

    def model(x, A, B, c):
        return A - B * np.exp(-c * x)

    a_lin, b_lin = linear_extrapolate(list(f_scales), list(qbers))
    fail = dict(A=a_lin, B=0.0, c=0.0, estimate_raw=a_lin,
                estimate=max(0.0, a_lin), se_estimate=float('nan'),
                converged=False)

    try:
        p0 = [max(q.max(), a_lin) + 1.0, max(q.max() - q.min(), 0.5), 1.0]
        bounds = ([-50, -50, 1e-4], [100, 100, 10])
        popt, pcov = curve_fit(model, f, q, p0=p0, bounds=bounds, maxfev=10000)
        A, B, c = (float(v) for v in popt)
        estimate_raw = A - B

        if c <= 1.5e-4 or not np.all(np.isfinite(pcov)):
            return fail

        # Propagate variance of (A - B): Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B)
        var_A, var_B, cov_AB = pcov[0, 0], pcov[1, 1], pcov[0, 1]
        var_est = var_A + var_B - 2 * cov_AB
        if var_est < 0 or not np.isfinite(var_est):
            return fail
        se_estimate = float(np.sqrt(var_est))

        relative_uncertainty = se_estimate / max(1.0, abs(estimate_raw))
        converged = relative_uncertainty < 0.5

        return dict(A=A, B=B, c=c, estimate_raw=estimate_raw,
                     estimate=max(0.0, estimate_raw),
                     se_estimate=se_estimate, converged=converged)
    except Exception:
        return fail


def zne_estimate_exponential(f_scales, qbers) -> Tuple[float, bool, float]:
    """Returns (clamped estimate, converged flag, raw pre-clamp estimate)."""
    r = exponential_extrapolate(f_scales, qbers)
    return r['estimate'], r['converged'], r['estimate_raw']


def zne_estimate_linear(f_scales, qbers, weights=None) -> float:
    a, _ = linear_extrapolate(f_scales, qbers, weights)
    return max(0.0, a)



# ──────────────────────────────────────────────────────────────────────
# 3. NOISE-SCALED SIMULATION CONFIG BUILDER
# ──────────────────────────────────────────────────────────────────────

def build_scaled_config(
    base_noise_model: str,
    f_scale: float,
    n_qubits: int,
    seed: int,
    p_eve: float,
    base_depolar_prob: float = 0.05,
    base_t1_ns: float = 500.0,
    base_t2_ns: float = 200.0,
    gate_time_ns: float = 50.0,
    sample_fraction: float = 0.15,
) -> SimulationConfig:
    """
    Build a SimulationConfig with noise scaled by f_scale, for the
    requested base_noise_model ('depolarizing' | 'amplitude_damping' |
    'phase_damping'). Fibre loss is intentionally not supported (ZNE
    does not apply — see design doc Section 4.1).
    """
    kwargs = dict(
        n_qubits=n_qubits, seed=seed,
        eve_present=(p_eve > 0), eve_intercept_prob=p_eve,
        sample_fraction=sample_fraction,
        label=f"ZNE f={f_scale:.2f} pEve={p_eve:.1f}",
    )

    if base_noise_model == NoiseModelType.DEPOLARIZING:
        kwargs["noise_model"]  = NoiseModelType.DEPOLARIZING
        kwargs["depolar_prob"] = scale_depolar(base_depolar_prob, f_scale)

    elif base_noise_model == NoiseModelType.AMPLITUDE_DAMPING:
        kwargs["noise_model"]  = NoiseModelType.AMPLITUDE_DAMPING
        kwargs["t1_ns"]        = scale_amplitude_damping(base_t1_ns, gate_time_ns, f_scale)
        # T2 is unused by amplitude damping's Kraus operators (same
        # convention qkd_app.py already uses) - pin it equal to T1 so
        # the T2 <= 2*T1 Bloch-sphere check in SimulationConfig can
        # never fail here, no matter how far f_scale pushes T1 down.
        kwargs["t2_ns"]        = kwargs["t1_ns"]
        kwargs["gate_time_ns"] = gate_time_ns

    elif base_noise_model == NoiseModelType.PHASE_DAMPING:
        kwargs["noise_model"]  = NoiseModelType.PHASE_DAMPING
        kwargs["t2_ns"]        = scale_phase_damping(base_t2_ns, gate_time_ns, f_scale)
        # T1 is unused by phase damping's Kraus operators - pin it equal
        # to T2 (same reasoning as above) instead of the old hardcoded
        # 10_000, which broke as soon as scaled T2 exceeded 20_000 ns.
        kwargs["t1_ns"]        = kwargs["t2_ns"]
        kwargs["gate_time_ns"] = gate_time_ns

    else:
        raise ValueError(f"ZNE not supported for noise_model={base_noise_model!r}")

    return SimulationConfig(**kwargs)


#Utility functions
def bootstrap_zne_intercept(
    per_seed_qbers: Dict[float, List[float]],   # f_scale -> list of per-seed QBER%
    f_scales: List[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Resample seeds with replacement, refit the linear ZNE intercept each
    time. Returns (mean, ci_low_2.5pct, ci_high_97.5pct) across n_boot
    resamples. per_seed_qbers[f] must have the same length (n_seeds) for
    every f_scale, and resampling draws the SAME seed indices across all
    f_scale values in a given draw (paired resampling), so trajectories
    stay internally consistent.
    """
    rng = np.random.default_rng(seed)
    n_seeds = len(next(iter(per_seed_qbers.values())))
    intercepts = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        means = [float(np.mean([per_seed_qbers[f][i] for i in idx])) for f in f_scales]
        a, _ = linear_extrapolate(f_scales, means)
        intercepts.append(max(0.0, a))
    intercepts = np.array(intercepts)
    return (float(np.mean(intercepts)),
            float(np.percentile(intercepts, 2.5)),
            float(np.percentile(intercepts, 97.5)))



def quadratic_extrapolate(
    f_scales: List[float],
    qbers:    List[float],
    weights:  Optional[List[float]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Weighted 2nd-order polynomial fit QBER(f) ~ a + b*f + c*f^2,
    evaluated at f=0 (i.e. the fit intercept, a).

    Numerically stable even with few points (unlike the 3-parameter
    exponential fit) since this is ordinary weighted linear regression
    in the coefficients, not a nonlinear optimisation.

    With only 4 f_scale points and 3 free coefficients, note this
    fit has just 1 residual degree of freedom — treat it as a
    curvature diagnostic, not a high-confidence point estimate.

    Returns (intercept_a0, full_coeffs) where full_coeffs is
    [c, b, a] as returned by np.polyfit (highest degree first).
    """
    f = np.asarray(f_scales, dtype=float)
    q = np.asarray(qbers, dtype=float)
    w = np.ones_like(f) if weights is None else np.asarray(weights, dtype=float)
    w = np.where(w <= 0, 1e-6, w)

    if len(f) < 3:
        # underdetermined for a quadratic — fall back to linear
        a, _ = linear_extrapolate(list(f_scales), list(qbers), weights)
        return a, np.array([0.0, 0.0, a])

    coeffs = np.polyfit(f, q, deg=2, w=np.sqrt(w))
    a0 = float(np.polyval(coeffs, 0.0))
    return max(0.0, a0), coeffs


def zne_estimate_quadratic(f_scales, qbers, weights=None) -> float:
    a0, _ = quadratic_extrapolate(f_scales, qbers, weights)
    return a0





# ──────────────────────────────────────────────────────────────────────
# 4. FULL SWEEP COLLECTOR  (the E11 grid, reusable from the notebook)
# ──────────────────────────────────────────────────────────────────────

def run_zne_sweep(
    base_noise_model: str,
    f_scales: List[float],
    p_eve_grid: List[float],
    seeds: List[int],
    n_qubits: int = 2000,
    base_depolar_prob: float = 0.05,
    base_t1_ns: float = 500.0,
    base_t2_ns: float = 200.0,
    gate_time_ns: float = 50.0,
    sample_fraction: float = 0.15,
) -> Dict:
    """
    Runs the full (p_eve x f_scale) grid across seeds, computing raw
    QBER + Wilson CI at every point, then linear and exponential ZNE
    extrapolation per p_eve.

    Returns a dict:
        raw[p_eve][f_scale] = list of (qber_pct, ci_low_pct, ci_high_pct) per seed
        zne_linear[p_eve]   = extrapolated QBER (%) at f=0, linear fit
        zne_exp[p_eve]      = extrapolated QBER (%) at f=0, exponential fit
        qber_f1_mean[p_eve] = mean raw QBER (%) at f_scale=1.0 (what you'd
                              normally report without ZNE)
    """
    raw: Dict[float, Dict[float, list]] = {}

    for p_eve in p_eve_grid:
        raw[p_eve] = {}
        for f_scale in f_scales:
            raw[p_eve][f_scale] = []
            for seed in seeds:
                cfg = build_scaled_config(
                    base_noise_model, f_scale, n_qubits, seed, p_eve,
                    base_depolar_prob, base_t1_ns, base_t2_ns,
                    gate_time_ns, sample_fraction,
                )
                r  = run_simulation(cfg, verbose=False)
                qr = r.qber_result
                raw[p_eve][f_scale].append((
                    qr.qber * 100,
                    qr.confidence_low * 100,
                    qr.confidence_high * 100,
                ))

    zne_linear: Dict[float, float] = {}
    zne_exp: Dict[float, float] = {}
    qber_f1_mean: Dict[float, float] = {}
    zne_exp_converged: Dict[float, bool] = {}
    zne_exp_raw_dict: Dict[float, float] = {}


    for p_eve in p_eve_grid:
        mean_qbers  = []
        mean_weights = []
        for f_scale in f_scales:
            vals = raw[p_eve][f_scale]
            qbers_here = [v[0] for v in vals]
            halfwidths = [max(1e-3, (v[2] - v[1]) / 2) for v in vals]
            mean_qbers.append(float(np.mean(qbers_here)))
            # inverse-variance weight from mean CI half-width across seeds
            mean_weights.append(1.0 / (float(np.mean(halfwidths)) ** 2))

        zne_linear[p_eve] = zne_estimate_linear(f_scales, mean_qbers, mean_weights)
        zne_exp_val, zne_exp_ok, zne_exp_raw = zne_estimate_exponential(f_scales,mean_qbers)
        zne_exp[p_eve] = zne_exp_val
        zne_exp_converged[p_eve] = zne_exp_ok
        zne_exp_raw_dict[p_eve] = zne_exp_raw
        qber_f1_mean[p_eve] = mean_qbers[f_scales.index(1.0)] if 1.0 in f_scales else mean_qbers[0]

    return {
        "raw": raw,
        "zne_linear": zne_linear,
        "zne_exp": zne_exp,
        "zne_exp_converged": zne_exp_converged,
        "zne_exp_raw": zne_exp_raw_dict,
        "qber_f1_mean": qber_f1_mean,
        "f_scales": f_scales,
        "p_eve_grid": p_eve_grid,
}


# ──────────────────────────────────────────────────────────────────────
# 5. UI-FACING SINGLE-CONFIG SWEEP  (interactive, not the notebook grid)
# ──────────────────────────────────────────────────────────────────────

_ZNE_SUPPORTED_MODELS = (
    NoiseModelType.DEPOLARIZING,
    NoiseModelType.AMPLITUDE_DAMPING,
    NoiseModelType.PHASE_DAMPING,
)


@dataclass
class ZNEResult:
    """Output of one run_zne_analysis() sweep."""
    base_label: str
    noise_model: str
    f_scales: List[float]
    n_seeds: int
    per_f_qber: Dict[float, Tuple[float, float, float]]   # mean%, ci_low%, ci_high%
    linear_intercept: float
    linear_slope: float
    exponential: Dict[str, float]
    quadratic_intercept: float
    bootstrap_ci: Optional[Tuple[float, float, float]]
    qber_at_f1: float
    recommended_estimate: float
    runtime_seconds: float
    n_qubits: int = 0
    """Qubits-per-point actually used by this sweep (0 = unknown, for
    results produced before this field existed)."""
    base_seed: int = 0
    """Seed used for the first sample at every f_scale (seed index 0).
    Equal to base_config.seed (or 0 if that was None) at call time -
    lets a caller check whether f=1/seed-0 exactly reproduces a given
    SimulationConfig's own run."""


def run_zne_analysis(
    base_config: SimulationConfig,
    f_scales: List[float],
    n_seeds: int = 5,
    method: str = "linear",
    bootstrap: bool = False,
) -> ZNEResult:
    """
    Run a Zero-Noise Extrapolation sweep seeded from one interactive
    SimulationConfig, treated as the f_scale=1.0 baseline.

    Unlike run_zne_sweep() (grid-oriented over p_eve x f_scale x seeds,
    built for notebook figures), this runs a single p_eve (whatever
    base_config already has) across f_scales x n_seeds simulations,
    reusing base_config's own noise parameters as the scaling basis
    rather than requiring them as separate kwargs.

    Raises ValueError if base_config.noise_model isn't one of the three
    scalable Kraus-noise models — fibre_loss/ideal have no ZNE meaning,
    the same restriction build_scaled_config already enforces.
    """
    if base_config.noise_model not in _ZNE_SUPPORTED_MODELS:
        raise ValueError(
            f"ZNE not supported for noise_model={base_config.noise_model!r}; "
            f"must be one of {_ZNE_SUPPORTED_MODELS}."
        )

    t0 = time.perf_counter()
    p_eve = base_config.eve_intercept_prob if base_config.eve_present else 0.0
    base_seed = base_config.seed if base_config.seed is not None else 0

    per_f_qber: Dict[float, Tuple[float, float, float]] = {}
    per_seed_qbers: Dict[float, List[float]] = {}
    mean_qbers: List[float] = []
    weights: List[float] = []

    for f_scale in f_scales:
        qbers_here: List[float] = []
        halfwidths: List[float] = []
        for i in range(n_seeds):
            cfg = build_scaled_config(
                base_config.noise_model, f_scale,
                n_qubits=base_config.n_qubits,
                seed=base_seed + i,
                p_eve=p_eve,
                base_depolar_prob=base_config.depolar_prob,
                base_t1_ns=base_config.t1_ns,
                base_t2_ns=base_config.t2_ns,
                gate_time_ns=base_config.gate_time_ns,
                sample_fraction=base_config.sample_fraction,
            )
            r = run_simulation(cfg, verbose=False)
            qr = r.qber_result
            qbers_here.append(qr.qber * 100)
            halfwidths.append(max(1e-3, (qr.confidence_high - qr.confidence_low) * 100 / 2))

        mean_q = float(np.mean(qbers_here))
        if n_seeds > 1:
            ci_lo = float(np.percentile(qbers_here, 2.5))
            ci_hi = float(np.percentile(qbers_here, 97.5))
        else:
            # Single-run mode has no seed spread to take a percentile of -
            # report the run's own analytic Wilson 95% CI instead of a
            # misleading zero-width band around the point estimate.
            ci_lo = mean_q - halfwidths[0]
            ci_hi = mean_q + halfwidths[0]
        per_f_qber[f_scale] = (mean_q, ci_lo, ci_hi)
        per_seed_qbers[f_scale] = qbers_here
        mean_qbers.append(mean_q)
        weights.append(1.0 / (float(np.mean(halfwidths)) ** 2))

    linear_a, linear_b = linear_extrapolate(f_scales, mean_qbers, weights)
    exp_result = exponential_extrapolate(f_scales, mean_qbers)
    quad_a, _ = quadratic_extrapolate(f_scales, mean_qbers, weights)

    boot_ci = None
    if bootstrap and n_seeds > 1:
        boot_ci = bootstrap_zne_intercept(per_seed_qbers, f_scales)
    elif bootstrap:
        print("[bb84_zne] WARNING: bootstrap requested with n_seeds=1; "
              "there is no seed spread to resample, so skipping it rather "
              "than returning a fake zero-width interval.")

    if method == "exponential" and exp_result["converged"]:
        recommended = exp_result["estimate"]
    else:
        recommended = max(0.0, linear_a)

    qber_at_f1 = per_f_qber[1.0][0] if 1.0 in per_f_qber else mean_qbers[0]

    return ZNEResult(
        base_label=base_config.label,
        noise_model=base_config.noise_model,
        f_scales=list(f_scales),
        n_seeds=n_seeds,
        per_f_qber=per_f_qber,
        linear_intercept=max(0.0, linear_a),
        linear_slope=linear_b,
        exponential=exp_result,
        quadratic_intercept=quad_a,
        bootstrap_ci=boot_ci,
        qber_at_f1=qber_at_f1,
        recommended_estimate=recommended,
        runtime_seconds=time.perf_counter() - t0,
        n_qubits=base_config.n_qubits,
        base_seed=base_seed,
    )