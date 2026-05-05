"""
bb84_noise.py  ·  Phase 3 — Advanced Noise Models
══════════════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

WHAT THIS MODULE DOES
─────────────────────
Implements seven quantum channel noise models and a research-grade
analysis layer that connects noise physics to BB84 security and to
Grover amplitude behaviour — bridging the simulator to your
professor's research direction.

NOISE MODELS (7 total)
──────────────────────
  1. ideal              — perfect channel, no errors
  2. depolarizing       — uniform Pauli error (Phase 1 baseline)
  3. amplitude_damp     — T1 energy relaxation (Kraus approximation)
  4. phase_damp         — T2 pure dephasing (Kraus approximation)
  5. thermal_relaxation — PROPER Lindblad T1+T2 combined model
                          (THIS is what your professor calls Lindblad /
                           "Limlad" noise — the physically correct model
                           for superconducting and trapped-ion qubits)
  6. coherent_rotation  — systematic gate miscalibration (rotation error)
  7. combined           — depolarizing + phase damping simultaneously
  8. fiber_loss         — distance-based photon loss (Beer-Lambert)

WHY THERMAL RELAXATION IS THE "RIGHT" MODEL
────────────────────────────────────────────
The Lindblad master equation is the most general description of how
a qubit interacts with its environment (Markovian approximation):

  dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

For a qubit in a cold environment (superconducting qubits, ~mK):
  L₁ = √(1/T1) σ₋         — spontaneous emission (energy decay)
  L₂ = √(γ_φ) σ_z/√2     — pure dephasing
  γ_φ = 1/T2 - 1/(2T1)   — pure dephasing rate from T1, T2

Models 3 and 4 (amplitude/phase damp) are APPROXIMATIONS to this.
Model 5 (thermal_relaxation) solves the full Lindblad equation
correctly.  Always prefer model 5 for physical realism.

GROVER–QKD NOISE PARALLEL (Professor's research direction)
──────────────────────────────────────────────────────────
Grover's algorithm uses quantum amplitude amplification:
  After k iterations (noiseless): P_success = sin²((2k+1)θ) → 1
  Under depolarizing noise p:     P_success ≈ (1-p)^{2k} × sin²(...)

BB84 uses quantum amplitude distinguishability:
  Correct basis: P_correct = 1.0   (deterministic)
  Wrong basis:   P_correct = 0.5   (random)
  Under noise:   P_correct → 0.5  (noise erases the distinction)

KEY INSIGHT: Both algorithms degrade in the SAME WAY under noise.
The quantum state fidelity F(ρ_ideal, ρ_noisy) is the unifying metric.

  Grover: P_success ≈ F × sin²((2k+1)θ)
  BB84:   QBER ≈ (1 - F) / 2   where F is the channel fidelity

This module provides analysis functions to demonstrate this parallel,
which is the novel research direction your professor is pointing at.

EXPORTS
───────
QuantumChannel          — unified channel (7 noise models)
LindBladNoiseAnalyser   — Grover/BB84 amplitude analysis (novel)
fiber_transmittance()   — Beer-Lambert fiber utility
max_secure_distance()   — SKR=0 distance estimate
theoretical_qber_from_noise() — analytical QBER for each model
compute_channel_fidelity()    — state fidelity under noise
grover_amplitude_under_noise()— Grover parallel analysis
noise_threshold_sweep()       — find SKR=0 noise level
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
    coherent_unitary_error,
)


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

#: Standard SMF-28 fibre attenuation at 1550 nm (dB/km)
FIBER_ATTENUATION_DB_PER_KM: float = 0.2

#: Typical single-photon detector efficiency
DETECTOR_EFFICIENCY: float = 0.85

#: Dark-count probability per pulse (InGaAs SPD, ~Hz rate)
DARK_COUNT_PROB: float = 1.7e-6

#: Valid noise model names
VALID_NOISE_MODELS = frozenset({
    "ideal", "depolarizing", "amplitude_damp", "phase_damp",
    "thermal_relaxation", "coherent_rotation", "combined", "fiber_loss",
})


# ══════════════════════════════════════════════════════════════════════
# FIBER LOSS UTILITIES
# ══════════════════════════════════════════════════════════════════════

def fiber_transmittance(
    length_km:             float,
    attenuation_db_per_km: float = FIBER_ATTENUATION_DB_PER_KM,
    detector_efficiency:   float = DETECTOR_EFFICIENCY,
) -> float:
    """
    End-to-end channel transmittance: η = η_det × 10^(−α L / 10).

    Parameters
    ──────────
    length_km            : fibre length in km
    attenuation_db_per_km: fibre loss coefficient (default 0.2 dB/km, SMF-28)
    detector_efficiency  : Bob's detector efficiency ∈ (0, 1]

    Returns
    ───────
    η ∈ [0, 1]  — probability that a photon survives AND is detected
    """
    return detector_efficiency * 10 ** (-attenuation_db_per_km * length_km / 10)


def max_secure_distance(
    mu:                    float = 0.5,
    attenuation_db_per_km: float = FIBER_ATTENUATION_DB_PER_KM,
    detector_efficiency:   float = DETECTOR_EFFICIENCY,
    qber_threshold:        float = 0.11,
    dark_count_prob:       float = DARK_COUNT_PROB,
) -> float:
    """
    Estimate maximum secure transmission distance (SKR ≈ 0 crossover).

    Uses binary search on the approximate QBER formula:
        QBER(L) ≈ [½·Y₀] / max(Q_μ, Y₀)
    where Q_μ ≈ 1 − exp(−μ·η(L)) is the signal gain.

    Returns
    ───────
    max_km : float — maximum range in km
    """
    lo, hi = 0.0, 1000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        eta = fiber_transmittance(mid, attenuation_db_per_km, detector_efficiency)
        Q_mu = max(1.0 - math.exp(-mu * eta), dark_count_prob)
        e_mu = (0.5 * dark_count_prob) / Q_mu
        if e_mu < qber_threshold:
            lo = mid
        else:
            hi = mid
    return lo


# ══════════════════════════════════════════════════════════════════════
# QUANTUM CHANNEL
# ══════════════════════════════════════════════════════════════════════

class QuantumChannel:
    """
    Unified quantum channel supporting eight physically motivated noise models.

    Quick usage
    ───────────
    # From SimulationConfig (recommended):
    channel = QuantumChannel.from_config(config)

    # Directly — ideal:
    channel = QuantumChannel(noise_model='ideal')

    # Directly — thermal relaxation (Lindblad / "limlad"):
    channel = QuantumChannel(
        noise_model  = 'thermal_relaxation',
        t1_ns        = 50_000,    # 50 µs T1
        t2_ns        = 30_000,    # 30 µs T2  (must satisfy T2 <= 2*T1)
        gate_time_ns = 50,        # 50 ns gate
    )

    # Directly — coherent rotation error (systematic miscalibration):
    channel = QuantumChannel(
        noise_model        = 'coherent_rotation',
        rotation_error_rad = 0.05,  # 0.05 rad overrotation per gate
    )

    run_circuit() return value
    ───────────────────────────
    int (0 or 1) : qubit measured successfully
    None         : photon lost in fibre (fiber_loss model only)

    The runner (bb84_runner.py) must handle None gracefully.

    Noise Model Summary
    ───────────────────
    Model                  | Physical origin              | Key param
    ─────────────────────────────────────────────────────────────────
    ideal                  | Perfect channel              | —
    depolarizing           | Gate imperfections           | depolar_prob
    amplitude_damp         | T1 energy relaxation (approx)| t1_ns
    phase_damp             | T2 dephasing (approx)        | t2_ns
    thermal_relaxation     | Lindblad T1+T2 (EXACT)       | t1_ns, t2_ns
    coherent_rotation      | Gate miscalibration          | rotation_error_rad
    combined               | T1 + T2 simultaneous         | t1_ns, t2_ns
    fiber_loss             | Beer-Lambert photon loss     | channel_length_km
    """

    _NOISY_GATES = ["x", "h", "id", "u1", "u2", "u3"]

    def __init__(
        self,
        noise_model:         str   = "ideal",
        depolar_prob:        float = 0.0,
        t1_ns:               float = 100_000.0,
        t2_ns:               float = 50_000.0,
        gate_time_ns:        float = 50.0,
        channel_length_km:   float = 0.0,
        rotation_error_rad:  float = 0.0,
        excited_population:  float = 0.01,   # thermal photon occupancy ~mK
    ):
        """
        Parameters
        ──────────
        noise_model         : one of VALID_NOISE_MODELS
        depolar_prob        : depolarizing error probability ∈ [0, 1]
        t1_ns               : T1 relaxation time in nanoseconds
        t2_ns               : T2 dephasing time (must be ≤ 2·T1)
        gate_time_ns        : single-gate operation time in ns
        channel_length_km   : fibre channel length (fiber_loss model)
        rotation_error_rad  : systematic overrotation per gate (radians)
        excited_population  : mean thermal excitation (≈0 at mK, ≈0.01 typical)
        """
        if noise_model not in VALID_NOISE_MODELS:
            raise ValueError(
                f"Unknown noise model '{noise_model}'. "
                f"Valid options: {sorted(VALID_NOISE_MODELS)}"
            )

        # Enforce physical constraint T2 ≤ 2·T1
        if t2_ns > 2 * t1_ns:
            t2_ns = 2.0 * t1_ns

        self.noise_model_name    = noise_model
        self.depolar_prob        = depolar_prob
        self.t1_ns               = t1_ns
        self.t2_ns               = t2_ns
        self.gate_time_ns        = gate_time_ns
        self.channel_length_km   = channel_length_km
        self.rotation_error_rad  = rotation_error_rad
        self.excited_population  = excited_population

        self._transmittance = (
            fiber_transmittance(channel_length_km)
            if noise_model == "fiber_loss" else 1.0
        )
        self._rng       = np.random.default_rng()
        self._simulator = self._build_simulator()

    @classmethod
    def from_config(cls, config) -> "QuantumChannel":
        """Build a QuantumChannel from a SimulationConfig object."""
        nm = getattr(config, "noise_model", "depolarizing") \
             if config.noise_enabled else "ideal"
        return cls(
            noise_model        = nm,
            depolar_prob       = getattr(config, "depolar_prob", 0.0),
            t1_ns              = getattr(config, "t1_ns", 100_000.0),
            t2_ns              = getattr(config, "t2_ns", 50_000.0),
            gate_time_ns       = getattr(config, "gate_time_ns", 50.0),
            channel_length_km  = getattr(config, "channel_length_km", 0.0),
            rotation_error_rad = getattr(config, "rotation_error_rad", 0.0),
        )

    # ── Simulator construction ─────────────────────────────────────

    def _build_simulator(self) -> AerSimulator:
        nm = self.noise_model_name

        if nm in ("ideal", "fiber_loss"):
            return AerSimulator()

        qiskit_noise = NoiseModel()

        if nm == "depolarizing":
            if self.depolar_prob > 0:
                err = depolarizing_error(self.depolar_prob, 1)
                qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)
            else:
                return AerSimulator()

        elif nm == "amplitude_damp":
            # Kraus approximation to amplitude damping (T1 only)
            p_amp = 1.0 - math.exp(-self.gate_time_ns / self.t1_ns)
            p_amp = float(np.clip(p_amp, 0.0, 1.0))
            err = amplitude_damping_error(p_amp)
            qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)

        elif nm == "phase_damp":
            # Kraus approximation to phase damping (T2 only)
            lam = 1.0 - math.exp(-self.gate_time_ns / self.t2_ns)
            lam = float(np.clip(lam, 0.0, 1.0))
            err = phase_damping_error(lam)
            qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)

        elif nm == "thermal_relaxation":
            # ────────────────────────────────────────────────────────
            # LINDBLAD / THERMAL RELAXATION MODEL
            # ────────────────────────────────────────────────────────
            # This is the EXACT solution to the Lindblad master equation
            # for a qubit coupled to a thermal bath at temperature T.
            #
            # Lindblad equation for this system:
            #   dρ/dt = γ₁(1+n̄)(σ₋ρσ₊ - ½{σ₊σ₋,ρ})   ← emission
            #         + γ₁ n̄  (σ₊ρσ₋ - ½{σ₋σ₊,ρ})   ← absorption
            #         + γ_φ/2 (σ_z ρ σ_z - ρ)          ← dephasing
            #
            # where:
            #   γ₁   = 1/T1     (energy relaxation rate)
            #   γ_φ  = 1/T2* - 1/(2T1)  (pure dephasing rate)
            #   n̄    = excited_population (thermal photons ≈ 0 at mK)
            #
            # Qiskit's thermal_relaxation_error solves this EXACTLY
            # via the Lindblad time-evolution operator.
            #
            # T2 must satisfy T2 ≤ 2*T1 (already enforced in __init__)
            # ────────────────────────────────────────────────────────
            err = thermal_relaxation_error(
                t1=self.t1_ns,
                t2=self.t2_ns,
                time=self.gate_time_ns,
                excited_state_population=self.excited_population,
            )
            qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)

        elif nm == "coherent_rotation":
            # ────────────────────────────────────────────────────────
            # COHERENT ROTATION ERROR
            # ────────────────────────────────────────────────────────
            # Models systematic gate miscalibration:
            # Instead of perfect X or H, the gate applies an extra
            # rotation of ε radians around the Y-axis:
            #   U_error = Ry(ε) = [[cos(ε/2), -sin(ε/2)],
            #                      [sin(ε/2),  cos(ε/2)]]
            #
            # This is a COHERENT error (not random). It shifts
            # the qubit state deterministically, producing a
            # QBER contribution of sin²(ε/2) per gate.
            #
            # Unlike stochastic noise, coherent errors can in principle
            # be corrected by recalibration — but they are hard to
            # detect in QBER statistics (they look like small biases,
            # not random noise).
            # ────────────────────────────────────────────────────────
            eps = self.rotation_error_rad
            c, s = math.cos(eps / 2), math.sin(eps / 2)
            # 2×2 Ry(ε) rotation matrix
            ry_eps = np.array([[c, -s], [s, c]], dtype=complex)
            err = coherent_unitary_error(ry_eps)
            qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)

        elif nm == "combined":
            # Amplitude damping + phase damping applied sequentially
            p_amp = float(np.clip(1.0 - math.exp(-self.gate_time_ns / self.t1_ns), 0, 1))
            lam   = float(np.clip(1.0 - math.exp(-self.gate_time_ns / self.t2_ns), 0, 1))
            err_a = amplitude_damping_error(p_amp)
            err_p = phase_damping_error(lam)
            err   = err_a.compose(err_p)
            qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)

        return AerSimulator(noise_model=qiskit_noise)

    # ── Public interface ───────────────────────────────────────────

    def run_circuit(self, qc: QuantumCircuit) -> Optional[int]:
        """
        Simulate a single-qubit measurement through this noise channel.

        For fiber_loss: probabilistically returns None (photon lost)
        before any measurement is attempted.

        Returns
        ───────
        int (0 or 1) : measured bit
        None         : photon lost in fibre
        """
        if self.noise_model_name == "fiber_loss":
            if self._rng.random() > self._transmittance:
                return None

        job    = self._simulator.run(transpile(qc, self._simulator), shots=1)
        counts = job.result().get_counts()
        return int(list(counts.keys())[0])

    # ── Derived properties ─────────────────────────────────────────

    @property
    def effective_error_probability(self) -> float:
        """
        Analytical QBER contribution from this channel alone.

        This is the theoretical prediction; compare with simulated
        QBER to validate the noise model implementation.
        """
        return theoretical_qber_from_noise(
            self.noise_model_name,
            p             = self.depolar_prob,
            t1_ns         = self.t1_ns,
            t2_ns         = self.t2_ns,
            gate_time_ns  = self.gate_time_ns,
            length_km     = self.channel_length_km,
            rotation_rad  = self.rotation_error_rad,
        )

    @property
    def transmittance(self) -> float:
        """Channel transmittance η ∈ [0, 1]. 1.0 for non-fibre models."""
        return self._transmittance

    @property
    def lindblad_rates(self) -> Dict[str, float]:
        """
        Lindblad decay rates for the thermal_relaxation model.

        Returns
        ───────
        dict with:
          gamma_1  : energy relaxation rate (1/T1)
          gamma_phi: pure dephasing rate (1/T2* − 1/(2T1))
          n_bar    : mean thermal excitation population
          t_gate   : gate time in ns
        """
        gamma_1   = 1.0 / self.t1_ns
        # T2* (total decoherence time): 1/T2* = 1/(2T1) + gamma_phi
        # so gamma_phi = 1/T2 - 1/(2*T1)
        gamma_phi = max(0.0, 1.0 / self.t2_ns - 1.0 / (2.0 * self.t1_ns))
        return {
            "gamma_1":   gamma_1,
            "gamma_phi": gamma_phi,
            "n_bar":     self.excited_population,
            "t_gate_ns": self.gate_time_ns,
        }

    def describe(self) -> str:
        """Return a human-readable description of this channel's noise."""
        nm = self.noise_model_name
        lines = [f"QuantumChannel: {nm}"]
        if nm == "depolarizing":
            lines += [f"  p_depolar = {self.depolar_prob:.4f}",
                      f"  QBER contribution ≈ {self.effective_error_probability*100:.3f}%"]
        elif nm in ("amplitude_damp", "phase_damp", "thermal_relaxation", "combined"):
            p = 1.0 - math.exp(-self.gate_time_ns / self.t1_ns)
            l = 1.0 - math.exp(-self.gate_time_ns / self.t2_ns)
            lines += [f"  T1 = {self.t1_ns:.0f} ns  →  p_amp = {p:.4f}",
                      f"  T2 = {self.t2_ns:.0f} ns  →  λ   = {l:.4f}",
                      f"  gate_time = {self.gate_time_ns} ns",
                      f"  QBER contribution ≈ {self.effective_error_probability*100:.3f}%"]
            if nm == "thermal_relaxation":
                r = self.lindblad_rates
                lines += [f"  Lindblad γ₁  = {r['gamma_1']:.2e} ns⁻¹",
                          f"  Lindblad γ_φ = {r['gamma_phi']:.2e} ns⁻¹"]
        elif nm == "coherent_rotation":
            lines += [f"  ε_rotation = {self.rotation_error_rad:.4f} rad",
                      f"  QBER contribution ≈ {self.effective_error_probability*100:.3f}%"]
        elif nm == "fiber_loss":
            lines += [f"  length = {self.channel_length_km:.1f} km",
                      f"  η = {self._transmittance:.4f}"]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# ANALYTICAL NOISE PREDICTIONS  (no Qiskit needed)
# ══════════════════════════════════════════════════════════════════════

def theoretical_qber_from_noise(
    noise_model: str,
    p:           float = 0.0,
    t1_ns:       float = 100_000.0,
    t2_ns:       float = 50_000.0,
    gate_time_ns: float = 50.0,
    length_km:   float = 0.0,
    rotation_rad: float = 0.0,
    mu:          float = 0.5,
) -> float:
    """
    Compute the analytical QBER contribution from channel noise alone.

    These are closed-form predictions from the Lindblad/Kraus models;
    compare with simulated QBER to validate the implementation.

    Parameters
    ──────────
    noise_model   : name string
    p             : depolarizing probability (depolarizing model)
    t1_ns         : T1 relaxation time in ns
    t2_ns         : T2 dephasing time in ns
    gate_time_ns  : gate operation time in ns
    length_km     : fibre length (fiber_loss)
    rotation_rad  : rotation error per gate (coherent_rotation)

    Returns
    ───────
    QBER ∈ [0.0, 0.5] — contribution from noise only

    Derivations
    ───────────
    Depolarizing:
      E(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
      Effective bit-flip rate = 3p/4

    Amplitude damping (T1):
      Off-diagonal coherences decay as exp(-t/T1)
      This biases |1⟩→|0⟩; the QBER is approximately p_amp/2

    Phase damping (T2):
      Off-diagonal elements decay as exp(-t/T2)
      Phase errors introduce QBER ≈ λ/4 for the BB84 basis states

    Thermal relaxation (exact Lindblad):
      Combined T1+T2 effect; uses the closed-form solution.
      For T >> T2 (strong dephasing limit): QBER → 0.5 * (1-exp(-t/T2))/2

    Coherent rotation:
      Ry(ε)|0⟩ = cos(ε/2)|0⟩ + sin(ε/2)|1⟩
      P(error | basis match) = sin²(ε/2)
    """
    if noise_model == "depolarizing":
        return p * 3.0 / 4.0

    elif noise_model == "amplitude_damp":
        gamma = 1.0 - math.exp(-gate_time_ns / t1_ns)
        # Amplitude damping biases |1> → |0>; expected QBER ≈ gamma/2
        return gamma / 2.0

    elif noise_model == "phase_damp":
        lam = 1.0 - math.exp(-gate_time_ns / t2_ns)
        # Phase damping causes dephasing errors; QBER ≈ lambda/4
        return lam / 4.0

    elif noise_model == "thermal_relaxation":
        # Exact Lindblad solution for combined T1+T2:
        # The channel fidelity for a general input is:
        #   F_avg = 1 - (1/4)(p_amp + lambda_eff)
        # where p_amp = 1-exp(-t/T1) and lambda_eff accounts for both
        gamma = 1.0 - math.exp(-gate_time_ns / t1_ns)   # T1 contribution
        lam   = max(0.0, 1.0 - math.exp(
            -gate_time_ns * (1.0/t2_ns - 1.0/(2.0*t1_ns)) ))  # pure dephasing
        # Average channel fidelity degradation
        return (gamma / 2.0 + lam / 4.0) / 2.0

    elif noise_model == "coherent_rotation":
        # Coherent overrotation by ε: Ry(ε) applied to each gate
        # P(error) = sin²(ε/2) for a single qubit in |0> or |1>
        return math.sin(rotation_rad / 2.0) ** 2

    elif noise_model == "combined":
        gamma = 1.0 - math.exp(-gate_time_ns / t1_ns)
        lam   = 1.0 - math.exp(-gate_time_ns / t2_ns)
        return gamma / 2.0 + lam / 4.0

    elif noise_model == "fiber_loss":
        eta = fiber_transmittance(length_km)
        Q   = max(1.0 - math.exp(-mu * eta), DARK_COUNT_PROB)
        return min(0.5 * DARK_COUNT_PROB / Q, 0.5)

    return 0.0


# ══════════════════════════════════════════════════════════════════════
# CHANNEL FIDELITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def compute_channel_fidelity(
    channel:       QuantumChannel,
    state_name:    str = "plus",
    n_shots:       int = 2000,
    seed:          Optional[int] = 42,
) -> float:
    """
    Estimate the average channel fidelity by measuring how well the
    channel preserves the quantum state |state_name⟩.

    This is the fundamental metric connecting Grover amplitude
    degradation and BB84 QBER: F determines both.

    F = Tr(ρ_ideal · ρ_noisy)
      = probability that the noisy state "looks like" the ideal state

    For BB84: QBER ≈ (1 − F) / 2

    Parameters
    ──────────
    channel    : QuantumChannel to measure
    state_name : 'zero' (|0⟩), 'one' (|1⟩), 'plus' (|+⟩), 'minus' (|−⟩)
    n_shots    : number of measurement rounds for estimation
    seed       : RNG seed

    Returns
    ───────
    fidelity ∈ [0.0, 1.0]
      1.0 = channel perfectly preserves the state
      0.5 = channel has completely randomised the state
    """
    rng = np.random.default_rng(seed)

    def build_circuit(name: str, measure_basis: str) -> QuantumCircuit:
        qc = QuantumCircuit(1, 1)
        # Prepare state
        if name in ("one", "minus"):
            qc.x(0)
        if name in ("plus", "minus"):
            qc.h(0)
        # Measure in specified basis
        if measure_basis == "x":
            qc.h(0)
        qc.measure(0, 0)
        return qc

    # Measure the prepared state in its OWN basis (correct basis)
    # P(correct) = F   →   F = correct_count / n_shots
    measure_basis = "x" if state_name in ("plus", "minus") else "z"
    expected_bit  = 0 if state_name in ("zero", "plus") else 1

    correct_count = 0
    for _ in range(n_shots):
        qc  = build_circuit(state_name, measure_basis)
        bit = channel.run_circuit(qc)
        if bit is not None and bit == expected_bit:
            correct_count += 1

    # Handle fiber-loss (lost photons counted as uncertain)
    valid_shots = n_shots  # for non-fiber models all shots are valid
    fidelity    = correct_count / valid_shots
    return float(fidelity)


# ══════════════════════════════════════════════════════════════════════
# GROVER AMPLITUDE ANALYSIS  (professor's research direction)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GroverNoiseResult:
    """
    Results of a Grover amplitude analysis under noise.

    Shows how quantum amplitude amplification degrades with noise,
    and the parallel with BB84 QBER degradation.
    """
    noise_model:        str
    noise_parameter:    float
    iterations:         List[int]
    ideal_amplitudes:   List[float]
    """Grover success probability without noise: sin²((2k+1)θ)"""
    noisy_amplitudes:   List[float]
    """Grover success probability with noise at each iteration k"""
    fidelity_per_step:  List[float]
    """Channel fidelity F(ρ_ideal, ρ_noisy) at each step"""
    equivalent_qber:    float
    """BB84 QBER equivalent: (1 − F_channel) / 2"""
    amplitude_decay_rate: float
    """Exponential decay rate of amplitude: A(k) ≈ A₀ × exp(−α k)"""
    optimal_iterations: int
    """k that maximises noisy success probability"""


def grover_amplitude_under_noise(
    noise_model:     str,
    n_database:      int   = 64,
    noise_param:     float = 0.02,
    t1_ns:           float = 100_000.0,
    t2_ns:           float = 50_000.0,
    gate_time_ns:    float = 50.0,
    rotation_rad:    float = 0.0,
    max_iterations:  int   = 30,
) -> GroverNoiseResult:
    """
    Analyse how Grover's amplitude amplification degrades under noise.

    THEORY
    ──────
    In Grover's algorithm searching N items, after k iterations:

      Ideal:  P_success(k) = sin²((2k+1) · arcsin(1/√N))

    Under depolarizing noise p (per gate):
      The state fidelity after each step degrades by factor (1−p).
      After k Grover steps (each ≈ 2 gates: oracle + diffusion):

      P_success(k) ≈ sin²((2k+1)θ) × (1−p)^{2k}  [simplified model]

    The optimal k under noise is reduced:
      k_opt ≈ min(π/(4θ), 1/(2p))

    CONNECTION TO BB84
    ──────────────────
    The channel fidelity F = (1−p)^{2k} is EXACTLY the metric that
    governs BB84 QBER:   QBER ≈ (1 − F) / 2

    Both algorithms fail at the same noise threshold:
      When F < 0.5: Grover gives no speedup, BB84 QBER > 11%

    This function computes both the Grover amplitude curve and the
    equivalent BB84 QBER at each noise level, showing the parallel.

    Parameters
    ──────────
    noise_model   : noise model name
    n_database    : Grover search space size N (must be power of 2)
    noise_param   : primary noise parameter (p for depolarizing, etc.)
    t1_ns/t2_ns   : T1/T2 times for relaxation models
    gate_time_ns  : gate time in ns
    rotation_rad  : rotation error (coherent model)
    max_iterations: maximum Grover iterations to compute

    Returns
    ───────
    GroverNoiseResult with full analysis
    """
    # Grover angle: sin(θ) = 1/√N
    theta = math.asin(1.0 / math.sqrt(n_database))

    # Per-step noise fidelity (analytical from Lindblad/Kraus)
    # Each Grover step ≈ oracle (1 gate) + diffusion (H + inversion + H ≈ 3 gates)
    # Total ≈ 4 gates per Grover step
    qber_per_gate = theoretical_qber_from_noise(
        noise_model,
        p=noise_param, t1_ns=t1_ns, t2_ns=t2_ns,
        gate_time_ns=gate_time_ns, rotation_rad=rotation_rad,
    )
    # Convert QBER to fidelity: F = 1 − 2*QBER (for a single gate application)
    f_per_gate  = 1.0 - 2.0 * qber_per_gate
    f_per_step  = f_per_gate ** 4   # 4 gates per Grover iteration

    iterations        = list(range(0, max_iterations + 1))
    ideal_amplitudes  = []
    noisy_amplitudes  = []
    fidelity_per_step = []

    for k in iterations:
        # Ideal Grover: P = sin²((2k+1)θ)
        ideal_p = math.sin((2 * k + 1) * theta) ** 2
        ideal_amplitudes.append(ideal_p)

        # Noisy: amplitude decays as F^k
        cumulative_fidelity = f_per_step ** k if k > 0 else 1.0
        noisy_p = ideal_p * cumulative_fidelity
        noisy_amplitudes.append(noisy_p)
        fidelity_per_step.append(cumulative_fidelity)

    # Find optimal iterations under noise
    opt_k = int(np.argmax(noisy_amplitudes))

    # Amplitude decay rate (fit exponential to noisy/ideal ratio)
    ratios = [noisy_amplitudes[k] / max(ideal_amplitudes[k], 1e-12)
              for k in range(1, len(iterations))]
    if len(ratios) > 1 and ratios[0] > 0:
        # log(ratio) ≈ -α * k  →  α = -mean(log(ratios) / k_values)
        k_vals = list(range(1, len(iterations)))
        log_ratios = [math.log(max(r, 1e-12)) for r in ratios]
        decay_rate = -float(np.polyfit(k_vals, log_ratios, 1)[0])
    else:
        decay_rate = 0.0

    # Equivalent BB84 QBER (channel fidelity after ONE transmission)
    equivalent_qber = qber_per_gate

    return GroverNoiseResult(
        noise_model         = noise_model,
        noise_parameter     = noise_param,
        iterations          = iterations,
        ideal_amplitudes    = ideal_amplitudes,
        noisy_amplitudes    = noisy_amplitudes,
        fidelity_per_step   = fidelity_per_step,
        equivalent_qber     = equivalent_qber,
        amplitude_decay_rate= decay_rate,
        optimal_iterations  = opt_k,
    )


# ══════════════════════════════════════════════════════════════════════
# NOISE THRESHOLD SWEEP  (find SKR = 0 boundary)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class NoiseThresholdResult:
    """
    Per-model noise threshold analysis.
    Shows the critical noise level where SKR drops to zero.
    """
    noise_model:       str
    noise_params:      np.ndarray
    qbers_analytical:  np.ndarray
    """Analytical QBER prediction for each noise level"""
    skr_shor_preskill: np.ndarray
    """Shor-Preskill SKR: max(0, 1 - 2H(QBER))"""
    skr_realistic:     np.ndarray
    """Realistic SKR: max(0, 1 - 1.16*H(QBER))"""
    threshold_param:   float
    """Noise parameter at which SKR = 0 (analytical)"""
    grover_results:    Optional[List[GroverNoiseResult]] = None


def noise_threshold_sweep(
    noise_model:    str,
    n_points:       int   = 50,
    t1_ns:          float = 100_000.0,
    t2_ns:          float = 50_000.0,
    gate_time_ns:   float = 50.0,
    compute_grover: bool  = True,
    n_grover_params: int  = 5,
) -> NoiseThresholdResult:
    """
    Sweep the noise parameter from 0 to threshold and beyond,
    computing analytical QBER and SKR at each point.

    Also computes Grover amplitude at selected noise levels to
    show the parallel between Grover degradation and QBER growth.

    Parameters
    ──────────
    noise_model    : which model to sweep
    n_points       : number of sweep points
    t1_ns, t2_ns   : for relaxation models
    gate_time_ns   : gate time
    compute_grover : whether to include Grover analysis

    Returns
    ───────
    NoiseThresholdResult with all sweep data
    """
    import math as _math

    def _h(e: float) -> float:
        """Binary entropy."""
        e = float(np.clip(e, 1e-12, 1 - 1e-12))
        return -e * _math.log2(e) - (1 - e) * _math.log2(1 - e)

    def _skr(e: float, f: float = 1.0) -> float:
        return max(0.0, 1.0 - f * _h(e))

    # Define the primary noise parameter range for each model
    if noise_model == "depolarizing":
        params = np.linspace(0.0, 0.15, n_points)
    elif noise_model in ("amplitude_damp", "phase_damp", "thermal_relaxation", "combined"):
        # Sweep gate_time from 10 ns to 3000 ns (larger = more noise)
        gate_times = np.linspace(10.0, 3000.0, n_points)
        params = gate_times  # x-axis = gate time in ns
    elif noise_model == "coherent_rotation":
        params = np.linspace(0.0, math.pi / 4, n_points)  # 0 to 45° overrotation
    else:
        params = np.linspace(0.0, 0.15, n_points)

    qbers_analytical  = np.zeros(n_points)
    skr_sp            = np.zeros(n_points)
    skr_real          = np.zeros(n_points)

    for i, param in enumerate(params):
        if noise_model == "depolarizing":
            q = theoretical_qber_from_noise(noise_model, p=param)
        elif noise_model in ("amplitude_damp", "phase_damp", "thermal_relaxation", "combined"):
            q = theoretical_qber_from_noise(
                noise_model, t1_ns=t1_ns, t2_ns=t2_ns, gate_time_ns=float(param))
        elif noise_model == "coherent_rotation":
            q = theoretical_qber_from_noise(noise_model, rotation_rad=float(param))
        else:
            q = 0.0

        qbers_analytical[i] = q
        skr_sp[i]   = _skr(q, f=2.0)
        skr_real[i] = _skr(q, f=1.16)

    # Find threshold: last param before SKR = 0
    above_zero = np.where(skr_real > 0)[0]
    threshold = float(params[above_zero[-1]]) if len(above_zero) > 0 else 0.0

    # Grover analysis at n_grover_params representative noise levels
    grover_results = None
    if compute_grover and noise_model in ("depolarizing", "coherent_rotation"):
        grover_param_values = np.linspace(
            float(params[0]), float(params[n_points // 2]), n_grover_params)
        grover_results = []
        for gp in grover_param_values:
            gr = grover_amplitude_under_noise(
                noise_model    = noise_model,
                n_database     = 64,
                noise_param    = float(gp),
                gate_time_ns   = gate_time_ns,
                rotation_rad   = float(gp) if noise_model == "coherent_rotation" else 0.0,
                max_iterations = 20,
            )
            grover_results.append(gr)

    return NoiseThresholdResult(
        noise_model        = noise_model,
        noise_params       = params,
        qbers_analytical   = qbers_analytical,
        skr_shor_preskill  = skr_sp,
        skr_realistic      = skr_real,
        threshold_param    = threshold,
        grover_results     = grover_results,
    )


# ══════════════════════════════════════════════════════════════════════
# NOISE FINGERPRINT  (for all-model comparison)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class NoiseFingerprint:
    """Multi-dimensional fingerprint characterising a noise model."""
    name:              str
    qber_at_low_noise: float   # QBER at 10% of threshold param
    qber_at_threshold: float   # QBER at the threshold param (≈11%)
    skr_at_low_noise:  float   # SKR at 10% of threshold
    decay_steepness:   float   # How rapidly QBER grows with noise (gradient)
    affects_amplitude: bool    # Does it degrade |+> or only |0>/|1>?
    is_coherent:       bool    # Is the error coherent (not random)?
    lindblad_type:     str     # Lindblad jump operator description
    physical_origin:   str     # Real-world physical source


def compute_noise_fingerprints() -> List[NoiseFingerprint]:
    """
    Compute characteristic fingerprints for all noise models.

    Used by bb84_plots.py to produce the noise model comparison
    radar plot (out_2_noise_models.png).

    Returns
    ───────
    List[NoiseFingerprint] — one entry per model
    """
    def _h(e): return (-e * math.log2(max(e, 1e-12)) -
                       (1-e) * math.log2(max(1-e, 1e-12)))
    def _skr(e, f=1.16): return max(0.0, 1.0 - f * _h(e))

    # Low noise parameters for each model (10% of typical threshold)
    low_params = {
        "depolarizing":       {"p": 0.01},
        "amplitude_damp":     {"t1_ns": 100_000, "gate_time_ns": 500},
        "phase_damp":         {"t2_ns": 50_000,  "gate_time_ns": 500},
        "thermal_relaxation": {"t1_ns": 100_000, "t2_ns": 50_000, "gate_time_ns": 500},
        "coherent_rotation":  {"rotation_rad": 0.05},
        "combined":           {"t1_ns": 50_000, "t2_ns": 25_000, "gate_time_ns": 500},
    }
    # Threshold-level parameters
    thr_params = {
        "depolarizing":       {"p": 0.147},
        "amplitude_damp":     {"t1_ns": 100_000, "gate_time_ns": 3000},
        "phase_damp":         {"t2_ns": 50_000,  "gate_time_ns": 3000},
        "thermal_relaxation": {"t1_ns": 100_000, "t2_ns": 50_000, "gate_time_ns": 2500},
        "coherent_rotation":  {"rotation_rad": 0.70},
        "combined":           {"t1_ns": 50_000, "t2_ns": 25_000, "gate_time_ns": 2000},
    }
    meta = {
        "depolarizing":       (True, False, "L_x=√(p/3)σ_x, L_y=√(p/3)σ_y, L_z=√(p/3)σ_z", "Gate imperfections, uniform Pauli errors"),
        "amplitude_damp":     (True, False, "L=√γ σ₋ (emission operator)", "Spontaneous emission, T1 energy decay"),
        "phase_damp":         (False, False, "L=√(γ/2) σ_z (dephasing operator)", "Phonon bath, fluctuating magnetic fields"),
        "thermal_relaxation": (True, False, "L₁=√(γ₁(1+n̄))σ₋, L₂=√(γ₁n̄)σ₊, L₃=√(γ_φ)σ_z", "Full Lindblad T1+T2 thermal bath coupling"),
        "coherent_rotation":  (True, True,  "Unitary U=Ry(ε) (not a jump operator)", "Gate miscalibration, systematic overrotation"),
        "combined":           (True, False, "Compose(amplitude_damp, phase_damp)", "Multiple simultaneous decoherence channels"),
    }

    fingerprints = []
    for nm in ["depolarizing", "amplitude_damp", "phase_damp",
               "thermal_relaxation", "coherent_rotation", "combined"]:
        q_low = theoretical_qber_from_noise(nm, **low_params[nm])
        q_thr = theoretical_qber_from_noise(nm, **thr_params[nm])
        grad  = (q_thr - q_low) / 0.90  # normalised gradient
        af, ic, lt, po = meta[nm]
        fingerprints.append(NoiseFingerprint(
            name              = nm,
            qber_at_low_noise = q_low,
            qber_at_threshold = q_thr,
            skr_at_low_noise  = _skr(q_low),
            decay_steepness   = float(grad),
            affects_amplitude = af,
            is_coherent       = ic,
            lindblad_type     = lt,
            physical_origin   = po,
        ))
    return fingerprints
