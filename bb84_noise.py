"""
bb84_noise.py  ·  Phase 3 — Advanced Noise Models
══════════════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Implements four physically motivated quantum channel noise models:

  1. Depolarizing  — uniform Pauli error (Phase 1 baseline)
  2. Amplitude Damping — T1 energy relaxation (spontaneous emission)
  3. Phase Damping    — T2 pure dephasing (quantum memory decoherence)
  4. Fiber Loss       — distance-based photon loss (Beer-Lambert)
  5. Combined         — depolarizing + phase damping simultaneously

PHYSICS BACKGROUND
──────────────────
Amplitude damping (T1):
  Models energy relaxation.  The qubit decays from |1⟩ → |0⟩ with
  probability  p_amp = 1 − exp(−t_gate / T1).
  Kraus operators:  K0 = [[1, 0], [0, √(1−γ)]],  K1 = [[0, √γ], [0, 0]]

Phase damping (T2):
  Models pure dephasing — loss of quantum phase coherence without
  energy exchange.  λ = 1 − exp(−t_gate / T2).
  Kraus operators:  K0 = [[1, 0], [0, √(1−λ)]],  K1 = [[0, 0], [0, √λ]]

Fiber loss (Beer-Lambert):
  Photon survival probability  η = 10^(−α·L / 10)
  where α ≈ 0.2 dB/km for standard SMF-28 at 1550 nm, L = distance.
  Lost photons are modelled as erasures (Bob gets no signal).

NOTE ON FUTURE PHASES
─────────────────────
Phase 3 only adds noise models to the channel.
Phase 4 adds attack models (PNS, entanglement).
Phase 5 adds post-processing (error correction, privacy amp).
These modules do NOT depend on each other for imports.
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    pauli_error,
)


# ══════════════════════════════════════════════════════════════════════
# FIBER LOSS UTILITIES
# ══════════════════════════════════════════════════════════════════════

#: Standard SMF-28 attenuation at 1550 nm (dB/km)
FIBER_ATTENUATION_DB_PER_KM: float = 0.2

#: Detector efficiency (fraction of arriving photons detected)
DETECTOR_EFFICIENCY: float = 0.85


def fiber_transmittance(
    length_km:          float,
    attenuation_db_per_km: float = FIBER_ATTENUATION_DB_PER_KM,
    detector_efficiency:   float = DETECTOR_EFFICIENCY,
) -> float:
    """
    Compute end-to-end transmittance η of a fiber channel.

    η = detector_efficiency × 10^(−α·L / 10)

    Parameters
    ──────────
    length_km            : channel length in km
    attenuation_db_per_km: fiber attenuation coefficient (default 0.2 dB/km)
    detector_efficiency  : Bob's detector efficiency (default 0.85)

    Returns
    ───────
    η ∈ [0.0, 1.0]  — probability a photon survives and is detected
    """
    fiber_loss_factor = 10 ** (-attenuation_db_per_km * length_km / 10)
    return detector_efficiency * fiber_loss_factor


def max_secure_distance(
    mu:                    float = 0.5,
    attenuation_db_per_km: float = FIBER_ATTENUATION_DB_PER_KM,
    detector_efficiency:   float = DETECTOR_EFFICIENCY,
    qber_threshold:        float = 0.11,
) -> float:
    """
    Estimate the maximum secure distance for BB84 over optical fiber.

    At distance L, the effective QBER due to dark counts is approximated.
    This function solves for L where SKR ≈ 0.

    Returns
    ───────
    max_km : approximate maximum secure range in km
    """
    # Simple estimate: transmittance must exceed a minimum threshold
    # derived from the requirement that SKR > 0
    # SKR ≈ Q1 * (1 - 2*H(e1)) > 0  →  e1 < 11%
    # At long distances, dark count rate Y0 dominates QBER
    Y0 = 1.7e-6  # dark count probability per pulse (typical)

    # Binary search for max distance
    lo, hi = 0.0, 1000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        eta = fiber_transmittance(mid, attenuation_db_per_km, detector_efficiency)
        Q_mu = 1 - math.exp(-mu * eta)  # gain (approximate)
        Q_mu = max(Q_mu, Y0)
        e_mu = (0.5 * Y0 + 0.25 * mu * eta * math.exp(-mu)) / max(Q_mu, 1e-12)
        if e_mu < qber_threshold:
            lo = mid
        else:
            hi = mid

    return lo


# ══════════════════════════════════════════════════════════════════════
# QUANTUM CHANNEL WITH PLUGGABLE NOISE MODEL
# ══════════════════════════════════════════════════════════════════════

class QuantumChannel:
    """
    Unified quantum channel supporting four noise models.

    Usage
    ─────
    channel = QuantumChannel.from_config(config)

    # or directly:
    channel = QuantumChannel(noise_model='amplitude_damp',
                             t1_ns=50_000, gate_time_ns=50)
    result  = channel.run_circuit(qc)   # returns int or None

    Return value of run_circuit()
    ─────────────────────────────
    int  (0 or 1) : qubit was detected and measured
    None          : photon was lost in fiber (fiber_loss model only)

    Callers (bb84_runner.py) must handle None as a lost qubit.

    Phase 4 note
    ────────────
    PNS attack also introduces photon loss.  bb84_attacks.py
    handles that at a higher level, independent of this class.
    """

    # Gates affected by noise in each model
    _NOISY_GATES = ["x", "h", "id", "u1", "u2", "u3"]

    def __init__(
        self,
        noise_model:    str   = "depolarizing",
        depolar_prob:   float = 0.0,
        t1_ns:          float = 100_000.0,
        t2_ns:          float = 50_000.0,
        gate_time_ns:   float = 50.0,
        channel_length_km: float = 0.0,
    ):
        self.noise_model_name = noise_model
        self.depolar_prob     = depolar_prob
        self.t1_ns            = t1_ns
        self.t2_ns            = t2_ns
        self.gate_time_ns     = gate_time_ns
        self.channel_length_km = channel_length_km

        self._transmittance = (
            fiber_transmittance(channel_length_km)
            if noise_model == "fiber_loss" else 1.0
        )
        self._rng       = np.random.default_rng()
        self._simulator = self._build_simulator()

        if noise_model != "depolarizing" or depolar_prob > 0:
            print(f"  [Channel] Model = '{noise_model}'"
                  + self._param_str())

    @classmethod
    def from_config(cls, config) -> "QuantumChannel":
        """Convenience constructor from a SimulationConfig object."""
        return cls(
            noise_model=config.noise_model if config.noise_enabled else "ideal",
            depolar_prob=config.depolar_prob,
            t1_ns=config.t1_ns,
            t2_ns=config.t2_ns,
            gate_time_ns=config.gate_time_ns,
            channel_length_km=config.channel_length_km,
        )

    # ── Simulator construction ─────────────────────────────────────────

    def _build_simulator(self) -> AerSimulator:
        nm = self.noise_model_name

        if nm in ("ideal", "fiber_loss"):
            return AerSimulator()  # pure loss handled in run_circuit()

        qiskit_noise = NoiseModel()

        if nm == "depolarizing" and self.depolar_prob > 0:
            err = depolarizing_error(self.depolar_prob, 1)
            qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)

        elif nm == "amplitude_damp":
            p_amp = 1.0 - math.exp(-self.gate_time_ns / self.t1_ns)
            p_amp = min(max(p_amp, 0.0), 1.0)
            err = amplitude_damping_error(p_amp)
            qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)

        elif nm == "phase_damp":
            lam = 1.0 - math.exp(-self.gate_time_ns / self.t2_ns)
            lam = min(max(lam, 0.0), 1.0)
            err = phase_damping_error(lam)
            qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)

        elif nm == "combined":
            p_amp = 1.0 - math.exp(-self.gate_time_ns / self.t1_ns)
            lam   = 1.0 - math.exp(-self.gate_time_ns / self.t2_ns)
            err_a = amplitude_damping_error(min(p_amp, 1.0))
            err_p = phase_damping_error(min(lam, 1.0))
            err   = err_a.compose(err_p)
            qiskit_noise.add_all_qubit_quantum_error(err, self._NOISY_GATES)

        elif nm == "depolarizing" and self.depolar_prob == 0:
            return AerSimulator()  # effectively ideal

        return AerSimulator(noise_model=qiskit_noise)

    def _param_str(self) -> str:
        nm = self.noise_model_name
        if nm == "depolarizing":
            return f"  p = {self.depolar_prob}"
        if nm == "amplitude_damp":
            p = 1.0 - math.exp(-self.gate_time_ns / self.t1_ns)
            return f"  T1 = {self.t1_ns:.0f} ns  →  p_amp = {p:.4f}"
        if nm == "phase_damp":
            lam = 1.0 - math.exp(-self.gate_time_ns / self.t2_ns)
            return f"  T2 = {self.t2_ns:.0f} ns  →  λ = {lam:.4f}"
        if nm == "fiber_loss":
            return (f"  L = {self.channel_length_km} km  "
                    f"→  η = {self._transmittance:.4f}")
        if nm == "combined":
            return "  (amplitude damp + phase damp)"
        return ""

    # ── Public interface ───────────────────────────────────────────────

    def run_circuit(self, qc: QuantumCircuit) -> Optional[int]:
        """
        Simulate a single-qubit measurement circuit through this channel.

        Returns
        ───────
        int  (0 or 1)  : measured bit
        None           : photon lost in fiber channel
        """
        # Fiber loss: probabilistic qubit erasure before measurement
        if self.noise_model_name == "fiber_loss":
            if self._rng.random() > self._transmittance:
                return None   # photon lost

        job    = self._simulator.run(transpile(qc, self._simulator), shots=1)
        counts = job.result().get_counts()
        return int(list(counts.keys())[0])

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def effective_error_probability(self) -> float:
        """
        Approximate single-qubit error probability for this channel.
        Used by analysis functions to predict QBER contribution.
        """
        nm = self.noise_model_name
        if nm == "depolarizing":
            return self.depolar_prob * 3 / 4   # depolarizing → effective bit-flip
        if nm == "amplitude_damp":
            return 1.0 - math.exp(-self.gate_time_ns / self.t1_ns)
        if nm == "phase_damp":
            lam = 1.0 - math.exp(-self.gate_time_ns / self.t2_ns)
            return lam / 2
        if nm == "fiber_loss":
            return 1.0 - self._transmittance
        if nm == "combined":
            p_a = 1.0 - math.exp(-self.gate_time_ns / self.t1_ns)
            lam = 1.0 - math.exp(-self.gate_time_ns / self.t2_ns)
            return 1.0 - (1.0 - p_a) * (1.0 - lam / 2)
        return 0.0

    @property
    def transmittance(self) -> float:
        """Channel transmittance η (1.0 for non-fiber models)."""
        return self._transmittance


# ══════════════════════════════════════════════════════════════════════
# ANALYTICAL NOISE BENCHMARKS  (no Qiskit needed)
# ══════════════════════════════════════════════════════════════════════

def theoretical_qber_from_noise(noise_model: str, **kwargs) -> float:
    """
    Compute the theoretical QBER contribution from channel noise alone.

    This is the analytical prediction; compare with simulated QBER
    to verify the noise model implementation.

    Parameters (keyword)
    ─────────────────────
    noise_model = 'depolarizing' : p (float) → depolar probability
    noise_model = 'amplitude_damp': t1_ns, gate_time_ns
    noise_model = 'phase_damp'   : t2_ns, gate_time_ns
    noise_model = 'fiber_loss'   : channel_length_km

    Returns
    ───────
    QBER contribution from noise alone (0.0 – 0.5)
    """
    if noise_model == "depolarizing":
        p = kwargs.get("p", 0.0)
        return p * 3 / 4

    if noise_model == "amplitude_damp":
        t1  = kwargs.get("t1_ns", 100_000.0)
        tg  = kwargs.get("gate_time_ns", 50.0)
        return (1.0 - math.exp(-tg / t1)) / 2

    if noise_model == "phase_damp":
        t2  = kwargs.get("t2_ns", 50_000.0)
        tg  = kwargs.get("gate_time_ns", 50.0)
        lam = 1.0 - math.exp(-tg / t2)
        return lam / 4

    if noise_model == "fiber_loss":
        L   = kwargs.get("channel_length_km", 0.0)
        eta = fiber_transmittance(L)
        # Dark-count contribution to QBER at long distances
        Y0  = 1.7e-6
        mu  = kwargs.get("mu", 0.5)
        Q   = max(1 - math.exp(-mu * eta), Y0)
        return min(0.5 * Y0 / Q, 0.5)

    return 0.0
