"""
bb84_noise.py
=============
Phase 3: physically-motivated quantum channel noise models for the BB84
QKD simulator, derived from the Lindblad master equation governing open
quantum systems (see report Sec. 2.2-2.4).

Five channel models
--------------------
IDEAL               - no noise (Phase 1 default)
DEPOLARIZING        - symmetric Pauli noise (Phase 1 baseline, kept for
                       backward compatibility)
AMPLITUDE_DAMPING   - T1 energy relaxation  (gamma = 1 - exp(-t_gate/T1))
PHASE_DAMPING       - T2 dephasing          (lambda = 1 - exp(-t_gate/T2))
COMBINED            - full T1+T2 thermal relaxation (most physical model
                       for superconducting hardware; T2 <= 2*T1 enforced)
FIBRE_LOSS          - distance-based photon loss, Beer-Lambert attenuation
                       P_survive = 10^(-alpha*L/10). NOT a Kraus channel:
                       it removes photons, it does not corrupt surviving
                       ones, so it degrades key RATE but not security.

Modelling note on Eve + fibre loss
-----------------------------------
Gate-level noise (depolarizing / amplitude / phase / combined) is baked
into the Aer noise model and therefore applies naturally on *every* hop
a qubit takes through the channel (Alice->Eve and Eve->Bob), since both
Eve.intercept() and Bob.measure() route through the same noisy simulator.

Fibre loss is different: it is evaluated explicitly inside run_circuit()
via a Bernoulli draw, not through Aer. To avoid double-attenuating a
single physical channel length when Eve is present, the loss draw is
only applied on the *final* hop into Bob's detector (apply_loss=True,
the default). Eve's own intercept measurement always calls
run_circuit(..., apply_loss=False), so the full channel length L is
attributed once, to the Alice/Eve -> Bob leg. This is a deliberate
simplification documented for future Phase 4 multi-hop work.

University of Ruhuna - Dept. of Computer Engineering
MIT Licence - see LICENSE
"""

from __future__ import annotations

import math
import random
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
)

from bb84_config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────
# NOISE MODEL TYPE CONSTANTS
# ──────────────────────────────────────────────────────────────────────

class NoiseModelType:
    """String constants identifying each Phase 3 channel model."""
    IDEAL             = "ideal"
    DEPOLARIZING      = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING     = "phase_damping"
    COMBINED          = "combined"
    FIBRE_LOSS        = "fibre_loss"

    ALL = (IDEAL, DEPOLARIZING, AMPLITUDE_DAMPING, PHASE_DAMPING, COMBINED, FIBRE_LOSS)

    LABELS = {
        IDEAL:             "Ideal (no noise)",
        DEPOLARIZING:      "Depolarising",
        AMPLITUDE_DAMPING: "Amplitude Damping (T1)",
        PHASE_DAMPING:     "Phase Damping (T2)",
        COMBINED:          "Combined T1+T2",
        FIBRE_LOSS:        "Fibre Loss",
    }


# Gates that ever appear in an Alice/Bob/Eve single-qubit circuit.
# Noise is attached to these so it is incurred once per gate actually used.
_GATE_NOISE_TARGETS = ["x", "h"]


# ──────────────────────────────────────────────────────────────────────
# QUANTUM CHANNEL
# ──────────────────────────────────────────────────────────────────────

class QuantumChannel:
    """
    Quantum channel backed by Qiskit Aer, supporting five physically
    motivated noise models built from Lindblad jump operators (Table 2
    of the Phase 3 report).

    Parameters
    ----------
    noise_model : str
        One of ``NoiseModelType.ALL``.  Unknown values fall back to the
        ideal simulator with a console warning (Sec. 3.3 design
        principle: unknown-model fallback, never a hard crash).
    depolar_prob : float
        Depolarising error probability per gate (DEPOLARIZING only).
    t1_ns, t2_ns : float
        T1 / T2 coherence times in nanoseconds (AMPLITUDE_DAMPING /
        PHASE_DAMPING / COMBINED).  T2 is clamped to <= 2*T1.
    gate_time_ns : float
        Single-qubit gate duration in nanoseconds.
    channel_length_km : float
        Fibre length for FIBRE_LOSS.
    fibre_attenuation_db_km : float
        Attenuation coefficient, default 0.2 dB/km (SMF-28 @ 1550 nm).
    loss_rng : random.Random, optional
        RNG used for the photon-survival Bernoulli draw when no
        per-shot seed is supplied to run_circuit().

    Example
    -------
    >>> ch = QuantumChannel(noise_model=NoiseModelType.AMPLITUDE_DAMPING,
    ...                      t1_ns=10_000, gate_time_ns=50)
    >>> bit = ch.run_circuit(qc)            # 0, 1
    >>> ch2 = QuantumChannel(noise_model=NoiseModelType.FIBRE_LOSS,
    ...                      channel_length_km=80)
    >>> bit2 = ch2.run_circuit(qc)          # 0, 1, or None (photon lost)
    """

    def __init__(
        self,
        noise_model: str = NoiseModelType.IDEAL,
        depolar_prob: float = 0.01,
        t1_ns: float = 10_000.0,
        t2_ns: float = 8_000.0,
        gate_time_ns: float = 50.0,
        channel_length_km: float = 0.0,
        fibre_attenuation_db_km: float = 0.2,
        loss_rng: Optional[random.Random] = None,
    ):
        if noise_model not in NoiseModelType.ALL:
            print(f"[bb84_noise] WARNING: unknown noise_model={noise_model!r}; "
                  f"falling back to ideal simulator.")
            noise_model = NoiseModelType.IDEAL

        self.noise_model = noise_model
        self.depolar_prob = depolar_prob

        # Bloch-sphere constraint: T2 <= 2*T1, always.
        self.t1_ns = float(t1_ns)
        self.t2_ns = float(min(t2_ns, 2 * t1_ns - 1e-6))
        self.gate_time_ns = float(gate_time_ns)

        self.channel_length_km = float(channel_length_km)
        self.alpha_db_km = float(fibre_attenuation_db_km)

        self._loss_rng = loss_rng if loss_rng is not None else random.Random()
        self._simulator = self._build_simulator()

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        loss_rng: Optional[random.Random] = None,
    ) -> "QuantumChannel":
        """
        Factory routing a SimulationConfig to the correct noise builder.

        Backward compatible: if ``config.noise_model`` is unset (None),
        legacy Phase 1 fields (``noise_enabled``, ``depolar_prob``)
        reproduce Phase 1 behaviour exactly.
        """
        model = config.noise_model
        if model is None:
            model = NoiseModelType.DEPOLARIZING if config.noise_enabled else NoiseModelType.IDEAL
        return cls(
            noise_model=model,
            depolar_prob=config.depolar_prob,
            t1_ns=config.t1_ns,
            t2_ns=config.t2_ns,
            gate_time_ns=config.gate_time_ns,
            channel_length_km=config.channel_length_km,
            loss_rng=loss_rng,
        )

    # ------------------------------------------------------------------
    # Derived physical quantities
    # ------------------------------------------------------------------
    @property
    def survival_probability(self) -> float:
        """Beer-Lambert photon survival probability, P = 10^(-alpha*L/10)."""
        return 10 ** (-self.alpha_db_km * self.channel_length_km / 10.0)

    @property
    def gamma(self) -> float:
        """Amplitude-damping parameter, gamma = 1 - exp(-t_gate / T1)."""
        return 1.0 - math.exp(-self.gate_time_ns / self.t1_ns)

    @property
    def lam(self) -> float:
        """Phase-damping parameter, lambda = 1 - exp(-t_gate / T2)."""
        return 1.0 - math.exp(-self.gate_time_ns / self.t2_ns)

    @property
    def description(self) -> str:
        """Short human-readable summary."""
        if self.noise_model == NoiseModelType.IDEAL:
            return "Ideal channel (no noise)"
        if self.noise_model == NoiseModelType.DEPOLARIZING:
            return f"Depolarising  p = {self.depolar_prob}"
        if self.noise_model == NoiseModelType.AMPLITUDE_DAMPING:
            return f"Amplitude damping  T1={self.t1_ns/1000:.3f} us  (gamma={self.gamma:.5f})"
        if self.noise_model == NoiseModelType.PHASE_DAMPING:
            return f"Phase damping  T2={self.t2_ns/1000:.3f} us  (lambda={self.lam:.5f})"
        if self.noise_model == NoiseModelType.COMBINED:
            return (f"Combined T1+T2  T1={self.t1_ns/1000:.3f} us  "
                    f"T2={self.t2_ns/1000:.3f} us  t_gate={self.gate_time_ns:.0f} ns")
        if self.noise_model == NoiseModelType.FIBRE_LOSS:
            return (f"Fibre loss  L={self.channel_length_km:.1f} km  "
                    f"(P_survive={self.survival_probability:.4f})")
        return "Unknown channel"

    # ------------------------------------------------------------------
    def run_circuit(
        self,
        qc: QuantumCircuit,
        shot_seed: Optional[int] = None,
        apply_loss: bool = True,
    ) -> Optional[int]:
        """
        Simulate *qc* through this channel and return the measured bit.

        Parameters
        ----------
        qc         : single-qubit circuit with one measurement.
        shot_seed  : per-shot seed for reproducibility.
        apply_loss : whether the FIBRE_LOSS Bernoulli draw applies on
                     this call.  Bob.measure() always uses the default
                     (True); Eve.intercept() passes False so a single
                     channel length is not attenuated twice (see module
                     docstring).

        Returns
        -------
        int or None
            0 or 1 for a successfully measured qubit; None if the
            photon was lost in transit (FIBRE_LOSS model, apply_loss=True
            calls only).
        """
        if self.noise_model == NoiseModelType.FIBRE_LOSS and apply_loss:
            draw = (random.Random(shot_seed).random()
                    if shot_seed is not None else self._loss_rng.random())
            if draw > self.survival_probability:
                return None  # photon lost - Bob detects nothing

        kwargs = {"shots": 1}
        if shot_seed is not None:
            kwargs["seed_simulator"] = int(shot_seed) % (2 ** 31)
        # NOTE: no transpile() here. x / h / measure / id are already
        # native AerSimulator basis gates, so transpiling is a no-op that
        # costs ~90 ms per call (measured) for zero benefit - it dominated
        # total runtime and made N >= 2000 sweeps impractical. Skipping it
        # gives an ~180x speedup with byte-identical results.
        job = self._simulator.run(qc, **kwargs)
        counts = job.result().get_counts()
        return int(list(counts.keys())[0])

    # ------------------------------------------------------------------
    def _build_simulator(self) -> AerSimulator:
        if self.noise_model in (NoiseModelType.IDEAL, NoiseModelType.FIBRE_LOSS):
            return AerSimulator()

        nm = NoiseModel()

        if self.noise_model == NoiseModelType.DEPOLARIZING:
            err = depolarizing_error(self.depolar_prob, 1)
            nm.add_all_qubit_quantum_error(err, ["x", "h", "id"])

        elif self.noise_model == NoiseModelType.AMPLITUDE_DAMPING:
            err = amplitude_damping_error(self.gamma)
            nm.add_all_qubit_quantum_error(err, _GATE_NOISE_TARGETS)

        elif self.noise_model == NoiseModelType.PHASE_DAMPING:
            err = phase_damping_error(self.lam)
            nm.add_all_qubit_quantum_error(err, _GATE_NOISE_TARGETS)

        elif self.noise_model == NoiseModelType.COMBINED:
            err = thermal_relaxation_error(self.t1_ns, self.t2_ns, self.gate_time_ns)
            nm.add_all_qubit_quantum_error(err, _GATE_NOISE_TARGETS)

        return AerSimulator(noise_model=nm)