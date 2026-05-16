"""
bb84_noise.py  ·  Phase 3 — Quantum Channel Noise Models
══════════════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

PURPOSE
───────
Provides QuantumChannel, the physical model of the fibre between Alice
and Bob.  Phase 3 extends Phase 1's single depolarizing model with
physically motivated noise channels derived from the Lindblad master
equation.

LINDBLAD MASTER EQUATION  (the physics behind every noise model here)
──────────────────────────────────────────────────────────────────────
Open quantum systems — those coupled to an environment — evolve under:

    dρ/dt = −i/ℏ [H, ρ] + Σ_k γ_k ( L_k ρ L_k† − ½{L_k†L_k, ρ} )

where ρ is the qubit's density matrix and L_k are "jump operators"
describing each noise channel.  Each model below corresponds to a
specific choice of jump operators:

    Noise model       Jump operator(s)       Physical meaning
    ────────────────  ─────────────────────  ─────────────────────────────
    depolarizing      X, Y, Z  (equal γ)     Random Pauli errors
    amplitude_damp    |0⟩⟨1|  (γ = f(T1))   Energy relaxation (T1 decay)
    phase_damp        |1⟩⟨1|  (λ = f(T2))   Dephasing, coherence loss (T2)
    combined          both above             Realistic qubit decoherence
    fiber_loss        —                      Photon absorption / scattering

QISKIT MAPPING
──────────────
Qiskit Aer implements Lindblad noise via Kraus operators which are
mathematically equivalent to the master equation approach.

    depolarizing_error(p)          → p/4 · (I + X + Y + Z) Kraus map
    amplitude_damping_error(γ)     → Kraus K0=[[1,0],[0,√(1−γ)]], K1=[[0,√γ],[0,0]]
    phase_damping_error(λ)         → Kraus K0=[[1,0],[0,√(1−λ)]], K1=[[0,0],[0,√λ]]
    thermal_relaxation_error(T1,T2,t)  → full T1+T2 combined Kraus map

FIBER LOSS
──────────
Not a Lindblad channel but equally important for QKD.  Modelled as a
Bernoulli trial: each photon survives with probability
    P_survive = 10^(−α·L / 10)
where α = 0.2 dB/km (standard SMF-28 fibre at 1550 nm).
Lost photons return None from run_circuit(); the runner excludes those
indices from sifting.  Key rate drops but QBER is unaffected by loss
alone — this is the key educational insight.

EXPORTS
───────
QuantumChannel  — build one with QuantumChannel.from_config(cfg)
NoiseModelType  — string constants for the noise_model config field
══════════════════════════════════════════════════════════════════════
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

from bb84.config import SimulationConfig


# ──────────────────────────────────────────────────────────────────────
# Noise model name constants  (avoid magic strings in notebooks)
# ──────────────────────────────────────────────────────────────────────

class NoiseModelType:
    DEPOLARIZING    = "depolarizing"     # Phase 1 baseline
    AMPLITUDE_DAMP  = "amplitude_damp"   # T1 energy relaxation  (Phase 3)
    PHASE_DAMP      = "phase_damp"       # T2 pure dephasing     (Phase 3)
    COMBINED        = "combined"         # T1 + T2 together      (Phase 3)
    FIBER_LOSS      = "fiber_loss"       # distance-based loss   (Phase 3)


# ──────────────────────────────────────────────────────────────────────
# QUANTUM CHANNEL
# ──────────────────────────────────────────────────────────────────────

class QuantumChannel:
    """
    Physical quantum channel between Alice and Bob.

    Build with the factory:
        channel = QuantumChannel.from_config(cfg)

    Or directly for custom experiments:
        channel = QuantumChannel(noise_model="phase_damp", t2_ns=5000)

    run_circuit(qc) returns int (0 or 1) for all models except
    fiber_loss, which may return None (photon lost).
    Callers must handle None when using fiber_loss.
    """

    # SMF-28 standard fibre attenuation at 1550 nm  (dB/km)
    FIBER_ATTENUATION_DB_PER_KM: float = 0.2

    def __init__(
        self,
        noise_enabled:     bool  = False,
        noise_model:       str   = NoiseModelType.DEPOLARIZING,
        depolar_prob:      float = 0.01,
        t1_ns:             float = 100_000.0,
        t2_ns:             float =  50_000.0,
        gate_time_ns:      float =      50.0,
        channel_length_km: float =       0.0,
    ):
        self.noise_enabled     = noise_enabled
        self.noise_model       = noise_model
        self.depolar_prob      = depolar_prob
        self.t1_ns             = t1_ns
        self.t2_ns             = t2_ns
        self.gate_time_ns      = gate_time_ns
        self.channel_length_km = channel_length_km

        # Derived loss probability for fiber_loss model
        self._photon_survival_prob = self._calc_survival_prob()

        # Build Qiskit simulator (or None for fiber_loss — no gate noise)
        self._simulator = self._build_simulator()

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: SimulationConfig) -> "QuantumChannel":
        """Construct QuantumChannel directly from a SimulationConfig."""
        return cls(
            noise_enabled     = cfg.noise_enabled,
            noise_model       = cfg.noise_model,
            depolar_prob      = cfg.depolar_prob,
            t1_ns             = cfg.t1_ns,
            t2_ns             = cfg.t2_ns,
            gate_time_ns      = cfg.gate_time_ns,
            channel_length_km = cfg.channel_length_km,
        )

    # ── Public interface ──────────────────────────────────────────────

    def run_circuit(self, qc: QuantumCircuit, shot_seed: Optional[int] = None) -> Optional[int]:
        """
        Simulate a single-qubit measurement circuit through this channel.

        Returns
        ───────
        int   (0 or 1)  — measured bit value
        None            — photon lost (fiber_loss model only)

        shot_seed : per-measurement seed derived from global seed + qubit
                    index.  Gives reproducible results without the global
                    seed_simulator=12345 bug that collapsed all outcomes.
        """
        # Fiber loss: probabilistic photon survival BEFORE gate noise
        if (self.noise_enabled
                and self.noise_model == NoiseModelType.FIBER_LOSS):
            if random.random() > self._photon_survival_prob:
                return None        # photon absorbed / scattered in fibre

        # All other models: Qiskit simulation with noise
        run_kwargs = {"shots": 1}
        if shot_seed is not None:
            run_kwargs["seed_simulator"] = int(shot_seed) % (2**31)
        job    = self._simulator.run(transpile(qc, self._simulator), **run_kwargs)
        counts = job.result().get_counts()
        return int(list(counts.keys())[0])

    @property
    def description(self) -> str:
        """Human-readable summary for printing / notebook display."""
        if not self.noise_enabled:
            return "Ideal channel (no noise)"
        if self.noise_model == NoiseModelType.DEPOLARIZING:
            return f"Depolarizing  p = {self.depolar_prob}"
        if self.noise_model == NoiseModelType.AMPLITUDE_DAMP:
            γ = self._amplitude_damping_gamma()
            return f"Amplitude damping  T1 = {self.t1_ns/1000:.0f} µs  γ = {γ:.4f}"
        if self.noise_model == NoiseModelType.PHASE_DAMP:
            λ = self._phase_damping_lambda()
            return f"Phase damping  T2 = {self.t2_ns/1000:.0f} µs  λ = {λ:.4f}"
        if self.noise_model == NoiseModelType.COMBINED:
            return (f"Combined T1/T2  T1 = {self.t1_ns/1000:.0f} µs  "
                    f"T2 = {self.t2_ns/1000:.0f} µs")
        if self.noise_model == NoiseModelType.FIBER_LOSS:
            return (f"Fiber loss  L = {self.channel_length_km} km  "
                    f"P_survive = {self._photon_survival_prob:.3f}")
        return f"Unknown noise model: {self.noise_model}"

    # ── Private: parameter calculations ──────────────────────────────

    def _amplitude_damping_gamma(self) -> float:
        """
        Amplitude damping parameter γ from T1 and gate time.

        Physical meaning: probability that |1⟩ decays to |0⟩ per gate.
        Derived from Lindblad jump operator L = √(γ) |0⟩⟨1|.

            γ = 1 − exp(−t_gate / T1)

        Small t_gate/T1 → γ ≈ t_gate/T1  (linear regime, typical for superconducting qubits)
        """
        return 1.0 - math.exp(-self.gate_time_ns / self.t1_ns)

    def _phase_damping_lambda(self) -> float:
        """
        Phase damping parameter λ from T2 and gate time.

        Physical meaning: probability that phase coherence is lost per gate.
        Derived from Lindblad jump operator L = √(λ) |1⟩⟨1|.

            λ = 1 − exp(−t_gate / T2)

        Note: T2 ≤ 2·T1 always (Bloch sphere constraint).
        Phase damping alone does NOT change qubit populations (no |0⟩↔|1⟩ transitions).
        """
        return 1.0 - math.exp(-self.gate_time_ns / self.t2_ns)

    def _calc_survival_prob(self) -> float:
        """
        Photon survival probability for fiber_loss model.

            P_survive = 10^(−α · L / 10)

        α = 0.2 dB/km (SMF-28 at 1550 nm)
        L = channel_length_km
        """
        if self.channel_length_km <= 0:
            return 1.0
        db_loss = self.FIBER_ATTENUATION_DB_PER_KM * self.channel_length_km
        return 10 ** (-db_loss / 10)

    # ── Private: Qiskit simulator construction ────────────────────────

    def _build_simulator(self) -> AerSimulator:
        """
        Build an AerSimulator with the appropriate noise model.

        Gate targets: ['x', 'h', 'id'] — the only gates in Phase 1/3 circuits.
        Phase 4 will add 'cx' if multi-qubit circuits are introduced.
        """
        if not self.noise_enabled:
            return AerSimulator()

        if self.noise_model == NoiseModelType.FIBER_LOSS:
            # Fiber loss is handled probabilistically in run_circuit().
            # No gate-level noise model needed.
            return AerSimulator()

        nm = NoiseModel()
        gates = ["x", "h", "id"]

        # ── Depolarizing  (Phase 1 baseline) ──────────────────────────
        if self.noise_model == NoiseModelType.DEPOLARIZING:
            err = depolarizing_error(self.depolar_prob, 1)
            nm.add_all_qubit_quantum_error(err, gates)

        # ── Amplitude damping  (T1 energy relaxation) ─────────────────
        elif self.noise_model == NoiseModelType.AMPLITUDE_DAMP:
            γ = self._amplitude_damping_gamma()
            err = amplitude_damping_error(γ)
            nm.add_all_qubit_quantum_error(err, gates)

        # ── Phase damping  (T2 pure dephasing) ────────────────────────
        elif self.noise_model == NoiseModelType.PHASE_DAMP:
            λ = self._phase_damping_lambda()
            err = phase_damping_error(λ)
            nm.add_all_qubit_quantum_error(err, gates)

        # ── Combined T1 + T2  (most physically realistic) ─────────────
        elif self.noise_model == NoiseModelType.COMBINED:
            # thermal_relaxation_error implements the full Lindblad model
            # with both T1 (energy) and T2 (dephasing) simultaneously.
            # Constraint enforced: T2 cannot exceed 2·T1.
            t2_safe = min(self.t2_ns, 2.0 * self.t1_ns - 1.0)
            err = thermal_relaxation_error(
                t1=self.t1_ns,
                t2=t2_safe,
                time=self.gate_time_ns,
            )
            nm.add_all_qubit_quantum_error(err, gates)

        else:
            print(f"  [Channel] Unknown noise_model='{self.noise_model}', "
                  f"using ideal channel.")
            return AerSimulator()

        return AerSimulator(noise_model=nm)
