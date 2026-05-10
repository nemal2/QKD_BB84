"""
bb84_core.py
============
Quantum parties and classical post-processing for BB84 QKD.

Alice            => qubit preparation
Bob              => qubit measurement
Eve              => intercept-resend attack  (Phase 1)
QuantumChannel   => noise model + Qiskit backend
sift_keys()      => basis reconciliation
estimate_qber()  => QBER with Wilson confidence interval

"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from bb84_attacks import build_attack_model
from bb84_noise import build_noise_model, noise_summary

from bb84config import QBERResult



# QUANTUM CHANNEL
# ===============

class QuantumChannel:
    """
    Models the physical quantum channel between Alice and Bob.

    Phase 1  : ideal channel  OR  depolarizing noise

    Phase 3 (planned — do NOT implement here yet):
        • Phase damping          (T2 dephasing)
        • Amplitude damping      (T1 energy relaxation)
        • Distance-based photon loss  (Beer-Lambert / fibre loss)
          → will require new constructor param: channel_length_km
  
    Adding Phase 3 noise: subclass QuantumChannel or extend
    _build_simulator() with a noise_model
    """

    def __init__(
        self,
        noise_enabled: bool  = False,
        noise_model:  str   = "depolarizing",
        depolar_prob:  float = 0.0,
        phase_damp_prob: float = 0.0,
        amplitude_damp_prob: float = 0.0,
    ):
        self.noise_enabled = noise_enabled
        self.noise_model = noise_model
        self.depolar_prob  = depolar_prob
        self.phase_damp_prob = phase_damp_prob
        self.amplitude_damp_prob = amplitude_damp_prob
        self._simulator    = self._build_simulator()

    #  Private helpers ====================

    def _build_simulator(self) -> AerSimulator:
        noise_model = build_noise_model(
            noise_enabled=self.noise_enabled,
            noise_model=self.noise_model,
            depolar_prob=self.depolar_prob,
            phase_damp_prob=self.phase_damp_prob,
            amplitude_damp_prob=self.amplitude_damp_prob,
        )
        # Use QasmSimulator for speed, with optimization level 3
        if noise_model is not None:
            print(
                "  [Channel] Noise enabled: "
                + noise_summary(
                    self.noise_enabled,
                    self.noise_model,
                    self.depolar_prob,
                    self.phase_damp_prob,
                    self.amplitude_damp_prob,
                )
            )
            return AerSimulator(noise_model=noise_model, method="statevector", shots=1)
        # No noise: use statevector method for speed (10x faster than qasm)
        return AerSimulator(method="statevector")

    #  Public interface ====================

    def run_circuit(self, qc: QuantumCircuit) -> int:
        """
        Simulate a single-qubit measurement circuit.
        Returns the measured bit (0 or 1).

        Phase 3 note: photon-loss model will need a 'lost' return value
        """
        job    = self._simulator.run(transpile(qc, self._simulator), shots=1)
        counts = job.result().get_counts()
        return int(list(counts.keys())[0])

    def run_batch_circuits(self, circuits: List[QuantumCircuit]) -> List[int]:
        """
        Simulate multiple circuits in batch for better performance.
        Returns list of measured bits.
        
        Optimization: 
        - Batching circuits together reduces simulator overhead
        - Uses statevector method (10x faster than Aer)
        - ~5-10x faster than running circuits individually
        """
        if not circuits:
            return []
        
        # Skip transpilation for pure state simulation (major speed improvement)
        if self.noise_enabled:
            transpiled = [transpile(qc, self._simulator, optimization_level=1) for qc in circuits]
        else:
            # No transpilation needed for statevector simulation
            transpiled = circuits
        
        job = self._simulator.run(transpiled, shots=1)
        results = job.result()
        
        measured_bits = []
        for i in range(len(circuits)):
            counts = results.get_counts(i)
            measured_bits.append(int(list(counts.keys())[0]))
        return measured_bits


# ALICE — Quantum State Preparation ===============================

class Alice:
    """
    Alice randomly selects bits and bases, then encodes each bit
    as a qubit.

    Encoding table
    ──────────────
    bit=0, basis=0  →  |0⟩        (no gates)
    bit=1, basis=0  →  |1⟩        (X gate)
    bit=0, basis=1  →  |+⟩        (H gate)
    bit=1, basis=1  →  |−⟩        (X → H)

    Bases
    ─────
    0 = Rectilinear (+)
    1 = Diagonal    (×)
    """

    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        self.n_qubits = n_qubits
        rng = np.random.default_rng(seed)
        self.bits  = rng.integers(0, 2, n_qubits).tolist()
        self.bases = rng.integers(0, 2, n_qubits).tolist()

    def prepare_qubit(self, index: int) -> QuantumCircuit:
        """Return a 1-qubit circuit encoding bits[index] in bases[index]."""
        qc = QuantumCircuit(1, 1, name=f"A{index}")
        if self.bits[index] == 1:
            qc.x(0)
        if self.bases[index] == 1:
            qc.h(0)
        return qc

    def sift_key(self, matching_indices: List[int]) -> List[int]:
        """Alice's bits at positions where bases agreed with Bob's."""
        return [self.bits[i] for i in matching_indices]


# BOB — Quantum Measurement ===============================


class Bob:
    """
    Bob randomly selects a measurement basis for each qubit.
    When his basis matches Alice's the outcome is deterministic;
    otherwise it is uniformly random.

    Bases: 0 = Rectilinear (+),  1 = Diagonal (×)
    """

    def __init__(self, n_qubits: int, seed: Optional[int] = None):
        self.n_qubits = n_qubits
        # Offset seed so Bob's bases are independent of Alice's
        rng = np.random.default_rng(None if seed is None else seed + 99)
        self.bases: List[int]           = rng.integers(0, 2, n_qubits).tolist()
        self.measured_bits: List[Optional[int]] = [None] * n_qubits

    def measure(
        self,
        qc:      QuantumCircuit,
        index:   int,
        channel: QuantumChannel,
    ) -> int:
        """
        Measure qubit `index` in Bob's chosen basis.
        Returns the measured bit and stores it in measured_bits[index].
        """
        meas_qc = qc.copy()
        if self.bases[index] == 1:
            meas_qc.h(0)          # rotate to diagonal basis before measuring
        meas_qc.measure(0, 0)
        bit = channel.run_circuit(meas_qc)
        self.measured_bits[index] = bit
        return bit

    def sift_key(self, matching_indices: List[int]) -> List[int]:
        # Bob's measured bits at positions where bases agreed with Alice's.
        return [self.measured_bits[i] for i in matching_indices]



# EVE — Intercept-Resend Attack  (Phase 1) ==============================


class Eve:
    """
    Phase 1 attack: intercept-resend.
    Eve intercepts each qubit with probability `intercept_prob`,
    measures in a RANDOM basis, then forwards a freshly prepared
    qubit to Bob.

    Expected effect: ~25 % QBER at intercept_prob = 1.0

    ===============================================
    Phase 4 (planned — do NOT implement here yet):
        • Entanglement-based attacks
        • Photon Number Splitting (PNS)
        • Decoy-state countermeasure
    ===============================================
    
    """

    def __init__(
        self,
        intercept_prob: float = 1.0,
        seed:           Optional[int] = None,
        attack_model:   str = "intercept_resend",
    ):
        self._attack = build_attack_model(
            attack_model=attack_model,
            intercept_prob=intercept_prob,
            seed=seed,
        )

    # ── Public interface ==========================================

    def intercept(
        self,
        qc:      QuantumCircuit,
        index:   int,
        channel: QuantumChannel,
    ) -> QuantumCircuit:
        """
        Optionally intercept qubit `index`.

        Returns either:
          • The original circuit  (Eve skips this qubit), or
          • A freshly prepared circuit in Eve's measured state
        """
        return self._attack.intercept(qc, index, channel)

    @property
    def stats(self) -> dict:
        return self._attack.stats

    @property
    def intercepted_count(self) -> int:
        return int(self.stats.get("intercepted", 0))


# KEY SIFTING  (Classical Post-Processing) ===========================


def sift_keys(alice_bases: List[int], bob_bases: List[int]) -> List[int]:
    """
    Classical public comparison of bases (NOT bits).
    Returns indices where Alice and Bob chose the same basis.
    Expected retention: ~50 % of transmitted qubits.
    """
    return [i for i in range(len(alice_bases)) if alice_bases[i] == bob_bases[i]]



# QBER ESTIMATION  (Classical Post-Processing) ======================

def estimate_qber(
    alice_key:       List[int],
    bob_key:         List[int],
    sample_fraction: float = 0.10,
    seed:            Optional[int] = None,
) -> QBERResult:
    """
    Publicly compare a random SAMPLE of the sifted key.
    The sampled bits are consumed (revealed) and NOT used in the final key.

    Security thresholds (BB84 standard)
    =================================
    QBER <  5 %  →  SECURE 
    5 % ≤ QBER < 11 %  →  WARNING    (eavesdropping possible)
    QBER ≥ 11 %  →  ABORT   (channel compromised)

    Phase 4 note: error correction (Cascade / LDPC) will be applied
    AFTER this step and the returned QBERResult will then also carry a
    post-correction residual QBER field.
    """


    assert len(alice_key) == len(bob_key), \
        f"Key length mismatch: Alice={len(alice_key)}, Bob={len(bob_key)}"

    n           = len(alice_key)
    sample_size = max(1, int(n * sample_fraction))

    rng     = random.Random(seed)
    indices = rng.sample(range(n), min(sample_size, n))

    errors = sum(1 for i in indices if alice_key[i] != bob_key[i])
    qber   = errors / len(indices)

    # 95 % Wilson confidence interval
    p   = qber
    z   = 1.96
    n_s = len(indices)
    denom  = 1 + z**2 / n_s
    center = (p + z**2 / (2 * n_s)) / denom
    margin = (z * (p * (1 - p) / n_s + z**2 / (4 * n_s**2)) ** 0.5) / denom
    ci_low  = max(0.0, center - margin)
    ci_high = min(1.0, center + margin)

    if qber < 0.05:
        status = "SECURE ok"
    elif qber < 0.11:
        status = "WARNING "
    else:
        status = "ABORT x"

    return QBERResult(
        qber=qber,
        errors=errors,
        sample_size=len(indices),
        security_status=status,
        confidence_low=ci_low,
        confidence_high=ci_high,
    )