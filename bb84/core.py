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
from typing import Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from bb84.config import QBERResult
from bb84.noise import QuantumChannel   # Phase 3: use the real noise-aware channel



# QuantumChannel imported from bb84_noise (Phase 3) — see import above


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
        # Per-shot seed base: unique prime multiple so Bob's shots don't
        # collide with Eve's shot seeds even at the same qubit index.
        self._shot_seed_base = (seed * 7919) if seed is not None else None

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
        shot_seed = (self._shot_seed_base + index) if self._shot_seed_base is not None else None
        bit = channel.run_circuit(meas_qc, shot_seed=shot_seed)
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
    ):
        self.intercept_prob    = intercept_prob
        self._rng = np.random.default_rng(None if seed is None else seed + 200)
        self.intercepted_count = 0
        self._eve_bases:  Dict[int, int] = {}
        self._eve_bits:   Dict[int, int] = {}
        # Different prime from Bob's so Eve's shot seeds never collide.
        self._shot_seed_base = (seed * 6271) if seed is not None else None

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
        if random.random() > self.intercept_prob:
            return qc                          # Eve passes this qubit through

        self.intercepted_count += 1

        # ── Step 1: Pick a random basis ────────────────────────────────
        eve_base = int(self._rng.integers(0, 2))
        self._eve_bases[index] = eve_base

        # ── Step 2: Measure Alice's qubit in Eve's basis ───────────────
        meas_qc = qc.copy()
        if eve_base == 1:
            meas_qc.h(0)
        meas_qc.measure(0, 0)
        shot_seed = (self._shot_seed_base + index) if self._shot_seed_base is not None else None
        measured_bit = channel.run_circuit(meas_qc, shot_seed=shot_seed)
        self._eve_bits[index] = measured_bit

        # ── Step 3: Re-prepare qubit from Eve's result ─────────────────
        new_qc = QuantumCircuit(1, 1, name=f"E{index}")
        if measured_bit == 1:
            new_qc.x(0)
        if eve_base == 1:
            new_qc.h(0)

        return new_qc                          # Bob receives this, not Alice's

    @property
    def stats(self) -> dict:
        return {
            "intercepted": self.intercepted_count,
            "bases":       self._eve_bases,
            "bits":        self._eve_bits,
        }


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
    indices = rng.sample(range(n), min(sample_size, n)) if n > 0 else []

    # No sifted bits survived (e.g. extreme fibre loss / heavy noise) → there is
    # no key to test, so QBER is undefined. Return a safe "no key" result instead
    # of dividing by zero.
    if not indices:
        return QBERResult(
            qber=0.0,
            errors=0,
            sample_size=0,
            security_status="ABORT x",
            confidence_low=0.0,
            confidence_high=0.0,
        )

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