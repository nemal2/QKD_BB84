"""
bb84_config.py
══════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Single source of truth for all simulation parameters.
All future phases add fields here with safe defaults so that
existing Phase 1 code continues to run without changes.

Phase history
─────────────
Phase 1  :  n_qubits, eve_*, noise_enabled, depolar_prob,
            sample_fraction, seed, label
Phase 3  :  noise_model, t1_ns, t2_ns, gate_time_ns,
            channel_length_km          ← added this phase
Phase 4  :  (planned) attack_type, mean_photon_number, decoy_*
Phase 5  :  (planned) ecc_enabled, privacy_amp_enabled
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════
# SIMULATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """One complete BB84 simulation configuration."""

    # ── PHASE 1 ───────────────────────────────────────────────────────
    n_qubits:           int   = 1000
    """Total qubits Alice transmits."""

    eve_present:        bool  = False
    """True → Eve performs intercept-resend."""

    eve_intercept_prob: float = 1.0
    """Fraction of qubits Eve intercepts [0.0 – 1.0]."""

    noise_enabled:      bool  = False
    """True → apply channel noise (model chosen by noise_model)."""

    depolar_prob:       float = 0.01
    """Depolarizing error probability per gate. Used when noise_model='depolarizing'."""

    sample_fraction:    float = 0.10
    """Fraction of sifted key consumed for QBER estimation."""

    seed:     Optional[int]   = 42
    """RNG seed. None = random run, int = reproducible."""

    label:              str   = "Simulation"
    """Human-readable name used in plots and console output."""

    # ── PHASE 3 — Noise models ─────────────────────────────────────────
    noise_model:        str   = "depolarizing"
    """
    Which noise model to apply when noise_enabled=True.

    Choices
    ───────
    'depolarizing'   — uniform Pauli noise (Phase 1 default)
    'amplitude_damp' — T1 energy relaxation  (qubit decays |1⟩→|0⟩)
    'phase_damp'     — T2 pure dephasing     (phase coherence lost)
    'combined'       — T1 + T2 together      (most physically realistic)
    'fiber_loss'     — distance-based photon loss in fibre
    """

    t1_ns:              float = 100_000.0
    """
    T1 relaxation time in nanoseconds.
    Used by noise_model='amplitude_damp' and 'combined'.
    Typical superconducting qubit: 10 000 – 500 000 ns (10–500 µs).
    """

    t2_ns:              float = 50_000.0
    """
    T2 dephasing time in nanoseconds.
    Used by noise_model='phase_damp' and 'combined'.
    Constraint: T2 ≤ 2·T1 always (Bloch sphere).
    Typical superconducting qubit: 5 000 – 200 000 ns.
    """

    gate_time_ns:       float = 50.0
    """
    Single-qubit gate operation time in nanoseconds.
    Used to compute γ = 1 − exp(−gate_time_ns / T1_ns)  and
                    λ = 1 − exp(−gate_time_ns / T2_ns).
    Typical superconducting qubit: 20 – 100 ns.
    """

    channel_length_km:  float = 0.0
    """
    Fibre channel length in kilometres.
    Used only when noise_model='fiber_loss'.
    Standard SMF-28 fibre: 0.2 dB/km attenuation at 1550 nm.
    Photon survival: P = 10^(−0.2 × L / 10).
    """


# ══════════════════════════════════════════════════════════════════════
# QBER RESULT
# ══════════════════════════════════════════════════════════════════════

@dataclass
class QBERResult:
    """Output from estimate_qber()."""
    qber:            float   # Quantum Bit Error Rate  (0.0 – 1.0)
    errors:          int     # disagreements in the sample
    sample_size:     int     # bits consumed for estimation
    security_status: str     # 'SECURE ✓' | 'WARNING ⚠' | 'ABORT ✗'
    confidence_low:  float   # 95 % Wilson CI lower bound
    confidence_high: float   # 95 % Wilson CI upper bound


# ══════════════════════════════════════════════════════════════════════
# SIMULATION RESULT
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """Complete output of one run_simulation() call."""

    config:                SimulationConfig
    n_transmitted:         int
    n_sifted:              int
    sifted_key_rate:       float
    qber_result:           QBERResult
    alice_final_key:       List[int]
    bob_final_key:         List[int]
    key_agreement_rate:    float
    eve_interception_rate: float
    runtime_seconds:       float

    # Phase 4 fields (default 0.0 so Phase 1/3 results still construct fine)
    pns_multi_photon_rate: float = 0.0
    pns_detection_rate:    float = 0.0

    # Phase 5 fields
    post_processing:       Optional[object] = None
    security_analysis:     Optional[object] = None

    @property
    def key_length(self) -> int:
        return len(self.alice_final_key)

    @property
    def keys_match(self) -> bool:
        return self.alice_final_key == self.bob_final_key

    @property
    def key_generation_rate(self) -> float:
        return self.key_length / self.n_transmitted if self.n_transmitted else 0.0
