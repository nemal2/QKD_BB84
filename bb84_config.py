"""
bb84_config.py  ·  BB84 QKD Research Platform
══════════════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering
Research Platform: BB84 Quantum Key Distribution Simulator

PURPOSE
───────
Single source of truth for every parameter across ALL phases.
Future phases ADD fields here; they NEVER break existing defaults.
Each field documents which phase activates it.

MODIFICATION GUIDE
──────────────────
Adding Phase N parameters:
  1. Add a new field with a safe default (so Phase 1 runs unchanged)
  2. Add a comment # PHASE N
  3. Document it in the field's docstring or inline comment
  4. Add a hook comment in bb84_runner.py at the right pipeline step
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════
# SIMULATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """
    Central control panel for one complete BB84 simulation.

    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 1  —  Baseline BB84  (active)                           │
    ├─────────────────────────────────────────────────────────────────┤
    │  PHASE 3  —  Advanced Noise Models  (active)                   │
    │  PHASE 4  —  Advanced Attacks & Decoy State  (active)          │
    │  PHASE 5  —  Error Correction + Privacy Amplification  (active)│
    └─────────────────────────────────────────────────────────────────┘
    """

    # ── PHASE 1 — Baseline ────────────────────────────────────────────
    n_qubits:            int   = 1000
    """Total qubits Alice transmits in one simulation run."""

    eve_present:         bool  = False
    """True → Eve performs an intercept-resend attack (Phase 1)."""

    eve_intercept_prob:  float = 1.0
    """Fraction of qubits Eve intercepts [0.0 – 1.0]. Phase 1."""

    noise_enabled:       bool  = False
    """True → apply depolarizing channel noise. Phase 1 baseline."""

    depolar_prob:        float = 0.01
    """Depolarizing error probability per gate. Phase 1."""

    sample_fraction:     float = 0.10
    """Fraction of sifted key consumed for QBER estimation."""

    seed:      Optional[int]   = 42
    """RNG seed. None = non-reproducible, int = reproducible run."""

    label:               str   = "Simulation"
    """Human-readable run name used in plots and reports."""

    # ── PHASE 3 — Advanced Noise Models ──────────────────────────────
    noise_model:         str   = "depolarizing"
    """
    Noise model to apply to the quantum channel.
      'depolarizing'    — uniform Pauli noise (Phase 1 baseline)
      'amplitude_damp'  — T1 energy relaxation  (Phase 3)
      'phase_damp'      — T2 pure dephasing     (Phase 3)
      'fiber_loss'      — distance-based photon loss  (Phase 3)
      'combined'        — depolarizing + phase damp   (Phase 3)
    """

    channel_length_km:   float = 0.0
    """
    Fiber channel length in kilometres.
    Active when noise_model='fiber_loss'. Phase 3.
    Standard SMF-28: 0.2 dB/km at 1550 nm.
    """

    t1_ns:               float = 100_000.0
    """T1 relaxation time in nanoseconds (amplitude damping). Phase 3."""

    t2_ns:               float = 50_000.0
    """T2 dephasing time in nanoseconds (phase damping). Phase 3."""

    gate_time_ns:        float = 50.0
    """Gate operation time in nanoseconds. Used with T1/T2. Phase 3."""

    # ── PHASE 4 — Advanced Attacks & Decoy State ─────────────────────
    attack_type:         str   = "intercept_resend"
    """
    Eavesdropping attack model.
      'intercept_resend' — random-basis intercept-resend (Phase 1)
      'pns'              — Photon Number Splitting       (Phase 4)
      'entanglement'     — Entanglement-based attack     (Phase 4)
    """

    mean_photon_number:  float = 0.5
    """
    Mean photon number μ of weak coherent pulses (WCP).
    Used by PNS attack model and decoy state protocol. Phase 4.
    Typical values: 0.1 – 0.5 photons/pulse.
    """

    decoy_state_enabled: bool  = False
    """
    True → apply 3-intensity decoy state protocol to bound
    single-photon gain and error rate. Phase 4.
    """

    mu_signal:           float = 0.5
    """Signal state intensity (μ_s) for decoy protocol. Phase 4."""

    mu_decoy:            float = 0.1
    """Decoy state intensity (μ_d) for decoy protocol. Phase 4."""

    mu_vacuum:           float = 0.0
    """Vacuum state intensity for decoy protocol. Phase 4."""

    pns_strategy:        str   = "block_single"
    """
    PNS attack strategy. Phase 4.
      'block_single'     — Eve blocks all single-photon pulses
                           (zero QBER increase, detected by rate drop)
      'intercept_single' — Eve intercept-resends single-photon pulses
                           (QBER increase, harder to attribute)
    """

    # ── PHASE 5 — Error Correction + Privacy Amplification ───────────
    ecc_enabled:         bool  = False
    """True → apply error correction to the sifted key. Phase 5."""

    ecc_scheme:          str   = "cascade"
    """
    Error correction scheme.
      'cascade'  — Cascade interactive protocol  (Phase 5)
      'ldpc'     — LDPC forward error correction (Phase 5, future)
    """

    ecc_passes:          int   = 4
    """Number of Cascade passes. Recommended: 4. Phase 5."""

    ecc_efficiency:      float = 1.16
    """
    Error correction efficiency f ≥ 1.
      f = 1.0  → Shannon-limit (ideal, impossible in practice)
      f = 1.16 → Realistic LDPC / Cascade
      f = 1.3  → Conservative
    Phase 5.
    """

    privacy_amp_enabled: bool  = False
    """True → apply privacy amplification after error correction. Phase 5."""

    security_parameter:  int   = 64
    """
    Security parameter s (bits). The probability of security failure
    is ≤ 2^(-s). Typical: 64 – 128 bits. Phase 5.
    """


# ══════════════════════════════════════════════════════════════════════
# QBER RESULT
# ══════════════════════════════════════════════════════════════════════

@dataclass
class QBERResult:
    """
    Output from estimate_qber().

    Phase 5 extension: add post_ecc_qber field after error correction.
    """
    qber:             float   # Quantum Bit Error Rate  (0.0 – 1.0)
    errors:           int     # disagreements found in sample
    sample_size:      int     # bits consumed for estimation
    security_status:  str     # 'SECURE ✓' | 'WARNING ⚠' | 'ABORT ✗'
    confidence_low:   float   # 95 % Wilson CI lower bound
    confidence_high:  float   # 95 % Wilson CI upper bound


# ══════════════════════════════════════════════════════════════════════
# SECURITY ANALYSIS RESULT  (Phase 5)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SecurityAnalysis:
    """
    Full information-theoretic security analysis of one run.
    Populated by bb84_analysis.py after Phase 5 post-processing.
    """
    binary_entropy_qber:      float = 0.0
    """H(QBER) — binary entropy at the observed error rate."""

    mutual_info_alice_bob:    float = 0.0
    """I(A;B) — mutual information between Alice and Bob (bits/symbol)."""

    holevo_bound_eve:         float = 0.0
    """χ(Eve) — Holevo bound on Eve's information per bit."""

    secret_key_rate_ideal:    float = 0.0
    """r = 1 − 2·H(e)  — Shor-Preskill ideal SKR (bits/qubit)."""

    secret_key_rate_realistic: float = 0.0
    """r = 1 − H(e) − f·H(e)  — realistic SKR with EC overhead."""

    secret_key_rate_gllp:     float = 0.0
    """GLLP / decoy-state SKR (populated if decoy_state_enabled)."""

    estimated_eve_info:       float = 0.0
    """Upper bound on Eve's mutual information with the raw key."""

    privacy_amp_compression:  float = 0.0
    """Fraction by which the key is compressed in privacy amplification."""

    final_secret_bits:        int   = 0
    """Bits of composably secure secret key after all post-processing."""


# ══════════════════════════════════════════════════════════════════════
# POST-PROCESSING RESULT  (Phase 5)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PostProcessingResult:
    """Records the output of Phase 5 post-processing."""
    ecc_applied:              bool       = False
    ecc_scheme:               str        = "none"
    ecc_passes_used:          int        = 0
    bits_after_ecc:           int        = 0
    post_ecc_errors:          int        = 0
    bits_leaked_ecc:          int        = 0
    privacy_amp_applied:      bool       = False
    bits_after_privacy_amp:   int        = 0
    alice_secret_key:         List[int]  = None
    bob_secret_key:           List[int]  = None
    secret_keys_match:        bool       = False

    def __post_init__(self):
        if self.alice_secret_key is None:
            self.alice_secret_key = []
        if self.bob_secret_key is None:
            self.bob_secret_key = []


# ══════════════════════════════════════════════════════════════════════
# SIMULATION RESULT  (master output)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """
    Complete output of one run_simulation() call.
    All phases write into this single object.
    """
    # ── Core metrics ─────────────────────────────────────────────────
    config:                 SimulationConfig

    n_transmitted:          int
    """Qubits Alice sent."""

    n_sifted:               int
    """Bits surviving basis sifting (~50 % of n_transmitted)."""

    sifted_key_rate:        float
    """n_sifted / n_transmitted."""

    qber_result:            QBERResult
    """QBER estimation result."""

    alice_final_key:        List[int]
    """Alice's final key (before Phase 5 if not enabled)."""

    bob_final_key:          List[int]
    """Bob's final key (before Phase 5 if not enabled)."""

    key_agreement_rate:     float
    """Fraction of alice_final_key bits matching bob_final_key."""

    eve_interception_rate:  float
    """Fraction of qubits Eve intercepted (0.0 if Eve absent)."""

    runtime_seconds:        float

    # ── Phase 4 ───────────────────────────────────────────────────────
    pns_multi_photon_rate:  float = 0.0
    """Fraction of pulses Eve could read silently via PNS."""

    pns_detection_rate:     float = 0.0
    """Bob's detection rate after PNS (reduced vs ideal)."""

    decoy_q1_estimate:      float = 0.0
    """Estimated single-photon gain from decoy analysis. Phase 4."""

    decoy_e1_estimate:      float = 0.0
    """Estimated single-photon phase error from decoy analysis. Phase 4."""

    # ── Phase 5 ───────────────────────────────────────────────────────
    post_processing:        Optional[PostProcessingResult] = None
    """Phase 5 post-processing output. None if ecc_enabled=False."""

    security_analysis:      Optional[SecurityAnalysis] = None
    """Full security analysis. None if not computed."""

    # ── Convenience properties ────────────────────────────────────────

    @property
    def key_length(self) -> int:
        return len(self.alice_final_key)

    @property
    def keys_match(self) -> bool:
        return self.alice_final_key == self.bob_final_key

    @property
    def key_generation_rate(self) -> float:
        """Bits of sifted key per transmitted qubit (before Phase 5)."""
        return self.key_length / self.n_transmitted if self.n_transmitted else 0.0

    @property
    def secret_key_rate(self) -> float:
        """
        Bits of secret key per transmitted qubit (after Phase 5).
        Returns key_generation_rate if Phase 5 not applied.
        """
        if self.post_processing and self.post_processing.privacy_amp_applied:
            return self.post_processing.bits_after_privacy_amp / self.n_transmitted
        if self.security_analysis:
            return self.security_analysis.secret_key_rate_realistic
        return self.key_generation_rate
