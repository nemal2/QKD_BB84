"""
bb84config.py
==============
Central configuration and result dataclasses for the BB84 QKD simulator.
All phases share these definitions — extend here as new phases land.

Modification log
----------------
Phase 1  (baseline)  : SimulationConfig, QBERResult, SimulationResult
Phase 3              : add configurable channel noise models
Phase 4              : add configurable attack model selection
Phase 5  (planned)   : add ecc_scheme: str field to SimulationConfig
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """
    Central knob-panel for one BB84 run.

    Phase 1 fields (active now)
    ─────────────────────────────
    n_qubits            : total qubits Alice transmits
    eve_present         : True → Eve performs intercept-resend
    eve_intercept_prob  : fraction of qubits Eve intercepts (0.0 – 1.0)
    noise_enabled       : True → add channel noise
    noise_model         : "depolarizing" | "phase_damping" | "amplitude_damping"
    depolar_prob        : depolarizing error probability per gate
    phase_damp_prob     : phase damping probability per gate
    amplitude_damp_prob : amplitude damping probability per gate
    sample_fraction     : fraction of sifted key used for QBER check
    seed                : None = random run, int = reproducible
    label               : human-readable run name for plots/reports

    attack_model        : "intercept_resend" (current) or future attack variants

    Phase 5 hooks
    ─────────────
    # ecc_scheme  : str = "none"   # "cascade" | "ldpc"
    # privacy_amp : bool = False
    """
    n_qubits:            int   = 1000
    eve_present:         bool  = False
    eve_intercept_prob:  float = 1.0
    noise_enabled:       bool  = False
    noise_model:         str   = "depolarizing"
    depolar_prob:        float = 0.01
    phase_damp_prob:     float = 0.00
    amplitude_damp_prob: float = 0.00
    attack_model:        str   = "intercept_resend"
    sample_fraction:     float = 0.10
    seed:      Optional[int]   = 42
    label:               str   = "Simulation"


# ──────────────────────────────────────────────────────────────────────────────
# QBER RESULT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QBERResult:
    """
    Output from estimate_qber().
    
    Phase 5 will extend this with post-error-correction residual QBER.
    """
    qber:             float   # Quantum Bit Error Rate  (0.0 – 1.0)
    errors:           int     # number of disagreements found in sample
    sample_size:      int     # bits consumed for QBER estimation
    security_status:  str     # "SECURE ✓" | "WARNING ⚠" | "ABORT ✗"
    confidence_low:   float   # 95% Wilson CI lower bound
    confidence_high:  float   # 95% Wilson CI upper bound


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION RESULT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Full output of one run_simulation() call.

    Phase 5 will add:
      corrected_key_length : int
      privacy_amplified_key: List[int]
      secret_key_rate      : float   (bits / qubit, after privacy amp)
    """
    config:                 SimulationConfig
    n_transmitted:          int
    n_sifted:               int
    sifted_key_rate:        float
    qber_result:            QBERResult
    alice_final_key:        List[int]
    bob_final_key:          List[int]
    key_agreement_rate:     float     # fraction of final key bits that match
    eve_interception_rate:  float     # 0.0 if Eve absent
    runtime_seconds:        float

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def key_length(self) -> int:
        return len(self.alice_final_key)

    @property
    def keys_match(self) -> bool:
        return self.alice_final_key == self.bob_final_key

    @property
    def key_generation_rate(self) -> float:
        """bits of final key produced per transmitted qubit."""
        if self.n_transmitted == 0:
            return 0.0
        return self.key_length / self.n_transmitted