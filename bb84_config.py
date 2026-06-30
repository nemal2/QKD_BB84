"""
bb84_config.py
==============
Configuration dataclasses for the BB84 QKD simulator.

SimulationConfig  - all tunable parameters in one place
QBERResult        - QBER estimation output with Wilson CI
SimulationResult  - full output of one simulation run

University of Ruhuna - Dept. of Computer Engineering
MIT Licence - see LICENSE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────
# SIMULATION CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """
    One complete BB84 simulation configuration.

    All parameters have safe defaults so the simplest usage is just::

        cfg = SimulationConfig()          # 1000 qubits, ideal channel
        cfg = SimulationConfig(n_qubits=500, eve_present=True)
    """

    # ── Core ──────────────────────────────────────────────────────────
    n_qubits: int = 1000
    """Total qubits Alice transmits."""

    seed: Optional[int] = 42
    """RNG seed.  None → new random run each time; int → reproducible."""

    label: str = "Simulation"
    """Human-readable name used in plots and console output."""

    sample_fraction: float = 0.15
    """Fraction of sifted key consumed for QBER estimation (0-1)."""

    # ── Eve (intercept-resend attack) ─────────────────────────────────
    eve_present: bool = False
    """True → Eve performs an intercept-resend attack."""

    eve_intercept_prob: float = 1.0
    """Fraction of qubits Eve intercepts (0.0-1.0)."""

    # ── Channel noise (Phase 1, legacy) ───────────────────────────────
    noise_enabled: bool = False
    """True → apply depolarising channel noise. Ignored if noise_model is set."""

    depolar_prob: float = 0.01
    """Depolarising error probability per gate (DEPOLARIZING model)."""

    # ── Channel noise (Phase 3) ────────────────────────────────────────
    noise_model: Optional[str] = None
    """
    Phase 3 channel model selector. One of:
    'ideal' | 'depolarizing' | 'amplitude_damping' | 'phase_damping' |
    'combined' | 'fibre_loss'  (see bb84_noise.NoiseModelType).

    Default None → fall back to the legacy ``noise_enabled`` /
    ``depolar_prob`` fields, reproducing Phase 1 behaviour exactly.
    Setting this field explicitly always takes precedence.
    """

    t1_ns: float = 10_000.0
    """T1 energy-relaxation time in ns (amplitude_damping / combined).
    Default 10 us, representative of current transmon hardware."""

    t2_ns: float = 8_000.0
    """T2 dephasing time in ns (phase_damping / combined).
    Must satisfy T2 <= 2*T1; enforced automatically in bb84_noise."""

    gate_time_ns: float = 50.0
    """Single-qubit gate duration in ns."""

    channel_length_km: float = 0.0
    """Fibre-optic channel length in km (fibre_loss model only)."""


# ──────────────────────────────────────────────────────────────────────
# QBER RESULT
# ──────────────────────────────────────────────────────────────────────

@dataclass
class QBERResult:
    """Output of ``estimate_qber()``."""

    qber: float
    """Quantum Bit Error Rate (0.0-1.0)."""

    errors: int
    """Bit disagreements found in the sample."""

    sample_size: int
    """Number of bits consumed for estimation."""

    security_status: str
    """``'SECURE ok'`` | ``'WARNING '`` | ``'ABORT x'``"""

    confidence_low: float
    """Lower bound of the 95 % Wilson confidence interval."""

    confidence_high: float
    """Upper bound of the 95 % Wilson confidence interval."""


# ──────────────────────────────────────────────────────────────────────
# SIMULATION RESULT
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Complete output of one ``run_simulation()`` call."""

    config: SimulationConfig
    n_transmitted: int
    n_sifted: int
    sifted_key_rate: float
    qber_result: QBERResult
    alice_final_key: List[int]
    bob_final_key: List[int]
    key_agreement_rate: float
    eve_interception_rate: float
    runtime_seconds: float
    n_lost: int = 0
    """Qubits never detected by Bob (FIBRE_LOSS model). 0 for all other channels."""

    @property
    def photon_survival_rate(self) -> float:
        """Empirical fraction of transmitted qubits that reached Bob's detector."""
        return 1.0 - (self.n_lost / self.n_transmitted if self.n_transmitted else 0.0)

    @property
    def key_length(self) -> int:
        """Number of bits in the final (post-QBER-sample) key."""
        return len(self.alice_final_key)

    @property
    def keys_match(self) -> bool:
        """True when Alice's and Bob's final keys are identical."""
        return self.alice_final_key == self.bob_final_key

    @property
    def key_generation_rate(self) -> float:
        """Final key bits produced per transmitted qubit."""
        return self.key_length / self.n_transmitted if self.n_transmitted else 0.0