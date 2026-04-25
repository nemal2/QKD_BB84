"""
bb84_config.py
==============
Configuration and result data models for the BB84 Phase 1 simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SimulationConfig:
    """Runtime configuration for one BB84 simulation run."""

    n_qubits: int = 500
    eve_present: bool = False
    eve_intercept_prob: float = 0.0
    noise_enabled: bool = False
    depolar_prob: float = 0.0
    sample_fraction: float = 0.10
    seed: Optional[int] = None
    label: str = "Custom"

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be > 0")
        if not 0.0 <= self.eve_intercept_prob <= 1.0:
            raise ValueError("eve_intercept_prob must be between 0 and 1")
        if not 0.0 <= self.depolar_prob <= 1.0:
            raise ValueError("depolar_prob must be between 0 and 1")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if not self.eve_present and self.eve_intercept_prob != 0.0:
            self.eve_intercept_prob = 0.0
        if not self.noise_enabled and self.depolar_prob != 0.0:
            self.depolar_prob = 0.0


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QBERResult:
    """
    Quantum Bit Error Rate summary with Wilson confidence interval.

    Field names match what bb84_core.estimate_qber() produces and
    what bb84_runner / bb84_plots consume.
    """

    qber: float
    errors: int           # was n_errors  — renamed to match core + runner
    sample_size: int      # was n_sampled — renamed to match core + runner
    confidence_low: float # was ci_low    — renamed to match core + runner
    confidence_high: float# was ci_high   — renamed to match core + runner
    security_status: str  # set by bb84_core.estimate_qber(), plain field
    qber_threshold: float = 0.11

    def __post_init__(self) -> None:
        if not 0.0 <= self.qber <= 1.0:
            raise ValueError("qber must be between 0 and 1")
        if self.errors < 0 or self.sample_size < 0:
            raise ValueError("errors and sample_size must be >= 0")
        if self.errors > self.sample_size:
            raise ValueError("errors cannot exceed sample_size")


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Complete output summary for one BB84 simulation run.

    Field names match what bb84_runner.run_simulation() produces and
    what bb84_plots consumes.
    """

    label: str                          # config.label
    n_transmitted: int                  # qubits sent by Alice
    n_sifted: int                       # bits after basis sifting
    key_length: int                     # bits after QBER sample removed
    sifted_key_rate: float              # n_sifted / n_transmitted
    key_generation_rate: float          # key_length / n_transmitted
    qber_result: QBERResult
    key_agreement_rate: float           # fraction of final key bits that match
    keys_match: bool                    # True when agreement == 1.0
    runtime_seconds: float
    alice_key: List[int] = field(default_factory=list)
    bob_key: List[int]   = field(default_factory=list)
    eve_intercept_rate: Optional[float] = None   # None when Eve absent
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a fully serialisable dict representation."""
        return asdict(self)


__all__ = [
    "SimulationConfig",
    "QBERResult",
    "SimulationResult",
]