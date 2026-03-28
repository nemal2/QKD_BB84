"""
bb84config.py
=============
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


@dataclass
class QBERResult:
    """Quantum Bit Error Rate (QBER) summary with confidence interval."""

    qber: float
    n_errors: int
    n_sampled: int
    ci_low: float
    ci_high: float
    qber_threshold: float = 0.11
    security_status: str = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.qber <= 1.0:
            raise ValueError("qber must be between 0 and 1")

        if self.n_errors < 0 or self.n_sampled < 0:
            raise ValueError("n_errors and n_sampled must be >= 0")

        if self.n_errors > self.n_sampled:
            raise ValueError("n_errors cannot exceed n_sampled")

        if self.n_sampled == 0:
            self.security_status = "INSUFFICIENT SAMPLE ⚠"
            return

        if self.qber <= self.qber_threshold * 0.5:
            self.security_status = "SECURE ✓"
        elif self.qber <= self.qber_threshold:
            self.security_status = "WARNING ⚠"
        else:
            self.security_status = "ABORT ✗"


@dataclass
class SimulationResult:
    """Complete output summary for one BB84 simulation run."""

    label: str
    n_transmitted: int
    n_sifted: int
    key_length: int
    sifted_key_rate: float
    key_generation_rate: float
    qber_result: QBERResult
    keys_match: bool
    runtime_seconds: float
    alice_key: List[int] = field(default_factory=list)
    bob_key: List[int] = field(default_factory=list)
    eve_intercept_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable dict representation."""
        return asdict(self)


__all__ = [
    "SimulationConfig",
    "QBERResult",
    "SimulationResult",
]
