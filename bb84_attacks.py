"""
bb84_attacks.py
===============
Attack model definitions for BB84 simulations.

The default and currently implemented attack is intercept-resend.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np
from qiskit import QuantumCircuit


class NoAttack:
    """Pass-through attack model used when Eve is disabled."""

    def intercept(self, qc: QuantumCircuit, index: int, channel: Any) -> QuantumCircuit:
        return qc

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "intercepted": 0,
            "bases": {},
            "bits": {},
        }


class InterceptResendAttack:
    """
    Standard BB84 intercept-resend attack.

    Eve intercepts with probability intercept_prob, measures in random basis,
    and forwards a re-prepared qubit in her measured state.
    """

    def __init__(self, intercept_prob: float = 1.0, seed: Optional[int] = None):
        self.intercept_prob = intercept_prob
        self._rng = np.random.default_rng(seed)
        self._intercepted_count = 0
        self._eve_bases: Dict[int, int] = {}
        self._eve_bits: Dict[int, int] = {}

    def intercept(self, qc: QuantumCircuit, index: int, channel: Any) -> QuantumCircuit:
        if random.random() > self.intercept_prob:
            return qc

        self._intercepted_count += 1

        eve_base = int(self._rng.integers(0, 2))
        self._eve_bases[index] = eve_base

        meas_qc = qc.copy()
        if eve_base == 1:
            meas_qc.h(0)
        meas_qc.measure(0, 0)
        measured_bit = channel.run_circuit(meas_qc)
        self._eve_bits[index] = measured_bit

        new_qc = QuantumCircuit(1, 1, name=f"E{index}")
        if measured_bit == 1:
            new_qc.x(0)
        if eve_base == 1:
            new_qc.h(0)
        return new_qc

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "intercepted": self._intercepted_count,
            "bases": self._eve_bases,
            "bits": self._eve_bits,
        }


def build_attack_model(
    attack_model: str = "intercept_resend",
    intercept_prob: float = 1.0,
    seed: Optional[int] = None,
):
    """Create an attack model instance by name."""
    name = attack_model.strip().lower()
    if name in {"none", "off"}:
        return NoAttack()
    if name in {"intercept_resend", "intercept-resend", "ir"}:
        return InterceptResendAttack(intercept_prob=intercept_prob, seed=seed)
    raise ValueError(
        f"Unsupported attack model '{attack_model}'. "
        "Supported: none, intercept_resend"
    )
