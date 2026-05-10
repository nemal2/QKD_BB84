"""
bb84_noise.py
=============
Noise model factory for BB84 quantum channel simulation.

This module centralizes channel noise definitions so the core protocol
logic can stay focused on BB84 flow.
"""

from __future__ import annotations

from typing import Optional

from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)


_SUPPORTED_MODELS = {
    "depolarizing",
    "phase_damping",
    "amplitude_damping",
}


def build_noise_model(
    noise_enabled: bool,
    noise_model: str = "depolarizing",
    depolar_prob: float = 0.0,
    phase_damp_prob: float = 0.0,
    amplitude_damp_prob: float = 0.0,
) -> Optional[NoiseModel]:
    """
    Build and return a Qiskit NoiseModel for the selected channel model.

    Returns None when noise is disabled or configured probability is zero.
    """
    if not noise_enabled:
        return None

    model_name = noise_model.strip().lower()
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported noise model '{noise_model}'. "
            f"Supported: {sorted(_SUPPORTED_MODELS)}"
        )

    noise = NoiseModel()

    if model_name == "depolarizing":
        if depolar_prob <= 0.0:
            return None
        error = depolarizing_error(depolar_prob, 1)
    elif model_name == "phase_damping":
        if phase_damp_prob <= 0.0:
            return None
        error = phase_damping_error(phase_damp_prob)
    else:
        if amplitude_damp_prob <= 0.0:
            return None
        error = amplitude_damping_error(amplitude_damp_prob)

    noise.add_all_qubit_quantum_error(error, ["x", "h", "id"])
    return noise


def noise_summary(
    noise_enabled: bool,
    noise_model: str,
    depolar_prob: float,
    phase_damp_prob: float,
    amplitude_damp_prob: float,
) -> str:
    """Return a short human-readable description of the chosen noise setup."""
    if not noise_enabled:
        return "disabled"

    model_name = noise_model.strip().lower()
    if model_name == "depolarizing":
        return f"depolarizing p={depolar_prob}"
    if model_name == "phase_damping":
        return f"phase_damping p={phase_damp_prob}"
    if model_name == "amplitude_damping":
        return f"amplitude_damping p={amplitude_damp_prob}"
    return f"unknown model={noise_model}"
