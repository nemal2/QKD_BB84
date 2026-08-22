import pytest

from bb84_noise import QuantumChannel, NoiseModelType


@pytest.fixture
def ideal_channel() -> QuantumChannel:
    """A noiseless, lossless quantum channel for deterministic testing."""
    return QuantumChannel(noise_model=NoiseModelType.IDEAL)
