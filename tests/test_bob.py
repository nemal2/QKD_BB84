"""Unit tests for Bob's qubit measurement (bb84_core.Bob)."""

from bb84_core import Alice, Bob


def test_measurement_is_deterministic_when_bases_match(ideal_channel):
    for bit in (0, 1):
        for basis in (0, 1):
            alice = Alice(n_qubits=1)
            alice.bits[0], alice.bases[0] = bit, basis
            bob = Bob(n_qubits=1)
            bob.bases[0] = basis  # force matching basis

            qc = alice.prepare_qubit(0)
            result = bob.measure(qc, 0, ideal_channel)

            assert result == bit
            assert bob.measured_bits[0] == bit


def test_measurement_is_uniformly_random_when_bases_mismatch(ideal_channel):
    # Basis complementarity (Sec. 2.1 of the report): measuring |0> in the
    # diagonal basis must collapse to 0/1 with equal probability.
    n = 400
    alice = Alice(n_qubits=n, seed=1)
    for i in range(n):
        alice.bits[i], alice.bases[i] = 0, 0  # always |0>, rectilinear basis

    bob = Bob(n_qubits=n, seed=1)
    for i in range(n):
        bob.bases[i] = 1  # always measure in the mismatched (diagonal) basis

    ones = sum(bob.measure(alice.prepare_qubit(i), i, ideal_channel) for i in range(n))

    fraction_ones = ones / n
    assert 0.40 <= fraction_ones <= 0.60, (
        f"expected ~50% random outcomes on a basis mismatch, got {fraction_ones:.3f}"
    )


def test_sift_key_returns_measured_bits_at_matching_indices():
    bob = Bob(n_qubits=5)
    bob.measured_bits = [0, 1, 1, 0, 1]
    assert bob.sift_key([0, 2, 4]) == [0, 1, 1]
