"""Unit tests for Alice's qubit encoding (bb84_core.Alice).

Checks the encoding table documented in bb84_core.py:
    bit=0, basis=0 -> |0>   (no gates)
    bit=1, basis=0 -> |1>   (X gate)
    bit=0, basis=1 -> |+>   (H gate)
    bit=1, basis=1 -> |->   (X, then H)
"""

from bb84_core import Alice


def _gate_names(qc):
    return [instr.operation.name for instr in qc.data]


def test_bit0_basis0_encodes_as_identity():
    alice = Alice(n_qubits=1)
    alice.bits[0], alice.bases[0] = 0, 0
    assert _gate_names(alice.prepare_qubit(0)) == []


def test_bit1_basis0_encodes_as_x():
    alice = Alice(n_qubits=1)
    alice.bits[0], alice.bases[0] = 1, 0
    assert _gate_names(alice.prepare_qubit(0)) == ["x"]


def test_bit0_basis1_encodes_as_h():
    alice = Alice(n_qubits=1)
    alice.bits[0], alice.bases[0] = 0, 1
    assert _gate_names(alice.prepare_qubit(0)) == ["h"]


def test_bit1_basis1_encodes_as_x_then_h():
    alice = Alice(n_qubits=1)
    alice.bits[0], alice.bases[0] = 1, 1
    assert _gate_names(alice.prepare_qubit(0)) == ["x", "h"]


def test_sift_key_returns_bits_at_matching_indices():
    alice = Alice(n_qubits=5)
    alice.bits = [0, 1, 1, 0, 1]
    assert alice.sift_key([0, 2, 4]) == [0, 1, 1]


def test_seeded_construction_is_reproducible():
    a1 = Alice(n_qubits=50, seed=7)
    a2 = Alice(n_qubits=50, seed=7)
    assert a1.bits == a2.bits
    assert a1.bases == a2.bases
