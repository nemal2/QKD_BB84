"""End-to-end integration tests exercising the full BB84 pipeline
(Alice -> channel -> Bob -> sifting -> QBER estimation) together, as
described in the Main Objective (Sec. 1.3.1 of the report).
"""

from bb84_core import Alice, Bob, sift_keys, estimate_qber


def test_ideal_channel_with_no_eavesdropper_yields_zero_qber(ideal_channel):
    n = 300
    alice = Alice(n_qubits=n, seed=11)
    bob = Bob(n_qubits=n, seed=11)

    for i in range(n):
        bob.measure(alice.prepare_qubit(i), i, ideal_channel)

    matching = sift_keys(alice.bases, bob.bases)
    alice_key = alice.sift_key(matching)
    bob_key = bob.sift_key(matching)

    result = estimate_qber(alice_key, bob_key, sample_fraction=1.0, seed=11)

    assert result.qber == 0.0
    assert result.security_status == "SECURE ok"


def test_full_pipeline_runs_end_to_end_with_expected_key_lengths(ideal_channel):
    n = 600  # the report's own recommended default (Sec. 4.1)
    alice = Alice(n_qubits=n, seed=12)
    bob = Bob(n_qubits=n, seed=12)

    for i in range(n):
        bob.measure(alice.prepare_qubit(i), i, ideal_channel)

    matching = sift_keys(alice.bases, bob.bases)
    assert 0.40 * n <= len(matching) <= 0.60 * n

    alice_key = alice.sift_key(matching)
    bob_key = bob.sift_key(matching)
    assert alice_key == bob_key  # ideal channel: sifted keys must agree exactly

    result = estimate_qber(alice_key, bob_key, seed=12)
    assert result.sample_size > 0
    assert result.security_status == "SECURE ok"
