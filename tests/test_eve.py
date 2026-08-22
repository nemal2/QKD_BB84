"""Unit tests for Eve's intercept-resend attack (bb84_core.Eve)."""

from bb84_core import Alice, Bob, Eve, sift_keys


def test_zero_probability_never_intercepts(ideal_channel):
    alice = Alice(n_qubits=50, seed=2)
    eve = Eve(intercept_prob=0.0, seed=2)

    for i in range(50):
        qc = alice.prepare_qubit(i)
        returned = eve.intercept(qc, i, ideal_channel)
        assert returned is qc  # passed straight through, untouched

    assert eve.intercepted_count == 0


def test_full_probability_always_intercepts(ideal_channel):
    n = 50
    alice = Alice(n_qubits=n, seed=2)
    eve = Eve(intercept_prob=1.0, seed=2)

    for i in range(n):
        eve.intercept(alice.prepare_qubit(i), i, ideal_channel)

    assert eve.intercepted_count == n
    assert set(eve.stats["bases"].keys()) == set(range(n))


def test_full_interception_produces_25_percent_qber(ideal_channel):
    """Reproduces the report's central validated law (bennett1984): a full
    intercept-resend attack disturbs a matched-basis qubit with probability
    1/2 (Eve picks the wrong basis) * 1/2 (wrong-basis measurement collapses
    randomly) = 1/4, i.e. a 25% QBER ceiling.
    """
    n = 1000
    alice = Alice(n_qubits=n, seed=3)
    bob = Bob(n_qubits=n, seed=3)
    eve = Eve(intercept_prob=1.0, seed=3)

    matching = sift_keys(alice.bases, bob.bases)
    errors = 0
    for i in matching:
        forwarded_qc = eve.intercept(alice.prepare_qubit(i), i, ideal_channel)
        result = bob.measure(forwarded_qc, i, ideal_channel)
        if result != alice.bits[i]:
            errors += 1

    qber = errors / len(matching)
    assert 0.20 <= qber <= 0.30, (
        f"expected QBER near the 25% theoretical ceiling, got {qber:.3f}"
    )
