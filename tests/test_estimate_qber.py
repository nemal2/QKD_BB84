"""Unit tests for QBER estimation and security classification
(bb84_core.estimate_qber).

Security thresholds under test (documented in bb84_core.py):
    QBER <  5%          -> SECURE ok
    5% <= QBER < 11%     -> WARNING
    QBER >= 11%          -> ABORT
"""

import pytest

from bb84_core import estimate_qber


def _make_keys(n: int, error_rate: float):
    """A deterministic key pair with exactly `error_rate` fraction of
    mismatches, so the exact QBER is known in advance."""
    n_errors = round(n * error_rate)
    alice_key = [0] * n
    bob_key = [0] * n
    for i in range(n_errors):
        bob_key[i] = 1
    return alice_key, bob_key


def test_identical_keys_give_zero_qber_and_secure_status():
    alice_key = [0, 1, 0, 1, 1, 0, 1, 0] * 50
    bob_key = list(alice_key)
    result = estimate_qber(alice_key, bob_key, sample_fraction=1.0, seed=0)

    assert result.qber == 0.0
    assert result.errors == 0
    assert result.security_status == "SECURE ok"


def test_low_error_rate_is_classified_secure():
    alice_key, bob_key = _make_keys(1000, 0.03)
    result = estimate_qber(alice_key, bob_key, sample_fraction=1.0, seed=0)

    assert result.qber == pytest.approx(0.03)
    assert result.security_status == "SECURE ok"


def test_mid_error_rate_is_classified_warning():
    alice_key, bob_key = _make_keys(1000, 0.07)
    result = estimate_qber(alice_key, bob_key, sample_fraction=1.0, seed=0)

    assert result.qber == pytest.approx(0.07)
    assert result.security_status == "WARNING "


def test_high_error_rate_is_classified_abort():
    alice_key, bob_key = _make_keys(1000, 0.15)
    result = estimate_qber(alice_key, bob_key, sample_fraction=1.0, seed=0)

    assert result.qber == pytest.approx(0.15)
    assert result.security_status == "ABORT x"


def test_empty_key_returns_documented_edge_case_result():
    result = estimate_qber([], [], seed=0)

    assert result.qber == 0.0
    assert result.sample_size == 0
    assert result.security_status == "SECURE ok"
    assert result.confidence_low == 0.0
    assert result.confidence_high == 1.0


def test_mismatched_key_lengths_raise_assertion_error():
    with pytest.raises(AssertionError):
        estimate_qber([0, 1, 0], [0, 1], seed=0)


def test_confidence_interval_contains_the_point_estimate():
    alice_key, bob_key = _make_keys(1000, 0.10)
    result = estimate_qber(alice_key, bob_key, sample_fraction=1.0, seed=0)

    assert result.confidence_low <= result.qber <= result.confidence_high


def test_larger_sample_gives_a_narrower_confidence_interval():
    alice_key, bob_key = _make_keys(2000, 0.10)

    small_sample = estimate_qber(alice_key, bob_key, sample_fraction=0.05, seed=0)
    large_sample = estimate_qber(alice_key, bob_key, sample_fraction=1.0, seed=0)

    small_width = small_sample.confidence_high - small_sample.confidence_low
    large_width = large_sample.confidence_high - large_sample.confidence_low
    assert large_width < small_width
