"""Tests for LDPC syndrome reconciliation integrated into the BB84
pipeline (reconciliation.reconcile_full_key, bb84_runner's Step 5).

Covers the chunking strategy (exact multiples, remainders, short keys),
correctness on a noiseless channel, the documented statistical rate of
undetected decode failures at moderate QBER, and the List[int] <->
np.uint8 conversion boundary.
"""

import numpy as np
import pytest

from bb84_config import SimulationConfig
from bb84_noise import NoiseModelType
from bb84_runner import run_simulation
from reconciliation import reconcile_full_key, LDPCReconciler, _MIN_LDPC_BLOCK


# ──────────────────────────────────────────────────────────────────────
# Chunking edge cases
# ──────────────────────────────────────────────────────────────────────

def test_exact_multiple_of_block_len_has_no_remainder():
    n = 4 * 160
    alice = [0] * n
    bob = [0] * n
    result = reconcile_full_key(alice, bob, p_est=0.01, block_len=160, seed=0)

    assert result.n_blocks == 4
    assert result.remainder_bits == 0
    assert result.total_input_bits == n


def test_partial_block_leaves_a_remainder():
    n = 3 * 160 + 47
    alice = [0] * n
    bob = [0] * n
    result = reconcile_full_key(alice, bob, p_est=0.01, block_len=160, seed=0)

    assert result.n_blocks == 3
    assert result.remainder_bits == 47
    assert result.total_input_bits == 3 * 160


def test_shorter_than_one_block_reconciles_nothing():
    alice = [0] * 50
    bob = [0] * 50
    result = reconcile_full_key(alice, bob, p_est=0.01, block_len=160, seed=0)

    assert result.n_blocks == 0
    assert result.remainder_bits == 50
    assert result.reconciled_alice_key == []
    assert result.reconciled_bob_key == []


def test_min_ldpc_block_below_threshold_is_skipped_by_the_runner():
    cfg = SimulationConfig(n_qubits=20, seed=3, ldpc_enabled=True)
    result = run_simulation(cfg, verbose=False)

    assert result.key_length < _MIN_LDPC_BLOCK
    assert result.ldpc_result is None


# ──────────────────────────────────────────────────────────────────────
# Correctness on an ideal (noiseless) channel
# ──────────────────────────────────────────────────────────────────────

def test_ideal_channel_reconciles_every_block_correctly():
    cfg = SimulationConfig(
        n_qubits=2000, seed=1, noise_model=NoiseModelType.IDEAL,
        ldpc_enabled=True, ldpc_block_len=160,
    )
    result = run_simulation(cfg, verbose=False)
    ldpc = result.ldpc_result

    assert ldpc is not None
    assert ldpc.n_blocks > 0
    assert ldpc.all_blocks_correct
    assert not ldpc.any_undetected_error
    assert ldpc.keys_match
    assert ldpc.reconciled_alice_key == ldpc.reconciled_bob_key


def test_default_config_is_unaffected_by_ldpc_fields():
    """Regression guard: ldpc_enabled=False (the default) must not
    change any existing SimulationResult field."""
    cfg = SimulationConfig(n_qubits=800, seed=42)
    result = run_simulation(cfg, verbose=False)

    assert result.ldpc_result is None
    assert result.key_length == len(result.alice_final_key)


# ──────────────────────────────────────────────────────────────────────
# Statistical behaviour at moderate QBER
# ──────────────────────────────────────────────────────────────────────

def test_undetected_error_rate_stays_low_at_moderate_qber():
    """BP syndrome decoding can converge to a syndrome-consistent but
    wrong error pattern near a code's threshold (a known, documented
    characteristic of this decoder - see reconciliation_report.md).
    This is a bounded-rate statistical check, not a hard zero."""
    n_undetected = 0
    n_blocks_total = 0
    for seed in range(20):
        cfg = SimulationConfig(
            n_qubits=3000, seed=seed, noise_model=NoiseModelType.DEPOLARIZING,
            depolar_prob=0.03, ldpc_enabled=True, ldpc_block_len=160,
        )
        result = run_simulation(cfg, verbose=False)
        ldpc = result.ldpc_result
        if ldpc is None:
            continue
        n_blocks_total += ldpc.n_blocks
        n_undetected += sum(b.claimed_success and not b.actually_correct for b in ldpc.blocks)

    assert n_blocks_total > 0
    assert n_undetected / n_blocks_total < 0.10


# ──────────────────────────────────────────────────────────────────────
# List[int] <-> np.uint8 boundary
# ──────────────────────────────────────────────────────────────────────

def test_reconcile_full_key_accepts_plain_int_lists():
    n = 160
    alice = [0, 1] * (n // 2)
    bob = list(alice)  # identical -> zero errors
    reconciler = LDPCReconciler(n=n, seed=0)

    result = reconcile_full_key(alice, bob, p_est=0.01, block_len=n, seed=0,
                                 reconciler=reconciler)

    assert result.all_blocks_correct
    assert all(isinstance(b, int) for b in result.reconciled_alice_key)
    assert all(isinstance(b, int) for b in result.reconciled_bob_key)


def test_mismatched_length_keys_raise_value_error():
    with pytest.raises(ValueError):
        reconcile_full_key([0, 1, 0], [0, 1], p_est=0.01, block_len=160, seed=0)
