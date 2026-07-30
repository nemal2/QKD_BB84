"""Unit tests for basis reconciliation (bb84_core.sift_keys)."""

import numpy as np

from bb84_core import sift_keys


def test_returns_indices_where_bases_agree():
    alice_bases = [0, 1, 0, 1, 1]
    bob_bases = [0, 0, 0, 1, 0]
    assert sift_keys(alice_bases, bob_bases) == [0, 2, 3]


def test_no_matches_returns_empty_list():
    assert sift_keys([0, 0, 0], [1, 1, 1]) == []


def test_all_matches_returns_every_index():
    bases = [0, 1, 1, 0]
    assert sift_keys(bases, bases) == [0, 1, 2, 3]


def test_retention_rate_is_approximately_one_half():
    # Independent random bases agree with probability 1/2 (Sec. 2.2 of the
    # report); with 20,000 trials the sample fraction should sit very close
    # to that value.
    rng = np.random.default_rng(42)
    n = 20_000
    alice_bases = rng.integers(0, 2, n).tolist()
    bob_bases = rng.integers(0, 2, n).tolist()

    retained_fraction = len(sift_keys(alice_bases, bob_bases)) / n
    assert 0.49 <= retained_fraction <= 0.51
