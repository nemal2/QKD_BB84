"""
bb84_eve_adaptive.py
=====================
Adaptive-eavesdropper threat-model extensions for stress-testing
ZNE-QBER.

`Eve` in bb84_core.py is a STATIC intercept-resend attacker:
`intercept_prob` is one fixed float, chosen once, independent of any
runtime signal. Every ZNE-QBER result in the paper validates against
only this static adversary.

ZNE-QBER's entire argument rests on one assumption: "Eve's
contribution to QBER does not scale with the noise-scaling factor f."
This module adds adversaries that try to make that assumption false,
using only information a real Eve could plausibly act on (her own
choices), never information that would break BB84's basic security
assumptions (e.g. neither class below ever learns Alice's basis before
Bob's).

Classes
-------
LinearScalingEve
    Deliberately scales her OWN interception probability linearly with
    f: p_eve(f) = clip(p_eve_base * f, 0, 1). At f=1 this is IDENTICAL
    to a standard static Eve with p=p_eve_base -- it is a strict
    generalisation, not a different primitive. This is the sharpest
    available test of ZNE-QBER's core assumption: if Eve's own
    contribution is now linear in f, just like hardware noise, a
    linear-in-f extrapolation to f=0 cannot distinguish her
    contribution from the hardware's and may extrapolate part of it
    away. Whether this actually breaks ZNE-QBER in practice (as
    opposed to merely being a symmetry a determined adversary could
    exploit if she can infer f) is an empirical question this module
    lets you answer directly.

BasisBiasedEve
    Standard intercept-resend mechanics, but Eve's own measurement
    basis is drawn Bernoulli(basis_bias) instead of uniformly (0.5).
    basis_bias > 0.5 biases toward rectilinear (basis=0). This never
    requires knowledge of Alice's basis -- only Eve's own choice, which
    is hers to set -- so it does not violate BB84's security
    assumptions. Tests whether basis-asymmetric noise interacting with
    a basis-biased attacker weakens the basis-resolved discriminator
    or ZNE-QBER differently than a uniform attacker does.

Both classes plug into a standalone one-shot runner (`run_bb84_point`)
below, mirroring bb84_runner.run_simulation's pipeline exactly but
accepting ANY object with an `.intercept(qc, index, channel)` method as
Eve. This is additive: bb84_core.py, bb84_runner.py and every existing
experiment are untouched.

University of Ruhuna - Dept. of Computer Engineering
MIT Licence - see LICENSE
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit

from bb84_core import Alice, Bob, sift_keys, estimate_qber
from bb84_config import QBERResult
from bb84_noise import QuantumChannel, NoiseModelType
from bb84_zne import scale_depolar, scale_amplitude_damping, scale_phase_damping, linear_extrapolate


# ──────────────────────────────────────────────────────────────────────
# 1. ADAPTIVE / ADVERSARIAL EVE VARIANTS
# ──────────────────────────────────────────────────────────────────────

class LinearScalingEve:
    """Intercept probability scales linearly with the noise-scaling
    factor f: p_eve(f) = clip(p_eve_base * f, 0, 1)."""

    def __init__(self, p_eve_base: float, f_scale: float, seed: Optional[int] = None):
        self.p_eve_base = p_eve_base
        self.f_scale = f_scale
        self.intercept_prob = min(max(p_eve_base * f_scale, 0.0), 1.0)
        self._rng = np.random.default_rng(None if seed is None else seed + 200)
        self.intercepted_count = 0
        self._shot_seed_base = (seed * 6271) if seed is not None else None

    def intercept(self, qc: QuantumCircuit, index: int, channel: QuantumChannel) -> QuantumCircuit:
        if random.random() > self.intercept_prob:
            return qc
        self.intercepted_count += 1
        eve_base = int(self._rng.integers(0, 2))
        meas_qc = qc.copy()
        if eve_base == 1:
            meas_qc.h(0)
        meas_qc.measure(0, 0)
        shot_seed = (self._shot_seed_base + index) if self._shot_seed_base is not None else None
        measured_bit = channel.run_circuit(meas_qc, shot_seed=shot_seed, apply_loss=False)
        new_qc = QuantumCircuit(1, 1, name=f"E{index}")
        if measured_bit == 1:
            new_qc.x(0)
        if eve_base == 1:
            new_qc.h(0)
        return new_qc


class BasisBiasedEve:
    """Standard intercept-resend mechanics; Eve's own measurement basis
    is Bernoulli(basis_bias) instead of uniform. Never inspects Alice's
    basis -- only her own choice is biased."""

    def __init__(self, intercept_prob: float, basis_bias: float = 0.5, seed: Optional[int] = None):
        self.intercept_prob = intercept_prob
        self.basis_bias = basis_bias
        self._rng = np.random.default_rng(None if seed is None else seed + 200)
        self.intercepted_count = 0
        self._shot_seed_base = (seed * 6271) if seed is not None else None

    def intercept(self, qc: QuantumCircuit, index: int, channel: QuantumChannel) -> QuantumCircuit:
        if random.random() > self.intercept_prob:
            return qc
        self.intercepted_count += 1
        eve_base = 0 if self._rng.random() < self.basis_bias else 1
        meas_qc = qc.copy()
        if eve_base == 1:
            meas_qc.h(0)
        meas_qc.measure(0, 0)
        shot_seed = (self._shot_seed_base + index) if self._shot_seed_base is not None else None
        measured_bit = channel.run_circuit(meas_qc, shot_seed=shot_seed, apply_loss=False)
        new_qc = QuantumCircuit(1, 1, name=f"E{index}")
        if measured_bit == 1:
            new_qc.x(0)
        if eve_base == 1:
            new_qc.h(0)
        return new_qc


# ──────────────────────────────────────────────────────────────────────
# 2. STANDALONE ONE-SHOT RUNNER  (accepts any Eve-like object)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveRunResult:
    qber: float          # percent
    ci_low: float        # percent
    ci_high: float       # percent
    n_sifted: int
    eve_intercepted: int


def run_bb84_point(
    n_qubits: int,
    seed: int,
    channel: QuantumChannel,
    eve=None,
    sample_fraction: float = 0.15,
) -> AdaptiveRunResult:
    """One BB84 run (Alice -> [eve] -> Bob -> sift -> QBER), identical
    physics to bb84_runner.run_simulation, but `eve` may be ANY object
    exposing `.intercept(qc, index, channel)` -- the stock bb84_core.Eve,
    or LinearScalingEve / BasisBiasedEve above."""
    alice = Alice(n_qubits, seed=seed)
    bob = Bob(n_qubits, seed=seed)
    for i in range(n_qubits):
        qc = alice.prepare_qubit(i)
        if eve is not None:
            qc = eve.intercept(qc, i, channel)
        bob.measure(qc, i, channel)

    matching = [i for i in sift_keys(alice.bases, bob.bases) if bob.measured_bits[i] is not None]
    a_key = alice.sift_key(matching)
    b_key = bob.sift_key(matching)
    qr: QBERResult = estimate_qber(a_key, b_key, sample_fraction, seed=(seed + 1000))
    eve_n = eve.intercepted_count if eve is not None else 0
    return AdaptiveRunResult(
        qber=qr.qber * 100,
        ci_low=qr.confidence_low * 100,
        ci_high=qr.confidence_high * 100,
        n_sifted=len(matching),
        eve_intercepted=eve_n,
    )


def build_scaled_channel(
    base_noise_model: str,
    f_scale: float,
    base_depolar_prob: float = 0.03,
    base_t1_ns: float = 500.0,
    base_t2_ns: float = 200.0,
    gate_time_ns: float = 50.0,
) -> QuantumChannel:
    """Same scaling rules as bb84_zne.build_scaled_config, but returns a
    bare QuantumChannel (no SimulationConfig/Eve coupling), so a custom
    adversary object can be attached independently."""
    if base_noise_model == NoiseModelType.DEPOLARIZING:
        return QuantumChannel(
            noise_model=NoiseModelType.DEPOLARIZING,
            depolar_prob=scale_depolar(base_depolar_prob, f_scale),
        )
    elif base_noise_model == NoiseModelType.AMPLITUDE_DAMPING:
        t1 = scale_amplitude_damping(base_t1_ns, gate_time_ns, f_scale)
        return QuantumChannel(
            noise_model=NoiseModelType.AMPLITUDE_DAMPING,
            t1_ns=t1, t2_ns=t1, gate_time_ns=gate_time_ns,
        )
    elif base_noise_model == NoiseModelType.PHASE_DAMPING:
        t2 = scale_phase_damping(base_t2_ns, gate_time_ns, f_scale)
        return QuantumChannel(
            noise_model=NoiseModelType.PHASE_DAMPING,
            t2_ns=t2, t1_ns=t2, gate_time_ns=gate_time_ns,
        )
    else:
        raise ValueError(f"Unsupported base_noise_model={base_noise_model!r} for ZNE scaling.")


# ──────────────────────────────────────────────────────────────────────
# 3. ZNE SWEEP AGAINST A CUSTOM ADVERSARY FACTORY
# ──────────────────────────────────────────────────────────────────────

def run_adaptive_zne_sweep(
    eve_factory: Callable[[float, int], object],
    f_scales: List[float],
    seeds: List[int],
    n_qubits: int = 2000,
    base_noise_model: str = NoiseModelType.DEPOLARIZING,
    base_depolar_prob: float = 0.03,
    base_t1_ns: float = 500.0,
    base_t2_ns: float = 200.0,
    gate_time_ns: float = 50.0,
    sample_fraction: float = 0.15,
) -> Dict:
    """
    Runs a ZNE-linear sweep exactly like bb84_zne.run_zne_sweep, but Eve
    is produced by `eve_factory(f_scale, seed) -> eve_object_or_None`
    instead of a fixed p_eve on a fixed static Eve. This lets the SAME
    ZNE-linear machinery be pointed at LinearScalingEve, BasisBiasedEve,
    a plain static bb84_core.Eve (as a control), or eve_factory returning
    None (no-Eve control) -- all through one code path, so the
    comparison is apples-to-apples.

    Returns dict with raw per-(f,seed) QBER, the mean per f, the
    ZNE-linear intercept, and the raw QBER at f=1 (what an operator
    would see without ZNE at all).
    """
    raw: Dict[float, list] = {}
    for f_scale in f_scales:
        channel = build_scaled_channel(
            base_noise_model, f_scale, base_depolar_prob, base_t1_ns, base_t2_ns, gate_time_ns,
        )
        vals = []
        for seed in seeds:
            eve = eve_factory(f_scale, seed)
            res = run_bb84_point(n_qubits, seed, channel, eve=eve, sample_fraction=sample_fraction)
            vals.append((res.qber, res.ci_low, res.ci_high))
        raw[f_scale] = vals

    mean_qbers, weights = [], []
    for f_scale in f_scales:
        vals = raw[f_scale]
        qbers_here = [v[0] for v in vals]
        halfwidths = [max(1e-3, (v[2] - v[1]) / 2) for v in vals]
        mean_qbers.append(float(np.mean(qbers_here)))
        weights.append(1.0 / (float(np.mean(halfwidths)) ** 2))

    a, b = linear_extrapolate(f_scales, mean_qbers, weights)
    zne_linear = max(0.0, a)
    qber_at_f1 = mean_qbers[f_scales.index(1.0)] if 1.0 in f_scales else mean_qbers[0]

    return {
        "raw": raw,
        "mean_qbers": dict(zip(f_scales, mean_qbers)),
        "zne_linear": zne_linear,
        "linear_slope": b,
        "qber_at_f1": qber_at_f1,
        "f_scales": f_scales,
    }
