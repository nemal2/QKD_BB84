"""
bb84_runner.py
==============
Simulation orchestrator for the BB84 QKD simulator.

Public API
----------
run_simulation(config, verbose=True)  → SimulationResult
run_comparison(scenarios)             → List[SimulationResult]
PRESET_SCENARIOS                      - six canonical Phase 1 scenarios
PHASE3_SCENARIOS                      - five Phase 3 noise-model scenarios

Pipeline steps
--------------
1. Quantum Transmission  (Alice → [Eve] → Bob)
2. Key Sifting           (basis reconciliation; lost photons excluded)
3. QBER Estimation       (random sample with Wilson 95 % CI)
4. Key Distillation      (remove QBER sample bits)

Phase 3 changes vs Phase 1
---------------------------
* QuantumChannel now constructed via QuantumChannel.from_config(config)
  so all five noise models are transparently selected.
* Bob.measured_bits[i] may be None (FIBRE_LOSS photon lost). Lost
  indices are tracked in a ``lost_qubits`` set, excluded from sifting,
  and counted in SimulationResult.n_lost.
* PHASE3_SCENARIOS preset list added.
* _print_header / _print_summary extended to show Phase 3 channel info
  and photon survival statistics.

Backward compatibility
----------------------
Phase 1 SimulationConfig objects (noise_model=None) produce identical
results because QuantumChannel.from_config() falls back to the legacy
noise_enabled / depolar_prob fields when noise_model is None.

University of Ruhuna – Dept. of Computer Engineering
MIT Licence – see LICENSE
"""

from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple

import numpy as np

from bb84_config import SimulationConfig, SimulationResult
from bb84_core   import Alice, Bob, Eve, estimate_qber
from bb84_noise  import QuantumChannel, NoiseModelType


# ──────────────────────────────────────────────────────────────────────
# PRESET SCENARIOS  (Phase 1 — kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────

PRESET_SCENARIOS: List[Tuple[str, SimulationConfig]] = [
    (
        "Ideal (no noise, no Eve)",
        SimulationConfig(n_qubits=600, seed=42, label="Ideal"),
    ),
    (
        "Eve - Partial Intercept (30 %)",
        SimulationConfig(n_qubits=600, seed=42,
                         eve_present=True, eve_intercept_prob=0.30,
                         label="Eve 30%"),
    ),
    (
        "Eve - Partial Intercept (50 %)",
        SimulationConfig(n_qubits=600, seed=42,
                         eve_present=True, eve_intercept_prob=0.50,
                         label="Eve 50%"),
    ),
    (
        "Eve - Full Intercept (100 %)",
        SimulationConfig(n_qubits=600, seed=42,
                         eve_present=True, eve_intercept_prob=1.0,
                         label="Eve 100%"),
    ),
    (
        "Channel Noise Only (p = 0.05)",
        SimulationConfig(n_qubits=600, seed=42,
                         noise_enabled=True, depolar_prob=0.05,
                         label="Noise p=0.05"),
    ),
    (
        "Eve (100 %) + Channel Noise",
        SimulationConfig(n_qubits=600, seed=42,
                         eve_present=True, eve_intercept_prob=1.0,
                         noise_enabled=True, depolar_prob=0.05,
                         label="Eve+Noise"),
    ),
]


# ──────────────────────────────────────────────────────────────────────
# PHASE 3 PRESET SCENARIOS
# ──────────────────────────────────────────────────────────────────────

PHASE3_SCENARIOS: List[Tuple[str, SimulationConfig]] = [
    (
        "Ideal (no noise)",
        SimulationConfig(
            n_qubits=500, seed=42,
            noise_model=NoiseModelType.IDEAL,
            label="Ideal",
        ),
    ),
    (
        "Depolarising (p = 0.05)",
        SimulationConfig(
            n_qubits=500, seed=42,
            noise_model=NoiseModelType.DEPOLARIZING,
            depolar_prob=0.05,
            label="Depolarising p=0.05",
        ),
    ),
    (
        "Amplitude Damping (T1 = 10 µs)",
        SimulationConfig(
            n_qubits=500, seed=42,
            noise_model=NoiseModelType.AMPLITUDE_DAMPING,
            t1_ns=10_000.0, gate_time_ns=50.0,
            label="Amplitude Damping",
        ),
    ),
    (
        "Phase Damping (T2 = 8 µs)",
        SimulationConfig(
            n_qubits=500, seed=42,
            noise_model=NoiseModelType.PHASE_DAMPING,
            t2_ns=8_000.0, gate_time_ns=50.0,
            label="Phase Damping",
        ),
    ),
    (
        "Combined T1+T2 (T1=10 µs, T2=8 µs)",
        SimulationConfig(
            n_qubits=500, seed=42,
            noise_model=NoiseModelType.COMBINED,
            t1_ns=10_000.0, t2_ns=8_000.0, gate_time_ns=50.0,
            label="Combined T1+T2",
        ),
    ),
    (
        "Fibre Loss (50 km)",
        SimulationConfig(
            n_qubits=500, seed=42,
            noise_model=NoiseModelType.FIBRE_LOSS,
            channel_length_km=50.0,
            label="Fibre Loss 50 km",
        ),
    ),
    (
        "Eve Intercept-Resend (100 %)",
        SimulationConfig(
            n_qubits=500, seed=42,
            noise_model=NoiseModelType.IDEAL,
            eve_present=True, eve_intercept_prob=1.0,
            label="Eve 100%",
        ),
    ),
]


# ──────────────────────────────────────────────────────────────────────
# SINGLE SIMULATION
# ──────────────────────────────────────────────────────────────────────

def run_simulation(
    config:  SimulationConfig,
    verbose: bool = True,
) -> SimulationResult:
    """
    Run one complete BB84 simulation.

    Parameters
    ----------
    config  : SimulationConfig instance.
    verbose : print step-by-step progress and a result summary.

    Returns
    -------
    SimulationResult with keys, QBER, timing, and statistics.

    Example
    -------
    >>> from bb84_config import SimulationConfig
    >>> from bb84_runner import run_simulation
    >>> result = run_simulation(SimulationConfig(n_qubits=500, seed=0))
    """
    if verbose:
        _print_header(config)

    start = time.time()

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    # ── Instantiate parties ───────────────────────────────────────────
    alice   = Alice(config.n_qubits, seed=config.seed)
    bob     = Bob(config.n_qubits,   seed=config.seed)

    # Phase 3: use from_config() factory — selects the correct noise model
    loss_rng = random.Random(config.seed if config.seed is not None else None)
    channel  = QuantumChannel.from_config(config, loss_rng=loss_rng)

    eve = Eve(config.eve_intercept_prob, seed=config.seed) if config.eve_present else None

    if verbose:
        print(f"\n  [1/4] Quantum Transmission  ({config.n_qubits} qubits)...")
        print(f"        Channel : {channel.description}")

    # ── Step 1: Quantum Transmission ─────────────────────────────────
    lost_qubits: set = set()

    for i in range(config.n_qubits):
        if verbose and i % 200 == 0:
            print(f"        ↳ {i}/{config.n_qubits}", end="\r")

        qc = alice.prepare_qubit(i)

        if eve is not None:
            qc = eve.intercept(qc, i, channel)

        result_bit = bob.measure(qc, i, channel)

        # FIBRE_LOSS: None means photon never reached Bob
        if result_bit is None:
            lost_qubits.add(i)

    if verbose:
        print(f"        ↳ {config.n_qubits}/{config.n_qubits} transmitted  "
              f"({len(lost_qubits)} lost)  ✓   ")

    # ── Step 2: Key Sifting ───────────────────────────────────────────
    if verbose:
        print(f"\n  [2/4] Key Sifting...")

    # Exclude positions where: bases differ OR photon was lost
    matching_indices = [
        i for i in range(config.n_qubits)
        if alice.bases[i] == bob.bases[i]
        and i not in lost_qubits
    ]
    alice_sifted = alice.sift_key(matching_indices)
    bob_sifted   = bob.sift_key(matching_indices)
    sift_rate    = len(matching_indices) / config.n_qubits

    if verbose:
        print(f"        ↳ {len(matching_indices)} bits retained  "
              f"({sift_rate:.1%}, expected ~50 % of arriving qubits)")

    # ── Step 3: QBER Estimation ───────────────────────────────────────
    if verbose:
        print(f"\n  [3/4] QBER Estimation  "
              f"(sample fraction = {config.sample_fraction:.0%})...")

    # Guard: if all photons were lost the sifted key is empty — return
    # a zeroed result immediately (key rate = 0, QBER undefined → 0).
    if len(alice_sifted) == 0:
        if verbose:
            print("        ↳ No sifted bits (all photons lost) — skipping QBER.")
        from bb84_config import QBERResult
        empty_qber = QBERResult(
            qber=0.0, errors=0, sample_size=0,
            security_status="SECURE ok",
            confidence_low=0.0, confidence_high=1.0,
        )
        return SimulationResult(
            config=config,
            n_transmitted=config.n_qubits,
            n_sifted=0,
            sifted_key_rate=0.0,
            qber_result=empty_qber,
            alice_final_key=[],
            bob_final_key=[],
            key_agreement_rate=0.0,
            eve_interception_rate=0.0,
            runtime_seconds=time.time() - start,
            n_lost=len(lost_qubits),
        )

    qber_result = estimate_qber(
        alice_sifted,
        bob_sifted,
        config.sample_fraction,
        seed=(config.seed + 1000) if config.seed is not None else None,
    )

    if verbose:
        print(f"        ↳ QBER   : {qber_result.qber * 100:.2f} %  "
              f"(95 % CI [{qber_result.confidence_low * 100:.1f}, "
              f"{qber_result.confidence_high * 100:.1f}] %)")
        print(f"        ↳ Errors : {qber_result.errors} / {qber_result.sample_size}")
        print(f"        ↳ Status : {qber_result.security_status}")

    # ── Step 4: Key Distillation ──────────────────────────────────────
    s           = qber_result.sample_size
    alice_final = alice_sifted[s:]
    bob_final   = bob_sifted[s:]

    if verbose:
        print(f"\n  [4/4] Key Distillation  →  {len(alice_final)} bits")

    agreement = (
        sum(a == b for a, b in zip(alice_final, bob_final)) / len(alice_final)
        if alice_final else 0.0
    )
    eve_rate = (
        eve.intercepted_count / config.n_qubits if eve is not None else 0.0
    )

    result = SimulationResult(
        config=config,
        n_transmitted=config.n_qubits,
        n_sifted=len(matching_indices),
        sifted_key_rate=sift_rate,
        qber_result=qber_result,
        alice_final_key=alice_final,
        bob_final_key=bob_final,
        key_agreement_rate=agreement,
        eve_interception_rate=eve_rate,
        runtime_seconds=time.time() - start,
        n_lost=len(lost_qubits),
    )

    if verbose:
        _print_summary(result)

    return result


# ──────────────────────────────────────────────────────────────────────
# MULTI-SCENARIO COMPARISON
# ──────────────────────────────────────────────────────────────────────

def run_comparison(
    scenarios: Optional[List[Tuple[str, SimulationConfig]]] = None,
    phase3:    bool = False,
) -> List[SimulationResult]:
    """
    Run a list of scenarios and print a compact comparison table.

    Parameters
    ----------
    scenarios : list of ``(name, SimulationConfig)`` pairs.
                Defaults to ``PHASE3_SCENARIOS`` if phase3=True,
                else ``PRESET_SCENARIOS``.
    phase3    : convenience flag to select PHASE3_SCENARIOS.

    Returns
    -------
    List of SimulationResult objects in the same order as *scenarios*.
    """
    if scenarios is None:
        scenarios = PHASE3_SCENARIOS if phase3 else PRESET_SCENARIOS

    max_name = max(len(name) for name, _ in scenarios)
    col      = max(max_name + 2, 40)

    print("\n" + "═" * (col + 42))
    print("  BB84 QKD – MULTI-SCENARIO COMPARISON")
    print("  University of Ruhuna – Dept. of Computer Engineering")
    print("═" * (col + 42))
    print(f"  {'Scenario':<{col}} {'QBER':>6}  {'Key':>5}  {'Lost':>5}  Status")
    print("  " + "─" * (col + 36))

    results: List[SimulationResult] = []
    for name, cfg in scenarios:
        r = run_simulation(cfg, verbose=False)
        results.append(r)
        lost_str = f"{r.n_lost:>4}" if r.n_lost > 0 else "   –"
        print(f"  {name:<{col}} {r.qber_result.qber * 100:>5.1f}%  "
              f"{r.key_length:>5}b  {lost_str}   {r.qber_result.security_status}")

    print("═" * (col + 42))
    return results


# ──────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ──────────────────────────────────────────────────────────────────────

def _print_header(config: SimulationConfig) -> None:
    w = 66
    print("\n" + "═" * w)
    print("  BB84 QKD SIMULATOR  –  Phase 3")
    print("  University of Ruhuna – Dept. of Computer Engineering")
    print("═" * w)
    print(f"  Label        : {config.label}")
    print(f"  Qubits       : {config.n_qubits}")
    if config.eve_present:
        print(f"  Eve Present  : True  (intercept p = {config.eve_intercept_prob})")
    else:
        print(f"  Eve Present  : False")

    # Phase 3: show resolved noise model
    if config.noise_model is not None:
        print(f"  Noise Model  : {config.noise_model}")
    elif config.noise_enabled:
        print(f"  Noise        : Depolarising  p = {config.depolar_prob}  (legacy)")
    else:
        print(f"  Noise        : Ideal (none)")

    print(f"  QBER Sample  : {config.sample_fraction:.0%} of sifted key")
    print(f"  Seed         : {config.seed}")
    print("═" * w)


def _print_summary(r: SimulationResult) -> None:
    w = 66
    qr = r.qber_result
    print(f"\n{'─' * w}")
    print("  RESULT SUMMARY")
    print(f"{'─' * w}")
    print(f"  Transmitted          : {r.n_transmitted} qubits")
    if r.n_lost > 0:
        print(f"  Lost (fibre)         : {r.n_lost} qubits  "
              f"(survival {r.photon_survival_rate:.1%})")
    print(f"  After sifting        : {r.n_sifted} bits  ({r.sifted_key_rate:.1%})")
    print(f"  Final key length     : {r.key_length} bits")
    print(f"  Key generation rate  : {r.key_generation_rate:.4f} bits/qubit")
    print(f"  QBER                 : {qr.qber * 100:.2f} %")
    print(f"  95 % CI              : [{qr.confidence_low * 100:.1f} %, "
          f"{qr.confidence_high * 100:.1f} %]")
    print(f"  Security status      : {qr.security_status}")
    print(f"  Key agreement        : {r.key_agreement_rate * 100:.2f} %")
    if r.eve_interception_rate > 0:
        print(f"  Eve intercept rate   : {r.eve_interception_rate * 100:.1f} %")
    print(f"  Runtime              : {r.runtime_seconds:.2f} s")
    print(f"{'─' * w}")
    ak = r.alice_final_key[:30]
    bk = r.bob_final_key[:30]
    print(f"\n  Alice key (first {len(ak)}) : {ak}")
    print(f"  Bob   key (first {len(bk)}) : {bk}")
    print(f"  Keys fully match     : {r.keys_match}")