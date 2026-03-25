"""
bb84_runner.py
==============
Simulation orchestrator and preset scenario definitions.

Exports
-------
run_simulation()    single end-to-end BB84 run
run_comparison()    multi-scenario batch runner
PRESET_SCENARIOS    five standard research scenarios

Internal
--------
_print_header()     console header before a run
_print_summary()    console summary after a run

Phase 5 placeholder
-------------------
After QBER estimation, a stub for error correction is clearly
marked below.  Drop Phase 5 code in that one spot.
"""

from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple

import numpy as np

from bb84_config import SimulationConfig, SimulationResult
from bb84_core   import Alice, Bob, Eve, QuantumChannel, sift_keys, estimate_qber




PRESET_SCENARIOS: List[Tuple[str, SimulationConfig]] = [
    (
        "Ideal (no noise, no Eve)",
        SimulationConfig(
            n_qubits=600, eve_present=False,
            noise_enabled=False, label="Ideal",
        ),
    ),
    (
        "Eve — Full Intercept (100 %)",
        SimulationConfig(
            n_qubits=600, eve_present=True,
            eve_intercept_prob=1.0, noise_enabled=False,
            label="Eve 100%",
        ),
    ),
    (
        "Eve — Partial Intercept (50 %)",
        SimulationConfig(
            n_qubits=600, eve_present=True,
            eve_intercept_prob=0.5, noise_enabled=False,
            label="Eve 50%",
        ),
    ),
    (
        "Channel Noise Only (p = 0.05)",
        SimulationConfig(
            n_qubits=600, eve_present=False,
            noise_enabled=True, depolar_prob=0.05,
            label="Noise p=0.05",
        ),
    ),
    (
        "Eve (100 %) + Channel Noise",
        SimulationConfig(
            n_qubits=600, eve_present=True,
            eve_intercept_prob=1.0, noise_enabled=True,
            depolar_prob=0.05, label="Eve+Noise",
        ),
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(
    config:  SimulationConfig,
    verbose: bool = True,
) -> SimulationResult:
    """
    Full BB84 pipeline for one SimulationConfig.

    Steps
    ─────
    1. Quantum transmission  (Alice → [Eve] → Bob)
    2. Key sifting           (basis reconciliation)
    3. QBER estimation       (random sample comparison)
    4. Key distillation      (remove sampled bits)
       └── Phase 5 stub:     error correction + privacy amplification

    Returns a SimulationResult with all metrics.
    """
    if verbose:
        _print_header(config)

    start = time.time()

    # ── Seed global RNGs for reproducibility ──────────────────────────────
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    # ── Instantiate parties ───────────────────────────────────────────────
    alice   = Alice(config.n_qubits, seed=config.seed)
    bob     = Bob(config.n_qubits,   seed=config.seed)
    channel = QuantumChannel(config.noise_enabled, config.depolar_prob)
    eve     = Eve(config.eve_intercept_prob, seed=config.seed) \
              if config.eve_present else None

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1 — Quantum Transmission
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [1/4] Quantum Transmission  ({config.n_qubits} qubits)...")

    for i in range(config.n_qubits):
        if verbose and i % 250 == 0:
            print(f"        ↳ {i}/{config.n_qubits} qubits sent", end="\r")

        qc = alice.prepare_qubit(i)

        if eve:
            qc = eve.intercept(qc, i, channel)      # Phase 4: swap attack here

        bob.measure(qc, i, channel)

    if verbose:
        print(f"        ↳ {config.n_qubits}/{config.n_qubits} qubits sent ✓")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — Key Sifting
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [2/4] Key Sifting (basis reconciliation)...")

    matching_indices = sift_keys(alice.bases, bob.bases)
    alice_sifted     = alice.sift_key(matching_indices)
    bob_sifted       = bob.sift_key(matching_indices)
    sift_rate        = len(matching_indices) / config.n_qubits

    if verbose:
        print(f"        ↳ {len(matching_indices)} bits retained "
              f"({sift_rate:.1%} — expected ~50 %)")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3 — QBER Estimation
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [3/4] QBER Estimation "
              f"(sample fraction = {config.sample_fraction:.0%})...")

    qber_result = estimate_qber(
        alice_sifted, bob_sifted,
        config.sample_fraction,
        seed=config.seed,
    )

    if verbose:
        print(f"        ↳ QBER   : {qber_result.qber * 100:.2f} %  "
              f"(95 % CI [{qber_result.confidence_low * 100:.1f} %, "
              f"{qber_result.confidence_high * 100:.1f} %])")
        print(f"        ↳ Errors : {qber_result.errors} / {qber_result.sample_size}")
        print(f"        ↳ Status : {qber_result.security_status}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4 — Key Distillation  (Phase 5 stub lives here)
    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: simply strip the bits consumed by QBER sampling.
    s           = qber_result.sample_size
    alice_final = alice_sifted[s:]
    bob_final   = bob_sifted[s:]

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  PHASE 5 HOOK — Error Correction + Privacy Amplification           │
    # │  Uncomment and implement when Phase 5 lands:                        │
    # │                                                                     │
    # │  if config.ecc_scheme != "none":                                    │
    # │      alice_final, bob_final = apply_error_correction(              │
    # │          alice_final, bob_final, scheme=config.ecc_scheme)         │
    # │  if config.privacy_amp:                                             │
    # │      alice_final = privacy_amplify(alice_final, qber_result.qber)  │
    # │      bob_final   = privacy_amplify(bob_final,   qber_result.qber)  │
    # └─────────────────────────────────────────────────────────────────────┘

    if verbose:
        print(f"\n  [4/4] Key Distillation...")
        print(f"        ↳ Final key : {len(alice_final)} bits")

    # ── Key agreement (ideal = 1.0 with no noise / Eve) ───────────────────
    if alice_final:
        matches   = sum(a == b for a, b in zip(alice_final, bob_final))
        agreement = matches / len(alice_final)
    else:
        agreement = 0.0

    eve_rate = (eve.intercepted_count / config.n_qubits) if eve else 0.0
    runtime  = time.time() - start

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
        runtime_seconds=runtime,
    )

    if verbose:
        _print_summary(result)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-SCENARIO COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(
    scenarios: Optional[List[Tuple[str, SimulationConfig]]] = None,
) -> List[SimulationResult]:
    """
    Run a list of (name, config) pairs and return all SimulationResults.
    Delegates plotting to bb84_plots.py so this module stays plot-free.

    Usage in notebook
    ─────────────────
    from bb84_runner import run_comparison, PRESET_SCENARIOS
    from bb84_plots  import plot_comparison

    results = run_comparison(PRESET_SCENARIOS)
    plot_comparison(PRESET_SCENARIOS, results)
    """
    if scenarios is None:
        scenarios = PRESET_SCENARIOS

    print("\n" + "═" * 60)
    print("  MULTI-SCENARIO COMPARISON")
    print("═" * 60)

    results: List[SimulationResult] = []
    for name, cfg in scenarios:
        print(f"\n  ▶  {name}")
        r = run_simulation(cfg, verbose=False)
        results.append(r)
        print(f"     QBER = {r.qber_result.qber * 100:.1f} %  │  "
              f"Key = {r.key_length} bits  │  "
              f"Status = {r.qber_result.security_status}  │  "
              f"{r.runtime_seconds:.1f}s")

    print("\n" + "═" * 60)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT HELPERS  (internal)
# ──────────────────────────────────────────────────────────────────────────────

def _print_header(config: SimulationConfig) -> None:
    print("\n" + "═" * 60)
    print("  BB84 QKD SIMULATOR — Phase 1 Baseline")
    print("  University of Ruhuna, Dept. of Computer Engineering")
    print("═" * 60)
    print(f"  Label        : {config.label}")
    print(f"  Qubits       : {config.n_qubits}")
    eve_str = (f"  (intercept p = {config.eve_intercept_prob})"
               if config.eve_present else "")
    print(f"  Eve Present  : {config.eve_present}{eve_str}")
    noise_str = (f"  (depolar p = {config.depolar_prob})"
                 if config.noise_enabled else "")
    print(f"  Noise        : {config.noise_enabled}{noise_str}")
    print(f"  QBER Sample  : {config.sample_fraction:.0%} of sifted key")
    print(f"  Seed         : {config.seed}")
    print("═" * 60)


def _print_summary(r: SimulationResult) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print("  RESULT SUMMARY")
    print(line)
    print(f"  Transmitted          : {r.n_transmitted} qubits")
    print(f"  After sifting        : {r.n_sifted} bits  ({r.sifted_key_rate:.1%})")
    print(f"  Final key length     : {r.key_length} bits")
    print(f"  Key generation rate  : {r.key_generation_rate:.4f} bits / qubit")
    print(f"  QBER                 : {r.qber_result.qber * 100:.2f} %")
    print(f"  95 % CI              : [{r.qber_result.confidence_low * 100:.1f} %, "
          f"{r.qber_result.confidence_high * 100:.1f} %]")
    print(f"  Security status      : {r.qber_result.security_status}")
    print(f"  Key agreement        : {r.key_agreement_rate * 100:.2f} %")
    if r.config.eve_present:
        print(f"  Eve intercept rate   : {r.eve_interception_rate * 100:.1f} %")
    print(f"  Runtime              : {r.runtime_seconds:.2f}s")
    print(line)
    print(f"\n  Alice key (first 30) : {r.alice_final_key[:30]}")
    print(f"  Bob   key (first 30) : {r.bob_final_key[:30]}")
    print(f"  Keys fully match     : {r.keys_match}")
    print()