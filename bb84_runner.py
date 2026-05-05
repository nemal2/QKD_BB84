"""
bb84_runner.py  (Phase 3 update)
=================================
Orchestration layer.  Now accepts noise_model so the notebook can
drive amplitude-damping, phase-damping, and fiber-loss channels from
bb84_noise.py without touching any other module.

Changes vs Phase 1
──────────────────
* run_simulation() gains  noise_model='depolarizing'  (default keeps
  Phase-1 behaviour identical)
* _build_channel() picks between bb84_core.QuantumChannel (depolar)
  and bb84_noise.QuantumChannel (all other models)
* PRESET_SCENARIOS unchanged — still works exactly as before
"""

from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple

import numpy as np

from bb84_config import SimulationConfig, SimulationResult
from bb84_core   import Alice, Bob, Eve, sift_keys, estimate_qber
from bb84_core   import QuantumChannel as _CoreChannel

# Noise module (Phase 3) — imported lazily so Phase 1 still works if missing
try:
    from bb84_noise import QuantumChannel as _NoiseChannel
    _HAS_NOISE = True
except ImportError:                          # pragma: no cover
    _HAS_NOISE = False


# ──────────────────────────────────────────────────────────────────────────────
# PRESET SCENARIOS  (unchanged from Phase 1)
# ──────────────────────────────────────────────────────────────────────────────

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
# CHANNEL FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def _build_channel(config: SimulationConfig, noise_model: str):
    """
    Return the right QuantumChannel for this run.

    noise_model values
    ──────────────────
    'depolarizing'   → bb84_core.QuantumChannel  (Phase 1, default)
    'amplitude_damp' → bb84_noise.QuantumChannel with T1 params
    'phase_damp'     → bb84_noise.QuantumChannel with T2 params
    'fiber_loss'     → bb84_noise.QuantumChannel with distance param
    'combined'       → bb84_noise.QuantumChannel combined model
    'ideal'          → bb84_core.QuantumChannel  (no noise)
    """
    if noise_model == "depolarizing":
        return _CoreChannel(config.noise_enabled, config.depolar_prob)

    if noise_model == "ideal":
        return _CoreChannel(noise_enabled=False, depolar_prob=0.0)

    # Phase-3 models
    if not _HAS_NOISE:
        raise ImportError(
            "bb84_noise.py not found. "
            "Place it in the same directory to use Phase-3 noise models."
        )

    return _NoiseChannel(
        noise_model      = noise_model,
        depolar_prob     = getattr(config, "depolar_prob",      0.0),
        t1_ns            = getattr(config, "t1_ns",             100_000.0),
        t2_ns            = getattr(config, "t2_ns",              50_000.0),
        gate_time_ns     = getattr(config, "gate_time_ns",           50.0),
        channel_length_km= getattr(config, "channel_length_km",       0.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(
    config:      SimulationConfig,
    verbose:     bool = True,
    noise_model: str  = "depolarizing",   # ← NEW param (default = Phase-1 behaviour)
) -> SimulationResult:
    """
    Full BB84 pipeline.

    Parameters
    ──────────
    config      : SimulationConfig
    verbose     : print progress + summary
    noise_model : which channel to use
                  'depolarizing' (default) | 'amplitude_damp' |
                  'phase_damp' | 'fiber_loss' | 'combined' | 'ideal'

    For Phase-3 noise models the config must carry the extra fields
    (t1_ns, t2_ns, gate_time_ns, channel_length_km).  The easiest way
    is to attach them directly:

        cfg = SimulationConfig(n_qubits=500, noise_enabled=False)
        cfg.t1_ns        = 40_000   # 40 µs T1
        cfg.gate_time_ns = 50
        result = run_simulation(cfg, noise_model='amplitude_damp')
    """
    if verbose:
        _print_header(config)

    start = time.time()

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    alice   = Alice(config.n_qubits, seed=config.seed)
    bob     = Bob(config.n_qubits,   seed=config.seed)
    channel = _build_channel(config, noise_model)
    eve     = Eve(config.eve_intercept_prob, seed=config.seed) \
              if config.eve_present else None

    # ── Step 1: Quantum Transmission ──────────────────────────────────────
    if verbose:
        print(f"\n  [1/4] Quantum Transmission  ({config.n_qubits} qubits)...")

    lost = 0   # photons lost in fiber model
    for i in range(config.n_qubits):
        if verbose and i % 250 == 0:
            print(f"        ↳ {i}/{config.n_qubits} qubits sent", end="\r")

        qc = alice.prepare_qubit(i)
        if eve:
            qc = eve.intercept(qc, i, channel)

        # fiber_loss model may return None for lost photons
        result_bit = channel.run_circuit(qc)
        if result_bit is None:
            bob.measured_bits[i] = None   # mark as lost
            lost += 1
        else:
            bob.measured_bits[i] = result_bit

    if verbose:
        print(f"        ↳ {config.n_qubits}/{config.n_qubits} qubits sent ✓"
              + (f"  ({lost} lost to fiber)" if lost else ""))

    # ── Step 2: Key Sifting ───────────────────────────────────────────────
    # Exclude positions where Bob's photon was lost
    valid = [i for i in range(config.n_qubits) if bob.measured_bits[i] is not None]
    alice_bases_valid = [alice.bases[i] for i in valid]
    bob_bases_valid   = [bob.bases[i]   for i in valid]

    matching_rel = sift_keys(alice_bases_valid, bob_bases_valid)
    matching_indices = [valid[j] for j in matching_rel]   # back to absolute idx

    alice_sifted = [alice.bits[i]         for i in matching_indices]
    bob_sifted   = [bob.measured_bits[i]  for i in matching_indices]

    sift_rate = len(matching_indices) / config.n_qubits

    if verbose:
        print(f"\n  [2/4] Key Sifting...")
        print(f"        ↳ {len(matching_indices)} bits retained ({sift_rate:.1%})")

    # ── Step 3: QBER Estimation ───────────────────────────────────────────
    if verbose:
        print(f"\n  [3/4] QBER Estimation (sample = {config.sample_fraction:.0%})...")

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

    # ── Step 4: Key Distillation ──────────────────────────────────────────
    s           = qber_result.sample_size
    alice_final = alice_sifted[s:]
    bob_final   = bob_sifted[s:]

    if verbose:
        print(f"\n  [4/4] Key Distillation...")
        print(f"        ↳ Final key : {len(alice_final)} bits")

    key_length          = len(alice_final)
    key_generation_rate = key_length / config.n_qubits

    if alice_final:
        matches            = sum(a == b for a, b in zip(alice_final, bob_final))
        key_agreement_rate = matches / key_length
    else:
        key_agreement_rate = 0.0

    keys_match     = (key_agreement_rate == 1.0)
    eve_rate       = (eve.intercepted_count / config.n_qubits) if eve else None
    runtime        = time.time() - start

    result = SimulationResult(
        label               = config.label,
        n_transmitted       = config.n_qubits,
        n_sifted            = len(matching_indices),
        key_length          = key_length,
        sifted_key_rate     = sift_rate,
        key_generation_rate = key_generation_rate,
        qber_result         = qber_result,
        key_agreement_rate  = key_agreement_rate,
        keys_match          = keys_match,
        runtime_seconds     = runtime,
        alice_key           = alice_final,
        bob_key             = bob_final,
        eve_intercept_rate  = eve_rate,
    )

    if verbose:
        _print_summary(result)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-SCENARIO COMPARISON  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(
    scenarios: Optional[List[Tuple[str, SimulationConfig]]] = None,
) -> List[SimulationResult]:
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
# PRETTY PRINT HELPERS  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def _print_header(config: SimulationConfig) -> None:
    print("\n" + "═" * 60)
    print("  BB84 QKD SIMULATOR — Phase 1 / 3")
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
    if r.eve_intercept_rate is not None:
        print(f"  Eve intercept rate   : {r.eve_intercept_rate * 100:.1f} %")
    print(f"  Runtime              : {r.runtime_seconds:.2f}s")
    print(line)
    print(f"\n  Alice key (first 30) : {r.alice_key[:30]}")
    print(f"  Bob   key (first 30) : {r.bob_key[:30]}")
    print(f"  Keys fully match     : {r.keys_match}")
    print()