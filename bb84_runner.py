"""
bb84_runner.py  ·  Simulation Orchestrator
══════════════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Central pipeline that wires all phases together.

Import surface (for notebooks):
    from bb84_runner import run_simulation, run_comparison, PRESET_SCENARIOS

Pipeline steps
──────────────
1. Quantum Transmission  (Alice → [Eve] → Bob)           ← Phase 1
2. Key Sifting           (basis reconciliation)           ← Phase 1
3. QBER Estimation       (random sample)                  ← Phase 1
4. Key Distillation      (remove sample bits)             ← Phase 1
5. Post-Processing       (error correction + privacy amp) ← Phase 5
6. Security Analysis     (information theory metrics)     ← Phase 5

Phase progression
─────────────────
Phase 3: noise handled by QuantumChannel.from_config()  ← NOW ACTIVE
Phase 4: bb84_attacks.py  (not yet implemented — guarded below)
Phase 5: bb84_postprocessing.py + bb84_analysis.py  (guarded below)
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple

import numpy as np

from bb84_config import SimulationConfig, SimulationResult
from bb84_noise  import QuantumChannel                      # Phase 3 ✓
from bb84_core   import Alice, Bob, sift_keys, estimate_qber

# ── Phase 4 — optional until bb84_attacks.py is created ──────────────
try:
    from bb84_attacks import (
        build_attacker, InterceptResendEve,
        PNSAttack, EntanglementAttack,
    )
    _PHASE4_AVAILABLE = True
except ImportError:
    _PHASE4_AVAILABLE = False

    class InterceptResendEve:           # minimal stub so isinstance() works
        intercepted_count = 0
        def intercept(self, qc, i, ch): return qc

    class PNSAttack:
        intercepted_count = 0
        eve_info_fraction = 0.0
        detection_rate    = 0.0
        def intercept(self, qc, i, ch, bit, base): return qc, False

    class EntanglementAttack:
        intercepted_count = 0
        def intercept(self, qc, i, ch): return qc

    def build_attacker(config):
        """Phase 1 fallback: intercept-resend Eve, no Phase 4 attacks."""
        if not config.eve_present:
            return None
        eve = InterceptResendEve()
        eve.intercept_prob = config.eve_intercept_prob
        eve._rng           = np.random.default_rng(config.seed)
        # Use the Phase 1 Eve from bb84_core if available
        try:
            from bb84_core import Eve
            return Eve(config.eve_intercept_prob, seed=config.seed)
        except ImportError:
            return None

# ── Phase 5 — optional until bb84_postprocessing.py is created ───────
try:
    from bb84_postprocessing import run_post_processing
    _PHASE5_PP_AVAILABLE = True
except ImportError:
    _PHASE5_PP_AVAILABLE = False
    def run_post_processing(*args, **kwargs):
        raise RuntimeError("bb84_postprocessing.py not yet implemented (Phase 5).")

try:
    from bb84_analysis import compute_security_analysis
    _PHASE5_SA_AVAILABLE = True
except ImportError:
    _PHASE5_SA_AVAILABLE = False
    def compute_security_analysis(result, verbose=False):
        """Phase 5 stub — no-op until bb84_analysis.py exists."""
        return None


# ══════════════════════════════════════════════════════════════════════
# PRESET SCENARIOS
# ══════════════════════════════════════════════════════════════════════

# Phase 1 + Phase 3 scenarios (Phase 4/5 presets commented out until ready)
PRESET_SCENARIOS: List[Tuple[str, SimulationConfig]] = [
    # ── Phase 1 ───────────────────────────────────────────────────────
    (
        "Ideal (no noise, no Eve)",
        SimulationConfig(n_qubits=600, label="Ideal"),
    ),
    (
        "Eve — Full Intercept (100 %)",
        SimulationConfig(n_qubits=600, eve_present=True,
                         eve_intercept_prob=1.0, label="Eve 100%"),
    ),
    (
        "Eve — Partial Intercept (50 %)",
        SimulationConfig(n_qubits=600, eve_present=True,
                         eve_intercept_prob=0.5, label="Eve 50%"),
    ),
    # ── Phase 3 ───────────────────────────────────────────────────────
    (
        "Depolarizing (p=0.05)",
        SimulationConfig(n_qubits=600, noise_enabled=True,
                         noise_model="depolarizing", depolar_prob=0.05,
                         label="Depolarizing"),
    ),
    (
        "Amplitude Damping (T1=10 µs)",
        SimulationConfig(n_qubits=600, noise_enabled=True,
                         noise_model="amplitude_damp",
                         t1_ns=10_000, gate_time_ns=50,
                         label="Amp.Damp T1=10µs"),
    ),
    (
        "Phase Damping (T2=5 µs)",
        SimulationConfig(n_qubits=600, noise_enabled=True,
                         noise_model="phase_damp",
                         t2_ns=5_000, gate_time_ns=50,
                         label="Phase.Damp T2=5µs"),
    ),
    (
        "Combined T1+T2 (realistic)",
        SimulationConfig(n_qubits=600, noise_enabled=True,
                         noise_model="combined",
                         t1_ns=10_000, t2_ns=8_000, gate_time_ns=50,
                         label="Combined"),
    ),
    (
        "Fiber Loss (50 km)",
        SimulationConfig(n_qubits=800, noise_enabled=True,
                         noise_model="fiber_loss", channel_length_km=50,
                         label="Fiber 50km"),
    ),
    # ── Phase 4 presets go here (uncomment when bb84_attacks.py is ready)
    # (
    #     "PNS Attack (μ=0.5)",
    #     SimulationConfig(n_qubits=600, eve_present=True,
    #                      attack_type="pns", mean_photon_number=0.5, ...),
    # ),
]

# Shorter set for quick notebook comparisons
PHASE1_SCENARIOS = PRESET_SCENARIOS[:3]
PHASE3_SCENARIOS = PRESET_SCENARIOS[3:]


# ══════════════════════════════════════════════════════════════════════
# SINGLE SIMULATION
# ══════════════════════════════════════════════════════════════════════

def run_simulation(
    config:  SimulationConfig,
    verbose: bool = True,
) -> SimulationResult:
    """
    Full BB84 pipeline for one SimulationConfig.

    Phase 3 changes vs Phase 1
    ──────────────────────────
    • QuantumChannel.from_config() now selects the correct noise model
      based on config.noise_model
    • Lost photons (fiber_loss) are excluded from sifting automatically
    • Steps count is still reported as 1-6 for consistency with the full
      platform; steps 5-6 are stubs until Phase 5 is implemented
    """
    if verbose:
        _print_header(config)

    start = time.time()

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    # ── Instantiate parties ────────────────────────────────────────────
    alice   = Alice(config.n_qubits, seed=config.seed)
    bob     = Bob(config.n_qubits,   seed=config.seed)
    channel = QuantumChannel.from_config(config)    # Phase 3 ✓

    attacker = build_attacker(config)
    is_pns   = isinstance(attacker, PNSAttack)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1 — Quantum Transmission
    # ═══════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [1/6] Quantum Transmission  ({config.n_qubits} qubits)...")
        print(f"        Channel: {channel.description}")

    lost_qubits: set = set()

    for i in range(config.n_qubits):
        if verbose and i % 250 == 0:
            print(f"        ↳ {i}/{config.n_qubits}", end="\r")

        qc = alice.prepare_qubit(i)

        # ── Attack ────────────────────────────────────────────────────
        if attacker is not None:
            if is_pns:
                qc, blocked = attacker.intercept(
                    qc, i, channel, alice.bits[i], alice.bases[i],
                )
                if blocked:
                    lost_qubits.add(i)
                    bob.measured_bits[i] = None
                    continue
            else:
                qc = attacker.intercept(qc, i, channel)

        # ── Bob measures ──────────────────────────────────────────────
        result_bit = bob.measure(qc, i, channel)
        if result_bit is None:
            lost_qubits.add(i)   # fiber_loss photon absorption

    if verbose:
        loss_str = (f"  ({len(lost_qubits)} photons lost)"
                    if lost_qubits else "")
        print(f"        ↳ {config.n_qubits}/{config.n_qubits} transmitted ✓{loss_str}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2 — Key Sifting  (exclude lost photons)
    # ═══════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [2/6] Key Sifting...")

    valid = [i for i in range(config.n_qubits)
             if i not in lost_qubits and bob.measured_bits[i] is not None]
    matching_indices = [i for i in valid if alice.bases[i] == bob.bases[i]]
    alice_sifted = alice.sift_key(matching_indices)
    bob_sifted   = bob.sift_key(matching_indices)
    sift_rate    = len(matching_indices) / config.n_qubits

    if verbose:
        print(f"        ↳ {len(matching_indices)} bits  "
              f"({sift_rate:.1%} of all transmitted, expected ~50 % of received)")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3 — QBER Estimation
    # ═══════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [3/6] QBER Estimation ({config.sample_fraction:.0%} sample)...")

    qber_result = estimate_qber(
        alice_sifted, bob_sifted,
        config.sample_fraction,
        seed=config.seed,
    )

    if verbose:
        print(f"        ↳ QBER   : {qber_result.qber * 100:.2f} %  "
              f"(95 % CI [{qber_result.confidence_low * 100:.1f}, "
              f"{qber_result.confidence_high * 100:.1f}] %)")
        print(f"        ↳ Errors : {qber_result.errors} / {qber_result.sample_size}")
        print(f"        ↳ Status : {qber_result.security_status}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4 — Key Distillation
    # ═══════════════════════════════════════════════════════════════════
    s           = qber_result.sample_size
    alice_final = alice_sifted[s:]
    bob_final   = bob_sifted[s:]

    if verbose:
        print(f"\n  [4/6] Key Distillation  →  {len(alice_final)} bits")

    if alice_final:
        agreement = sum(a == b for a, b in zip(alice_final, bob_final)) \
                    / len(alice_final)
    else:
        agreement = 0.0

    # ── Attack statistics ─────────────────────────────────────────────
    eve_rate      = 0.0
    pns_multi_rate = 0.0
    pns_det_rate   = 0.0
    if attacker is not None:
        eve_rate = (attacker.intercepted_count / config.n_qubits
                    if hasattr(attacker, "intercepted_count") else 0.0)
        if is_pns:
            pns_multi_rate = attacker.eve_info_fraction
            pns_det_rate   = attacker.detection_rate

    runtime = time.time() - start

    result = SimulationResult(
        config                = config,
        n_transmitted         = config.n_qubits,
        n_sifted              = len(matching_indices),
        sifted_key_rate       = sift_rate,
        qber_result           = qber_result,
        alice_final_key       = alice_final,
        bob_final_key         = bob_final,
        key_agreement_rate    = agreement,
        eve_interception_rate = eve_rate,
        runtime_seconds       = runtime,
        pns_multi_photon_rate = pns_multi_rate,
        pns_detection_rate    = pns_det_rate,
    )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5 — Post-Processing  (Phase 5 — not yet implemented)
    # ═══════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [5/6] Post-Processing skipped (Phase 5 — not yet implemented)")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 6 — Security Analysis  (Phase 5 — not yet implemented)
    # ═══════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [6/6] Security Analysis skipped (Phase 5 — not yet implemented)")

    if verbose:
        _print_summary(result)

    return result


# ══════════════════════════════════════════════════════════════════════
# MULTI-SCENARIO COMPARISON
# ══════════════════════════════════════════════════════════════════════

def run_comparison(
    scenarios: Optional[List[Tuple[str, SimulationConfig]]] = None,
) -> List[SimulationResult]:
    """
    Run a list of named scenarios and return all SimulationResults.
    Plotting is in bb84_plots.py / bb84_phase3_plots.py.
    """
    if scenarios is None:
        scenarios = PRESET_SCENARIOS

    print("\n" + "═" * 65)
    print("  MULTI-SCENARIO COMPARISON")
    print("═" * 65)
    print(f"  {'Scenario':<38} {'QBER':>6}  {'Key':>5}  Status")
    print("  " + "─" * 60)

    results: List[SimulationResult] = []
    for name, cfg in scenarios:
        r = run_simulation(cfg, verbose=False)
        results.append(r)
        print(f"  {name:<38} {r.qber_result.qber * 100:>5.1f}%  "
              f"{r.key_length:>5}b  {r.qber_result.security_status}")

    print("═" * 65)
    return results


# ══════════════════════════════════════════════════════════════════════
# PRETTY PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════

def _print_header(config: SimulationConfig) -> None:
    print("\n" + "═" * 62)
    print("  BB84 QKD RESEARCH PLATFORM — University of Ruhuna")
    print("  Dept. of Computer Engineering")
    print("═" * 62)
    print(f"  Label        : {config.label}")
    print(f"  Qubits       : {config.n_qubits}")
    print(f"  Attack       : "
          + ("none" if not config.eve_present else
             f"{config.attack_type}  p={config.eve_intercept_prob}"))
    print(f"  Noise model  : "
          + (config.noise_model if config.noise_enabled else "ideal"))
    print(f"  Seed         : {config.seed}")
    print("═" * 62)


def _print_summary(r: SimulationResult) -> None:
    line = "─" * 62
    pp   = r.post_processing
    sa   = r.security_analysis
    print(f"\n{line}")
    print("  RESULT SUMMARY")
    print(line)
    print(f"  Transmitted        : {r.n_transmitted} qubits")
    print(f"  After sifting      : {r.n_sifted} bits  ({r.sifted_key_rate:.1%})")
    print(f"  Final key (raw)    : {r.key_length} bits")
    print(f"  Key gen. rate      : {r.key_generation_rate:.4f} bits/qubit")
    print(f"  QBER               : {r.qber_result.qber * 100:.2f} %")
    print(f"  Security status    : {r.qber_result.security_status}")
    print(f"  Key agreement      : {r.key_agreement_rate * 100:.2f} %")
    if r.eve_interception_rate > 0:
        print(f"  Eve intercept rate : {r.eve_interception_rate * 100:.1f} %")
    if pp and pp.privacy_amp_applied:
        print(f"  Secret key bits    : {pp.bits_after_privacy_amp}")
    if sa and sa.secret_key_rate_realistic > 0:
        print(f"  SKR (realistic)    : {sa.secret_key_rate_realistic:.4f} bits/qubit")
    print(f"  Runtime            : {r.runtime_seconds:.2f}s")
    print(line)
    print(f"\n  Alice key (first 20): {r.alice_final_key[:20]}")
    print(f"  Bob   key (first 20): {r.bob_final_key[:20]}")
    print()