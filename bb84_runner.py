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

# Phase 1 + Phase 2 scenarios 
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
    # ── Phase 2 ───────────────────────────────────────────────────────
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
   
]

# Shorter set for quick notebook comparisons
PHASE1_SCENARIOS = PRESET_SCENARIOS[:3]
PHASE3_SCENARIOS = PRESET_SCENARIOS[3:]



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
PHASE1_NOTEBOOK_SCENARIOS = [
    ("Ideal (no noise, no Eve)",
     SimulationConfig(n_qubits=600, seed=42,sample_fraction=0.15, label="Ideal")),
    
    ("Eve - Partial Intercept (30 %)",
     SimulationConfig(n_qubits=600, eve_present=True,
                      eve_intercept_prob=0.30,sample_fraction=0.15, seed=42, label="Eve 30%")),
    ("Eve - Partial Intercept (50 %)",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,
                      eve_intercept_prob=0.5,sample_fraction=0.15, label="Eve 50%")),
    ("Eve - Full Intercept (100 %)",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,
                      eve_intercept_prob=1.0,sample_fraction=0.15, label="Eve 100%")),
    ("Channel Noise Only (p = 0.05)",
     SimulationConfig(n_qubits=600, noise_enabled=True,seed=42,
                      depolar_prob=0.05,sample_fraction=0.15, label="Noise p=0.05")),
    ("Eve (100 %) + Channel Noise",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,
                      eve_intercept_prob=1.0, noise_enabled=True,
                      depolar_prob=0.05,sample_fraction=0.15, label="Eve+Noise")),
]

PHASE1_NOTEBOOK_SCENARIOS_sample_frac = [
    ("Ideal (no noise, no Eve)",
     SimulationConfig(n_qubits=600, seed=42,sample_fraction=0.20, label="Ideal")),
    
    ("Eve - Partial Intercept (30 %)",
     SimulationConfig(n_qubits=600, eve_present=True,
                      eve_intercept_prob=0.30, seed=42,sample_fraction=0.20, label="Eve 30%")),
    ("Eve - Partial Intercept (50 %)",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,sample_fraction=0.20,
                      eve_intercept_prob=0.5, label="Eve 50%")),
    ("Eve - Full Intercept (100 %)",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,sample_fraction=0.20,
                      eve_intercept_prob=1.0, label="Eve 100%")),
    ("Channel Noise Only (p = 0.05)",
     SimulationConfig(n_qubits=600, noise_enabled=True,seed=42,sample_fraction=0.20,
                      depolar_prob=0.05, label="Noise p=0.05")),
    ("Eve (100 %) + Channel Noise",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,sample_fraction=0.20,
                      eve_intercept_prob=1.0, noise_enabled=True,
                      depolar_prob=0.05, label="Eve+Noise")),
]

PHASE1_DETECTION_PROOF: List[Tuple[str, SimulationConfig]] = [
    (
        "n=1000 · Ideal (no noise, no Eve)",
        SimulationConfig(n_qubits=1000, eve_present=False,sample_fraction=0.15,
                         noise_enabled=False, seed=42,
                         label="n1000 Ideal"),
    ),
    (
        "n=1000 · Eve 30 % - borderline (no noise)",
        SimulationConfig(n_qubits=1000, eve_present=True,sample_fraction=0.15,
                         eve_intercept_prob=0.30, noise_enabled=False,
                         seed=42, label="n1000 Eve30%"),
    ),
    (
        "n=1000 · Eve 50 % - Partial Intercept (no noise)",
        SimulationConfig(n_qubits=1000, eve_present=True,sample_fraction=0.15,
                         eve_intercept_prob=0.50, noise_enabled=False,
                         seed=42, label="n1000 Eve50%"),
    ),
    (
        "n=1000 · Eve 100 % - Full Intercept (no noise)",
        SimulationConfig(n_qubits=1000, eve_present=True,sample_fraction=0.15,
                         eve_intercept_prob=1.00, noise_enabled=False,
                         seed=42, label="n1000 Eve100%"),
    ),
    (
        "n=1000 · Noise only (p=0.05)",
        SimulationConfig(n_qubits=1000, eve_present=False,sample_fraction=0.15,
                         noise_enabled=True, depolar_prob=0.05,
                         seed=42, label="n1000 Noise"),
    ),
    (
        "n=1000 · Eve (100 %) + Channel Noise",
        SimulationConfig(n_qubits=1000, eve_present=True,seed=42,sample_fraction=0.15,
                      eve_intercept_prob=1.0, noise_enabled=True,
                      depolar_prob=0.05, label="Eve+Noise"))
]


PHASE1_DETECTION_PROOF_frac: List[Tuple[str, SimulationConfig]] = [
    (
        "n=1000 · Ideal (no noise, no Eve)",
        SimulationConfig(n_qubits=1000, eve_present=False,sample_fraction=0.2,
                         noise_enabled=False, seed=42,
                         label="n1000 Ideal"),
    ),
    (
        "n=1000 · Eve 30 % - borderline (no noise)",
        SimulationConfig(n_qubits=1000, eve_present=True,sample_fraction=0.2,
                         eve_intercept_prob=0.30, noise_enabled=False,
                         seed=42, label="n1000 Eve30%"),
    ),
    (
        "n=1000 · Eve 50 % - Partial Intercept (no noise)",
        SimulationConfig(n_qubits=1000, eve_present=True,sample_fraction=0.2,
                         eve_intercept_prob=0.50, noise_enabled=False,
                         seed=42, label="n1000 Eve50%"),
    ),
    (
        "n=1000 · Eve 100 % - Full Intercept (no noise)",
        SimulationConfig(n_qubits=1000, eve_present=True,sample_fraction=0.2,
                         eve_intercept_prob=1.00, noise_enabled=False,
                         seed=42, label="n1000 Eve100%"),
    ),
    (
        "n=1000 · Noise only (p=0.05)",
        SimulationConfig(n_qubits=1000, eve_present=False,sample_fraction=0.2,
                         noise_enabled=True, depolar_prob=0.05,
                         seed=42, label="n1000 Noise"),
    ),
    (
        "n=1000 · Eve (100 %) + Channel Noise",
        SimulationConfig(n_qubits=1000, eve_present=True,seed=42,sample_fraction=0.2,
                      eve_intercept_prob=1.0, noise_enabled=True,
                      depolar_prob=0.05, label="Eve+Noise"))
]

PHASE1_DETECTION_PROOF_HIGH: List[Tuple[str, SimulationConfig]] = [
    (
        "n=2000 · Ideal (no noise, no Eve)",
        SimulationConfig(n_qubits=2000, eve_present=False,sample_fraction=0.15,
                         noise_enabled=False, seed=42,
                         label="n1000 Ideal"),
    ),
    (
        "n=2000 · Eve 30 % - borderline (no noise)",
        SimulationConfig(n_qubits=2000, eve_present=True,sample_fraction=0.15,
                         eve_intercept_prob=0.30, noise_enabled=False,
                         seed=42, label="n1000 Eve30%"),
    ),
    (
        "n=2000 · Eve 50 % - Partial Intercept (no noise)",
        SimulationConfig(n_qubits=2000, eve_present=True,sample_fraction=0.15,
                         eve_intercept_prob=0.50, noise_enabled=False,
                         seed=42, label="n1000 Eve50%"),
    ),
    (
        "n=2000 · Eve 100 % - Full Intercept (no noise)",
        SimulationConfig(n_qubits=2000, eve_present=True,sample_fraction=0.15,
                         eve_intercept_prob=1.00, noise_enabled=False,
                         seed=42, label="n1000 Eve100%"),
    ),
    (
        "n=2000 · Noise only (p=0.05)",
        SimulationConfig(n_qubits=2000, eve_present=False,sample_fraction=0.15,
                         noise_enabled=True, depolar_prob=0.05,
                         seed=42, label="n1000 Noise"),
    ),
    (
        "n=2000 · Eve (100 %) + Channel Noise",
        SimulationConfig(n_qubits=2000, eve_present=True,seed=42,sample_fraction=0.15,
                      eve_intercept_prob=1.0, noise_enabled=True,
                      depolar_prob=0.05, label="Eve+Noise"))
]

# Convenience: everything in one list for a complete comparison table
ALL_PHASE1_SCENARIOS = PHASE1_NOTEBOOK_SCENARIOS + PHASE1_DETECTION_PROOF


# ══════════════════════════════════════════════════════════════════════
# PHASE-1-STYLE VERBOSE OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════════

def _p1_header(cfg: SimulationConfig) -> None:
    """Prints the old Phase 1 four-line box header."""
    w = 60
    print("\n" + "═" * w)
    print("  BB84 QKD SIMULATOR — Phase 1 Baseline")
    print("  University of Ruhuna, Dept. of Computer Engineering")
    print("═" * w)
    print(f"  Label        : {cfg.label}")
    print(f"  Qubits       : {cfg.n_qubits}")
    eve_str = (f"True  (intercept p = {cfg.eve_intercept_prob})"
               if cfg.eve_present else "False")
    print(f"  Eve Present  : {eve_str}")
    print(f"  Noise        : {cfg.noise_enabled}"
          + (f"  (depolar p = {cfg.depolar_prob})" if cfg.noise_enabled else ""))
    print(f"  QBER Sample  : {cfg.sample_fraction:.0%} of sifted key")
    print(f"  Seed         : {cfg.seed}")
    print("═" * w)


def _p1_summary(r: SimulationResult) -> None:
    """Prints the old Phase 1 result block."""
    w = 60
    qr = r.qber_result
    print(f"\n{'─' * w}")
    print("  RESULT SUMMARY")
    print(f"{'─' * w}")
    print(f"  Transmitted          : {r.n_transmitted} qubits")
    print(f"  After sifting        : {r.n_sifted} bits  ({r.sifted_key_rate:.1%})")
    print(f"  Final key length     : {r.key_length} bits")
    print(f"  Key generation rate  : {r.key_generation_rate:.4f} bits / qubit")
    print(f"  QBER                 : {qr.qber * 100:.2f} %")
    print(f"  95 % CI              : [{qr.confidence_low * 100:.1f} %, "
          f"{qr.confidence_high * 100:.1f} %]")
    print(f"  Security status      : {qr.security_status}")
    print(f"  Key agreement        : {r.key_agreement_rate * 100:.2f} %")
    if r.eve_interception_rate > 0:
        print(f"  Eve intercept rate   : {r.eve_interception_rate * 100:.1f} %")
    print(f"  Runtime              : {r.runtime_seconds:.2f}s")
    print(f"{'─' * w}")

    ak = r.alice_final_key[:30]
    bk = r.bob_final_key[:30]
    print(f"\n  Alice key (first {len(ak)}) : {ak}")
    print(f"  Bob   key (first {len(bk)}) : {bk}")
    print(f"  Keys fully match     : {r.keys_match}")


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def run_simulation_p1(
    config: SimulationConfig,
    verbose: bool = True,
) -> SimulationResult:
    """
    Wrapper around run_simulation() that prints the old Phase 1
    [1/4]…[4/4] verbose format your existing ipynb cells expect.

    Steps printed
    ─────────────
    [1/4] Quantum Transmission
    [2/4] Key Sifting
    [3/4] QBER Estimation
    [4/4] Key Distillation
    (Post-processing / security analysis are Phase 5 — silently skipped)
    """
    if verbose:
        _p1_header(config)

    # Delegate all real work to the current runner (verbose=False so we
    # control all output ourselves).
    result = run_simulation(config, verbose=False)

    if not verbose:
        return result

    qr = result.qber_result

    # ── Step 1 ────────────────────────────────────────────────────────
    print(f"\n  [1/4] Quantum Transmission  ({config.n_qubits} qubits)...")
    if config.noise_enabled:
        print(f"        [Channel] Depolarizing noise  p = {config.depolar_prob}")
    print(f"        ↳ {config.n_qubits}/{config.n_qubits} qubits sent ✓")

    # ── Step 2 ────────────────────────────────────────────────────────
    print(f"\n  [2/4] Key Sifting (basis reconciliation)...")
    print(f"        ↳ {result.n_sifted} bits retained "
          f"({result.sifted_key_rate:.1%} — expected ~50 %)")

    # ── Step 3 ────────────────────────────────────────────────────────
    print(f"\n  [3/4] QBER Estimation (sample fraction = {config.sample_fraction:.0%})...")
    print(f"        ↳ QBER   : {qr.qber * 100:.2f} %  "
          f"(95 % CI [{qr.confidence_low * 100:.1f} %, "
          f"{qr.confidence_high * 100:.1f} %])")
    print(f"        ↳ Errors : {qr.errors} / {qr.sample_size}")
    print(f"        ↳ Status : {qr.security_status}")

    # ── Step 4 ────────────────────────────────────────────────────────
    print(f"\n  [4/4] Key Distillation...")
    print(f"        ↳ Final key : {result.key_length} bits")

    _p1_summary(result)
    return result


def run_comparison_p1(
    scenarios: Optional[List[Tuple[str, SimulationConfig]]] = None,
    verbose_each: bool = False,
) -> List[SimulationResult]:
    """
    Runs a scenario list and prints the old Cell 5 ▶ name / detail
    format.  Pass verbose_each=True to also print per-run step output.

    Parameters
    ──────────
    scenarios    : list of (name, SimulationConfig).  Defaults to
                   PHASE1_NOTEBOOK_SCENARIOS (the original five).
    verbose_each : if True, run_simulation_p1() verbose mode fires for
                   every scenario (noisy but informative).
    """
    if scenarios is None:
        scenarios = PHASE1_NOTEBOOK_SCENARIOS

    print("\n" + "═" * 60)
    print("  MULTI-SCENARIO COMPARISON")
    print("═" * 60)

    results: List[SimulationResult] = []
    for name, cfg in scenarios:
        if verbose_each:
            r = run_simulation_p1(cfg, verbose=True)
        else:
            r = run_simulation(cfg, verbose=False)

        results.append(r)

        # ── Old Cell 5 output style ────────────────────────────────
        status = r.qber_result.security_status
        qber_pct = r.qber_result.qber * 100
        print(f"\n  ▶  {name}")
        print(f"     QBER = {qber_pct:.1f} %  │  Key = {r.key_length} bits  "
              f"│  Status = {status}  │  {r.runtime_seconds:.1f}s")

    print("\n" + "═" * 60)
    return results




























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
        seed=(config.seed + 1000) if config.seed is not None else None,
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
    scenarios=None,
) -> list:
    if scenarios is None:
        scenarios = PRESET_SCENARIOS

    # Compute column width dynamically from the longest name
    max_name = max(len(name) for name, _ in scenarios)
    col = max(max_name + 2, 42)          # at least 42 chars wide

    print("\n" + "═" * (col + 30))
    print("  MULTI-SCENARIO COMPARISON")
    print("═" * (col + 30))
    print(f"  {'Scenario':<{col}} {'QBER':>6}  {'Key':>5}  Status")
    print("  " + "─" * (col + 26))

    results = []
    for name, cfg in scenarios:
        r = run_simulation(cfg, verbose=False)
        results.append(r)
        print(f"  {name:<{col}} {r.qber_result.qber * 100:>5.1f}%  "
              f"{r.key_length:>5}b  {r.qber_result.security_status}")

    print("═" * (col + 30))
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