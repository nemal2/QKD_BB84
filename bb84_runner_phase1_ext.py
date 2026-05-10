from __future__ import annotations
import time
from typing import List, Optional, Tuple
from bb84_config import SimulationConfig, SimulationResult
from bb84_runner import run_simulation

PHASE1_NOTEBOOK_SCENARIOS = [
    ("Ideal (no noise, no Eve)",
     SimulationConfig(n_qubits=600, seed=42, label="Ideal")),
    
    ("Eve — Partial Intercept (30 %)",
     SimulationConfig(n_qubits=600, eve_present=True,
                      eve_intercept_prob=0.30, seed=42, label="Eve 30%")),
    ("Eve — Partial Intercept (50 %)",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,
                      eve_intercept_prob=0.5, label="Eve 50%")),
    ("Eve — Full Intercept (100 %)",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,
                      eve_intercept_prob=1.0, label="Eve 100%")),
    ("Channel Noise Only (p = 0.05)",
     SimulationConfig(n_qubits=600, noise_enabled=True,seed=42,
                      depolar_prob=0.05, label="Noise p=0.05")),
    ("Eve (100 %) + Channel Noise",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,
                      eve_intercept_prob=1.0, noise_enabled=True,
                      depolar_prob=0.05, label="Eve+Noise")),
]

PHASE1_NOTEBOOK_SCENARIOS_sample_frac = [
    ("Ideal (no noise, no Eve)",
     SimulationConfig(n_qubits=600, seed=42,sample_fraction=0.20, label="Ideal")),
    
    ("Eve — Partial Intercept (30 %)",
     SimulationConfig(n_qubits=600, eve_present=True,
                      eve_intercept_prob=0.30, seed=42,sample_fraction=0.20, label="Eve 30%")),
    ("Eve — Partial Intercept (50 %)",
     SimulationConfig(n_qubits=600, eve_present=True,seed=42,sample_fraction=0.20,
                      eve_intercept_prob=0.5, label="Eve 50%")),
    ("Eve — Full Intercept (100 %)",
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
        SimulationConfig(n_qubits=1000, eve_present=False,
                         noise_enabled=False, seed=42,
                         label="n1000 Ideal"),
    ),
    (
        "n=1000 · Eve 30 % — borderline (no noise)",
        SimulationConfig(n_qubits=1000, eve_present=True,
                         eve_intercept_prob=0.30, noise_enabled=False,
                         seed=42, label="n1000 Eve30%"),
    ),
    (
        "n=1000 · Eve 50 % — Partial Intercept (no noise)",
        SimulationConfig(n_qubits=1000, eve_present=True,
                         eve_intercept_prob=0.50, noise_enabled=False,
                         seed=42, label="n1000 Eve50%"),
    ),
    (
        "n=1000 · Eve 100 % — Full Intercept (no noise)",
        SimulationConfig(n_qubits=1000, eve_present=True,
                         eve_intercept_prob=1.00, noise_enabled=False,
                         seed=42, label="n1000 Eve100%"),
    ),
    (
        "n=1000 · Noise only (p=0.05)",
        SimulationConfig(n_qubits=1000, eve_present=False,
                         noise_enabled=True, depolar_prob=0.05,
                         seed=42, label="n1000 Noise"),
    ),
    (
        "n=1000 · Eve (100 %) + Channel Noise",
        SimulationConfig(n_qubits=1000, eve_present=True,seed=42,
                      eve_intercept_prob=1.0, noise_enabled=True,
                      depolar_prob=0.05, label="Eve+Noise"))
]

ALL_PHASE1_SCENARIOS = PHASE1_NOTEBOOK_SCENARIOS + PHASE1_DETECTION_PROOF


def _p1_header(cfg):
    w = 60
    print("\n" + "=" * w)
    print("  BB84 QKD SIMULATOR — Phase 1 Baseline")
    print("  University of Ruhuna, Dept. of Computer Engineering")
    print("=" * w)
    print(f"  Label        : {cfg.label}")
    print(f"  Qubits       : {cfg.n_qubits}")
    eve_str = (f"True  (intercept p = {cfg.eve_intercept_prob})"
               if cfg.eve_present else "False")
    print(f"  Eve Present  : {eve_str}")
    print(f"  Noise        : {cfg.noise_enabled}"
          + (f"  (depolar p = {cfg.depolar_prob})" if cfg.noise_enabled else ""))
    print(f"  QBER Sample  : {cfg.sample_fraction:.0%} of sifted key")
    print(f"  Seed         : {cfg.seed}")
    print("=" * w)


def _p1_summary(r):
    w = 60
    qr = r.qber_result
    print(f"\n{chr(45)*w}")
    print("  RESULT SUMMARY")
    print(f"{chr(45)*w}")
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
    print(f"{chr(45)*w}")
    ak = r.alice_final_key[:30]
    bk = r.bob_final_key[:30]
    print(f"\n  Alice key (first {len(ak)}) : {ak}")
    print(f"  Bob   key (first {len(bk)}) : {bk}")
    print(f"  Keys fully match     : {r.keys_match}")


def run_simulation_p1(config, verbose=True):
    if verbose:
        _p1_header(config)
    result = run_simulation(config, verbose=False)
    if not verbose:
        return result
    qr = result.qber_result
    print(f"\n  [1/4] Quantum Transmission  ({config.n_qubits} qubits)...")
    if config.noise_enabled:
        print(f"        [Channel] Depolarizing noise  p = {config.depolar_prob}")
    print(f"        \u21b3 {config.n_qubits}/{config.n_qubits} qubits sent \u2713")
    print(f"\n  [2/4] Key Sifting (basis reconciliation)...")
    print(f"        \u21b3 {result.n_sifted} bits retained "
          f"({result.sifted_key_rate:.1%} — expected ~50 %)")
    print(f"\n  [3/4] QBER Estimation (sample fraction = {config.sample_fraction:.0%})...")
    print(f"        \u21b3 QBER   : {qr.qber * 100:.2f} %  "
          f"(95 % CI [{qr.confidence_low * 100:.1f} %, "
          f"{qr.confidence_high * 100:.1f} %])")
    print(f"        \u21b3 Errors : {qr.errors} / {qr.sample_size}")
    print(f"        \u21b3 Status : {qr.security_status}")
    print(f"\n  [4/4] Key Distillation...")
    print(f"        \u21b3 Final key : {result.key_length} bits")
    _p1_summary(result)
    return result


def run_comparison_p1(scenarios=None, verbose_each=False):
    if scenarios is None:
        scenarios = PHASE1_NOTEBOOK_SCENARIOS
    print("\n" + "=" * 60)
    print("  MULTI-SCENARIO COMPARISON")
    print("=" * 60)
    results = []
    for name, cfg in scenarios:
        r = (run_simulation_p1(cfg, verbose=True)
             if verbose_each else run_simulation(cfg, verbose=False))
        results.append(r)
        print(f"\n  \u25b6  {name}")
        print(f"     QBER = {r.qber_result.qber * 100:.1f} %  \u2502  "
              f"Key = {r.key_length} bits  \u2502  "
              f"Status = {r.qber_result.security_status}  \u2502  "
              f"{r.runtime_seconds:.1f}s")
    print("\n" + "=" * 60)
    return results
