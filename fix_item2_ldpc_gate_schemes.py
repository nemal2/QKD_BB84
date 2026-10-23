"""
fix_item2_ldpc_gate_schemes.py  (CORRECTED)
============================================
Re-runs basis-aware vs basis-unaware LDPC under de-idealized gate encodings.

WHY THE FIRST VERSION WAS WRONG
--------------------------------
The original script used inject_errors() — random uniform bit flips at a
target QBER rate.  That makes p_rect ≈ p_diag ≈ qber_target, which destroys
the basis asymmetry that basis-aware LDPC exploits.  No asymmetry → no gain,
which is exactly what the broken output showed (all schemes identical, aware
sometimes *worse* than unaware).

This version uses the actual Qiskit-Aer simulator to generate errors, exactly
as reconciliation.generate_sifted_pool() does, but with gate_scheme support
added (one line change: alice.prepare_qubit(i, gate_scheme=gate_scheme)).
Under real depolarizing noise, the diagonal basis goes through more gates than
rectilinear, so p_diag > p_rect naturally — that asymmetry is what the
basis-aware prior exploits, and it changes depending on the gate scheme.

WHAT THIS SCRIPT ANSWERS (Item 2)
----------------------------------
The paper measured 5–10× FER and +3 to +9 pp net-key under the PRIMARY
encoding, which has a zero-gate rectilinear-bit-0 path.  This script re-runs
the same experiment under Scheme A (zero-gate path removed, asymmetry reduced)
and Scheme B (gate counts equalized, asymmetry removed entirely).  The result
shows how much of the gain survives on a realistic encoding.

USAGE
-----
  python fix_item2_ldpc_gate_schemes.py

  Set N_TRIALS_FAST = True at the top for a 5-minute sanity check (10 trials)
  before the full 40-trial run.  Expect 30–90 minutes total on a laptop.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from bb84_core import Alice, Bob
from bb84_noise import QuantumChannel, NoiseModelType
from reconciliation import LDPCReconciler, h2, SiftedPool

# ── experiment knobs ────────────────────────────────────────────────
N_TRIALS_FAST  = False           # True → 10 trials for quick sanity check
N_TRIALS       = 10 if N_TRIALS_FAST else 40
BLOCK_LEN      = 1000
SEED_BASE      = 2000
SCHEMES        = ["PRIMARY", "A", "B"]
BATCH_SIZE     = 8000

# Target aggregate QBERs.  We calibrate depolar_prob for each (scheme, target)
# pair so the comparison is at matched noise levels.
QBER_TARGETS   = [0.02, 0.06, 0.10]

# Pool size: 20× block_len gives stable per-basis estimates
POOL_SIZE      = BLOCK_LEN * 20


# ── gate_scheme-aware pool generator ────────────────────────────────
# This is generate_sifted_pool() from reconciliation.py with ONE change:
# alice.prepare_qubit(i, gate_scheme=gate_scheme) instead of
# alice.prepare_qubit(i).  Everything else is byte-identical.

def generate_pool_with_scheme(
    n_sifted_target: int,
    depolar_prob: float,
    seed: int,
    gate_scheme: Optional[str] = None,
) -> SiftedPool:
    """
    Generate a sifted key pool with real simulator noise.
    gate_scheme=None → PRIMARY encoding (original bb84_core behavior).
    gate_scheme='A'  → Scheme A (id gate added to every path).
    gate_scheme='B'  → Scheme B (all paths equalized to 2 gates).
    """
    n_qubits = int(n_sifted_target * 2.2) + 64
    alice    = Alice(n_qubits, seed=seed)
    bob      = Bob(n_qubits, seed=seed)
    channel  = QuantumChannel(
        noise_model=NoiseModelType.DEPOLARIZING,
        depolar_prob=depolar_prob,
    )
    sim = channel._simulator

    circuits = []
    for i in range(n_qubits):
        # ← only change vs reconciliation.generate_sifted_pool()
        qc = alice.prepare_qubit(i, gate_scheme=gate_scheme)
        if bob.bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)
        circuits.append(qc)

    measured = np.empty(n_qubits, dtype=np.uint8)
    for start in range(0, n_qubits, BATCH_SIZE):
        chunk = circuits[start : start + BATCH_SIZE]
        job = sim.run(chunk, shots=1,
                      seed_simulator=(seed * 7919 + start) % (2**31))
        result = job.result()
        for j in range(len(chunk)):
            counts = result.get_counts(j)
            measured[start + j] = int(list(counts.keys())[0])

    a_bases    = np.array(alice.bases)
    b_bases    = np.array(bob.bases)
    match      = a_bases == b_bases
    a_sift     = np.array(alice.bits, dtype=np.uint8)[match]
    b_sift     = measured[match]
    bases_sift = a_bases[match]

    if len(a_sift) < n_sifted_target:
        raise RuntimeError(
            f"Sifting shortfall: {len(a_sift)} < {n_sifted_target}. "
            "Increase head-room."
        )

    return SiftedPool(
        alice_bits  = a_sift[:n_sifted_target].copy(),
        bob_bits    = b_sift[:n_sifted_target].copy(),
        depolar_prob= depolar_prob,
        n_transmitted=n_qubits,
        alice_bases = bases_sift[:n_sifted_target].copy(),
        bit_values  = a_sift[:n_sifted_target].copy(),
    )


# ── measure actual per-basis error rates from a pool ────────────────

def measure_per_basis_qber(pool: SiftedPool) -> Tuple[float, float, float]:
    """
    Return (p_rect, p_diag, p_aggregate) from a pool's ground-truth bits.
    This simulates using the sacrificed QBER sample — we use the full pool
    here since it's a simulation.
    """
    errs      = pool.alice_bits != pool.bob_bits
    rect_mask = pool.alice_bases == 0
    diag_mask = pool.alice_bases == 1

    p_rect = float(np.mean(errs[rect_mask])) if rect_mask.any() else 0.0
    p_diag = float(np.mean(errs[diag_mask])) if diag_mask.any() else 0.0
    p_agg  = float(np.mean(errs))
    return p_rect, p_diag, p_agg


# ── calibrate depolar_prob to hit a target aggregate QBER ────────────

def calibrate_depolar(
    target_qber: float,
    scheme: str,
    seed: int = 9999,
    tol: float = 0.003,
    max_iter: int = 15,
) -> float:
    """
    Bisect to find the depolar_prob that produces target_qber ± tol under
    the given gate scheme.  Uses a small pool (4000 bits) per probe.
    """
    gate_scheme = None if scheme == "PRIMARY" else scheme
    lo, hi = 0.001, 0.40

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        pool = generate_pool_with_scheme(4000, mid, seed, gate_scheme)
        _, _, p_agg = measure_per_basis_qber(pool)
        if abs(p_agg - target_qber) < tol:
            return mid
        if p_agg < target_qber:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── helpers ──────────────────────────────────────────────────────────

def h2_inv(target: float) -> float:
    """Inverse binary entropy: find p in [0,0.5] such that h2(p) = target."""
    if target <= 0.0: return 0.0
    if target >= 1.0: return 0.5
    lo, hi = 1e-9, 0.5 - 1e-9
    for _ in range(60):
        mid = (lo + hi) / 2
        (lo if h2(mid) < target else hi).__class__   # dummy
        if h2(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── one paired trial ─────────────────────────────────────────────────

@dataclass
class TrialResult:
    fer_unaware:  float
    fer_sameRate: float
    fer_pipeline: float
    net_unaware:  float
    net_sameRate: float
    net_pipeline: float
    p_rect: float
    p_diag: float
    p_agg:  float


def run_trial(
    scheme: str,
    depolar_prob: float,
    seed: int,
    recon: LDPCReconciler,
) -> TrialResult:
    """
    One paired (unaware / aware-same-rate / aware-pipeline) trial.
    Errors come from the real simulator, not injected.
    """
    gate_scheme = None if scheme == "PRIMARY" else scheme

    pool = generate_pool_with_scheme(POOL_SIZE, depolar_prob, seed, gate_scheme)
    p_rect, p_diag, p_agg = measure_per_basis_qber(pool)

    # Block for reconciliation
    a_block     = pool.alice_bits[:BLOCK_LEN]
    b_block     = pool.bob_bits[:BLOCK_LEN]
    bases_block = pool.alice_bases[:BLOCK_LEN]

    # Per-position prior
    p_vec = np.where(bases_block == 0, p_rect, p_diag).astype(float)
    p_vec = np.clip(p_vec, 1e-6, 0.499)

    # Mixture-entropy effective p for pipeline rate selection
    f_rect = float(np.mean(bases_block == 0))
    f_diag = 1.0 - f_rect
    mix_h  = f_rect * h2(max(p_rect, 1e-9)) + f_diag * h2(max(p_diag, 1e-9))
    rate_p_eff = h2_inv(mix_h) if mix_h > 0 else p_agg

    # 1. Basis-UNAWARE (uniform prior, rate from aggregate)
    r_u = recon.reconcile(a_block, b_block, p_est=p_agg)

    # 2. Basis-AWARE same-rate (per-pos prior, rate still from aggregate)
    r_s = recon.reconcile(a_block, b_block, p_est=p_agg, p_vec=p_vec)

    # 3. Basis-AWARE pipeline (per-pos prior + mixture-entropy rate)
    r_p = recon.reconcile(a_block, b_block, p_est=p_agg,
                          p_vec=p_vec, rate_p=rate_p_eff)

    return TrialResult(
        fer_unaware  = 0.0 if r_u.actually_correct else 1.0,
        fer_sameRate = 0.0 if r_s.actually_correct else 1.0,
        fer_pipeline = 0.0 if r_p.actually_correct else 1.0,
        net_unaware  = r_u.net_key_fraction,
        net_sameRate = r_s.net_key_fraction,
        net_pipeline = r_p.net_key_fraction,
        p_rect=p_rect, p_diag=p_diag, p_agg=p_agg,
    )


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Item 2 Fix (CORRECTED): Basis-aware LDPC under de-idealized schemes")
    print(f"  block_len={BLOCK_LEN}  trials={N_TRIALS}  "
          f"pool_size={POOL_SIZE}")
    if N_TRIALS_FAST:
        print("  *** FAST MODE (10 trials) — set N_TRIALS_FAST=False for paper ***")
    print("=" * 70)

    # ── Step 1: calibrate depolar_prob for each (scheme, qber_target) ──
    print("\nCalibrating depolar_prob for each scheme × QBER target ...")
    depolar_map: Dict[Tuple[str, float], float] = {}
    for scheme in SCHEMES:
        for qt in QBER_TARGETS:
            print(f"  {scheme:8s}  QBER_target={qt:.0%} ...", end=" ", flush=True)
            dp = calibrate_depolar(qt, scheme, seed=8888)
            depolar_map[(scheme, qt)] = dp
            print(f"depolar_prob = {dp:.4f}")

    # ── Step 2: run trials ──────────────────────────────────────────
    all_results: Dict[Tuple[str, float], List[TrialResult]] = {}

    for scheme in SCHEMES:
        print(f"\n── Scheme: {scheme} ──")
        recon = LDPCReconciler(n=BLOCK_LEN, seed=SEED_BASE)

        for qt in QBER_TARGETS:
            dp = depolar_map[(scheme, qt)]
            print(f"  QBER_target={qt:.0%}  (depolar={dp:.4f}) ", end="", flush=True)
            trials: List[TrialResult] = []
            t0 = time.perf_counter()

            for trial in range(N_TRIALS):
                seed = SEED_BASE + trial * 100
                res  = run_trial(scheme, dp, seed, recon)
                trials.append(res)
                if (trial + 1) % 10 == 0:
                    print(".", end="", flush=True)

            elapsed = time.perf_counter() - t0
            all_results[(scheme, qt)] = trials
            print(f"  done in {elapsed:.0f}s")

    # ── Step 3: print results ───────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("RESULTS — Per-basis error rates (source of basis-aware gain)")
    print("=" * 80)
    print(f"{'Scheme':10s} {'QBER_tgt':8s}  {'p_rect':8s}  {'p_diag':8s}  "
          f"{'p_agg':8s}  {'D/R ratio':10s}")
    print("-" * 60)
    for scheme in SCHEMES:
        for qt in QBER_TARGETS:
            trials = all_results[(scheme, qt)]
            p_r  = np.mean([t.p_rect for t in trials])
            p_d  = np.mean([t.p_diag for t in trials])
            p_a  = np.mean([t.p_agg  for t in trials])
            ratio = p_d / p_r if p_r > 1e-6 else float("inf")
            print(f"{scheme:10s} {qt:8.0%}  {p_r:8.4f}  {p_d:8.4f}  "
                  f"{p_a:8.4f}  {ratio:10.2f}x")

    print("\n  KEY: a D/R ratio >> 1 means there is asymmetry for basis-aware to exploit.")
    print("  PRIMARY has the largest asymmetry (p_rect ≈ 0 for bit=0).")
    print("  Scheme B (equalized) should have ratio ≈ 1 — the gain should vanish.\n")

    print("\n" + "=" * 80)
    print("RESULTS — FER (Frame Error Rate)")
    print("=" * 80)
    hdr = (f"{'Scheme':10s} {'QBER_tgt':8s}  {'Unaware':9s}  "
           f"{'Aware-SR':10s}  {'Aware-PL':10s}  {'FER ratio (PL/unaware)':22s}")
    print(hdr)
    print("-" * 78)
    for scheme in SCHEMES:
        for qt in QBER_TARGETS:
            trials = all_results[(scheme, qt)]
            fer_u  = np.mean([t.fer_unaware  for t in trials])
            fer_s  = np.mean([t.fer_sameRate for t in trials])
            fer_p  = np.mean([t.fer_pipeline for t in trials])
            ratio  = fer_p / fer_u if fer_u > 1e-4 else 0.0
            flag   = ""
            if ratio < 0.5:
                flag = "  ← >2× improvement"
            print(f"{scheme:10s} {qt:8.0%}  {fer_u:9.3f}  "
                  f"{fer_s:10.3f}  {fer_p:10.3f}  {ratio:22.3f}{flag}")

    print("\n" + "=" * 80)
    print("RESULTS — Net Key Fraction")
    print("=" * 80)
    hdr2 = (f"{'Scheme':10s} {'QBER_tgt':8s}  {'Unaware':9s}  "
            f"{'Aware-SR':10s}  {'Aware-PL':10s}  {'Delta (pp)':12s}")
    print(hdr2)
    print("-" * 68)
    for scheme in SCHEMES:
        for qt in QBER_TARGETS:
            trials = all_results[(scheme, qt)]
            nk_u   = np.mean([t.net_unaware  for t in trials])
            nk_s   = np.mean([t.net_sameRate for t in trials])
            nk_p   = np.mean([t.net_pipeline for t in trials])
            delta  = (nk_p - nk_u) * 100
            flag   = "  ← positive gain" if delta > 1.0 else ""
            print(f"{scheme:10s} {qt:8.0%}  {nk_u:9.3f}  "
                  f"{nk_s:10.3f}  {nk_p:10.3f}  {delta:+12.1f} pp{flag}")

    # ── Step 4: summary for paper ───────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SUMMARY FOR PAPER (how gains change as encoding is de-idealized)")
    print("=" * 80)
    print("""
What to write in the paper (replacing the deferred Future Work item):

For each QBER level, the table below shows FER ratio (aware-pipeline / unaware)
and net-key gain (pp).  PRIMARY = original encoding (zero-gate path present).
Scheme A = minimal fix (zero-gate path removed).  Scheme B = fully equalized.
""")
    for qt in QBER_TARGETS:
        print(f"  QBER ≈ {qt:.0%}")
        for scheme in SCHEMES:
            trials = all_results[(scheme, qt)]
            fer_u  = np.mean([t.fer_unaware  for t in trials])
            fer_p  = np.mean([t.fer_pipeline for t in trials])
            nk_u   = np.mean([t.net_unaware  for t in trials])
            nk_p   = np.mean([t.net_pipeline for t in trials])
            ratio  = fer_p / fer_u if fer_u > 1e-4 else 0.0
            delta  = (nk_p - nk_u) * 100
            p_r    = np.mean([t.p_rect for t in trials])
            p_d    = np.mean([t.p_diag for t in trials])
            print(f"    {scheme:8s}  FER ratio={ratio:.3f}  "
                  f"Δnet-key={delta:+.1f}pp  "
                  f"(p_rect={p_r:.4f}, p_diag={p_d:.4f})")
        print()

    print("Update abstract numbers if Scheme A gains differ substantially from PRIMARY.")
    print("If gains survive under Scheme A → the technique is robust to the artifact.")
    print("If gains collapse under Scheme A → the abstract must be revised downward.")


if __name__ == "__main__":
    main()