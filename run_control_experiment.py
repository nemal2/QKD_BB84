"""
run_control_experiment.py
==========================
Control experiment for the Sec. VII caveat: does removing the literal
zero-noisy-gate path change the P(bit=1 | rectilinear error) = 1.000
result?

Compares baseline (gate_scheme=None, i.e. your existing model, UNCHANGED)
against control schemes A, B, C from bb84_gate_schemes.py, at a scan of
depolarizing-noise strengths, under the DEPOLARIZING model (same model
family used for the original Table VII / calibrated-QBER numbers).

Nothing here modifies any existing experiment: it only imports the same
bb84_core / bb84_noise objects your other notebooks already use, and
only activates new code paths when gate_scheme is explicitly passed.

Output: one row per (scheme, depolar_prob) with:
    overall QBER, per-basis error counts, and P(bit=1 | error) for each
    basis -- the exact quantity reported in Table VII.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from bb84_core import Alice, Bob
from bb84_noise import QuantumChannel, NoiseModelType
from bb84_gate_schemes import gate_count_table


def generate_labeled_pool(n_qubits, depolar_prob, seed, gate_scheme=None, batch_size=8000):
    """Batched sifted-pool generation, mirroring reconciliation.generate_sifted_pool
    but exposing gate_scheme (baseline behavior is bit-for-bit identical when
    gate_scheme=None)."""
    alice = Alice(n_qubits, seed=seed)
    bob = Bob(n_qubits, seed=seed)
    channel = QuantumChannel(
        noise_model=NoiseModelType.DEPOLARIZING if depolar_prob > 0 else NoiseModelType.IDEAL,
        depolar_prob=depolar_prob,
    )
    sim = channel._simulator

    circuits = []
    for i in range(n_qubits):
        qc = alice.prepare_qubit(i, gate_scheme=gate_scheme)
        if bob.bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)
        circuits.append(qc)

    measured = np.empty(n_qubits, dtype=np.uint8)
    for start in range(0, n_qubits, batch_size):
        chunk = circuits[start:start + batch_size]
        job = sim.run(chunk, shots=1, seed_simulator=(seed * 7919 + start) % (2 ** 31))
        result = job.result()
        for j in range(len(chunk)):
            counts = result.get_counts(j)
            measured[start + j] = int(list(counts.keys())[0])

    a_bits = np.array(alice.bits, dtype=np.uint8)
    a_bases = np.array(alice.bases)
    b_bases = np.array(bob.bases)
    match = a_bases == b_bases

    return a_bits[match], measured[match], a_bases[match]


def basis_bit_correlation_row(scheme_label, depolar_prob, alice_bits, bob_bits, bases, n_transmitted):
    overall_qber = float(np.mean(alice_bits != bob_bits))
    row = {
        "scheme": scheme_label,
        "depolar_prob": depolar_prob,
        "n_sifted": len(alice_bits),
        "overall_qber": overall_qber,
    }
    for basis, name in ((0, "rect"), (1, "diag")):
        sel = bases == basis
        err = sel & (alice_bits != bob_bits)
        n_err = int(err.sum())
        p_bit1_given_error = float(np.mean(alice_bits[err] == 1)) if n_err > 0 else float("nan")
        row[f"{name}_errors"] = n_err
        row[f"P(bit=1|{name}_error)"] = p_bit1_given_error
    return row


def main(n_qubits=16000, seed=7, batch_size=8000, probs=None):
    if probs is None:
        # Six noise strengths, same flavor as the paper's "six QBER levels"
        probs = [0.01, 0.02, 0.03, 0.04, 0.06, 0.10]

    schemes = [None, "A", "B", "C"]
    rows = []

    print("Gate-count tables per scheme (bit,basis) -> #gates:")
    print("  baseline: {(0,0):0,(1,0):1,(0,1):1,(1,1):2}  [no id gates in original model]")
    for s in ("A", "B", "C"):
        print(f"  scheme {s}: {gate_count_table(s)}")
    print()

    t_start = time.time()
    for scheme in schemes:
        label = scheme if scheme is not None else "baseline (current model)"
        for p in probs:
            t0 = time.time()
            a_bits, b_bits, bases = generate_labeled_pool(
                n_qubits=n_qubits, depolar_prob=p, seed=seed, gate_scheme=scheme, batch_size=batch_size
            )
            row = basis_bit_correlation_row(label, p, a_bits, b_bits, bases, n_qubits)
            row["runtime_s"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"  [{label:>26s}]  p={p:<5}  QBER={row['overall_qber']*100:5.2f}%  "
                  f"rect_err={row['rect_errors']:<4}  P(bit=1|rect_err)={row['P(bit=1|rect_error)']}  "
                  f"diag_err={row['diag_errors']:<4}  P(bit=1|diag_err)={row['P(bit=1|diag_error)']:.3f}  "
                  f"[{row['runtime_s']}s]")

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = main()
    df.to_csv("control_experiment_results.csv", index=False)
    print("\nSaved control_experiment_results.csv")