"""
prep_uncertainty.py
===================
Classical ignorance vs Quantum uncertainty in QKD preparation.

The core distinction
─────────────────────
  Classical ignorance  — we have a definite state but lack knowledge of it.
                         The qubit IS |0⟩ or IS |1⟩; we just forgot which.
                         → Represented by a MIXED density matrix (Tr ρ² < 1).

  Quantum uncertainty  — there is no fact of the matter about the state.
                         The qubit genuinely has no definite value until measured.
                         → Represented by a PURE density matrix (Tr ρ² = 1).

Key experimental signature
──────────────────────────
  Both types can give the same probabilities in ONE basis.
  They DIFFER when measured in a COMPLEMENTARY (mutually unbiased) basis.
  That difference is measureable and is exploited in BB84.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List

from prep.hilbert import QubitState, BB84_STATES
from prep.density import DensityMatrix


# ── Canonical examples ────────────────────────────────────────────────────────

def classical_ignorance_half() -> Dict[str, Any]:
    """
    50 / 50 classical mixture of |0⟩ and |1⟩.

    Physical story: Alice flipped a fair coin, prepared |0⟩ (heads) or
    |1⟩ (tails), then handed Bob the qubit without telling him the result.
    The qubit IS in one definite state — Bob is simply ignorant of which.

    Density matrix:  ρ = ½|0⟩⟨0| + ½|1⟩⟨1| = I/2  (maximally mixed)
    """
    dm = DensityMatrix.from_ensemble(
        [QubitState.ket0(), QubitState.ket1()], [0.5, 0.5]
    )
    z0, z1 = dm.measure_prob("z")
    x0, x1 = dm.measure_prob("x")
    return {
        "name":        "Classical Ignorance",
        "subtitle":    "50 / 50 mixture of |0⟩ and |1⟩",
        "story":       (
            "Alice flipped a fair coin. Heads → prepared |0⟩. Tails → prepared |1⟩. "
            "Bob receives the qubit but was not told the coin outcome."
        ),
        "density_matrix": dm,
        "purity":      dm.purity(),
        "bloch_vector": dm.bloch_vector(),
        "bloch_radius": dm.bloch_radius(),
        "coherence":   dm.coherence(),
        "z_basis":     {"P(0)": z0, "P(1)": z1},
        "x_basis":     {"P(+)": x0, "P(-)": x1},
        "state_type":  "Mixed",
        "matrix_note": "Off-diagonal elements are zero — no quantum coherence.",
    }


def quantum_uncertainty_plus() -> Dict[str, Any]:
    """
    Pure superposition state  |+⟩ = (|0⟩ + |1⟩)/√2.

    Physical story: Alice actively prepared the qubit in the |+⟩ state.
    There is no hidden coin, no missing information. The qubit genuinely
    has no Z-basis value — that is a fact about quantum mechanics, not
    about our ignorance.

    Density matrix:  ρ = |+⟩⟨+| = ½[[1,1],[1,1]]
    """
    state = QubitState.plus()
    dm    = DensityMatrix.from_pure(state)
    z0, z1 = dm.measure_prob("z")
    x0, x1 = dm.measure_prob("x")
    return {
        "name":        "Quantum Uncertainty",
        "subtitle":    "Pure superposition |+⟩ = (|0⟩+|1⟩)/√2",
        "story":       (
            "Alice deliberately prepared the qubit in state |+⟩. "
            "No hidden variable determines the Z-basis outcome — "
            "quantum mechanics says the value does not exist until measured."
        ),
        "density_matrix": dm,
        "purity":      dm.purity(),
        "bloch_vector": dm.bloch_vector(),
        "bloch_radius": dm.bloch_radius(),
        "coherence":   dm.coherence(),
        "z_basis":     {"P(0)": z0, "P(1)": z1},
        "x_basis":     {"P(+)": x0, "P(-)": x1},
        "state_type":  "Pure",
        "matrix_note": "Off-diagonal elements are 0.5 — quantum coherence present.",
    }


# ── Side-by-side comparison ───────────────────────────────────────────────────

def compare_uncertainty_types() -> Dict[str, Any]:
    """
    Full comparison of classical ignorance vs quantum uncertainty.
    Highlights that both are indistinguishable in Z-basis but separable
    in X-basis (a mutually unbiased basis).
    """
    classical = classical_ignorance_half()
    quantum   = quantum_uncertainty_plus()

    z_same = np.isclose(
        classical["z_basis"]["P(0)"], quantum["z_basis"]["P(0)"], atol=1e-6
    )
    x_differs = not np.isclose(
        classical["x_basis"]["P(+)"], quantum["x_basis"]["P(+)"], atol=1e-6
    )

    return {
        "classical":               classical,
        "quantum":                 quantum,
        "z_basis_indistinguishable": z_same,
        "x_basis_distinguishable": x_differs,
        "key_insight": (
            "In the Z-basis both give P(0) = P(1) = 0.5 — they look identical. "
            "Switch to the X-basis: the pure state |+⟩ gives P(+) = 1.0 "
            "while the mixed state gives P(+) = 0.5. That difference is real "
            "and measurable — it is the fingerprint of quantum coherence."
        ),
        "qkd_relevance": (
            "Alice's diagonal basis states |+⟩ and |−⟩ are pure quantum states. "
            "An eavesdropper who guesses the wrong basis destroys the coherence, "
            "turning a pure state into a mixed one and introducing detectable errors."
        ),
    }


# ── General ensemble analysis ─────────────────────────────────────────────────

def analyse_ensemble(
    states: List[QubitState],
    probs: List[float],
    name: str = "Custom ensemble",
) -> Dict[str, Any]:
    """
    Compute density matrix and uncertainty metrics for an arbitrary ensemble.
    Useful for interactive exploration in the Streamlit page.
    """
    dm = DensityMatrix.from_ensemble(states, probs)
    z0, z1 = dm.measure_prob("z")
    x0, x1 = dm.measure_prob("x")
    y0, y1 = dm.measure_prob("y")
    return {
        "name":        name,
        "density_matrix": dm,
        "purity":      dm.purity(),
        "bloch_vector": dm.bloch_vector(),
        "bloch_radius": dm.bloch_radius(),
        "coherence":   dm.coherence(),
        "z_basis":     {"P(0)": z0, "P(1)": z1},
        "x_basis":     {"P(+)": x0, "P(-)": x1},
        "y_basis":     {"P(+i)": y0, "P(-i)": y1},
    }


# ── Distinguishability helper ─────────────────────────────────────────────────

def trace_distance(dm1: DensityMatrix, dm2: DensityMatrix) -> float:
    """
    Trace distance  D(ρ, σ) = ½ Tr|ρ − σ|

    Operational meaning: the maximum probability advantage in distinguishing
    the two states with a single measurement.
    D = 0 → identical states.  D = 1 → perfectly distinguishable.
    """
    diff   = dm1.matrix - dm2.matrix
    eigs   = np.linalg.eigvalsh(diff).real
    return float(0.5 * np.sum(np.abs(eigs)))


def fidelity(dm1: DensityMatrix, dm2: DensityMatrix) -> float:
    """
    Fidelity  F(ρ, σ) = (Tr √(√ρ σ √ρ))²

    For two pure states: F = |⟨ψ₁|ψ₂⟩|².
    F = 1 → identical.  F = 0 → orthogonal (perfectly distinguishable).
    """
    rho = dm1.matrix
    sig = dm2.matrix
    sqrt_rho = _matrix_sqrt(rho)
    M = sqrt_rho @ sig @ sqrt_rho
    sqrt_M_trace = float(np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(M).real, 0))))
    return float(np.clip(sqrt_M_trace ** 2, 0.0, 1.0))


def _matrix_sqrt(m: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(m)
    evals = np.maximum(evals.real, 0.0)
    return (evecs * np.sqrt(evals)) @ evecs.conj().T


# ── BB84 state pair analysis ──────────────────────────────────────────────────

def bb84_basis_analysis() -> List[Dict[str, Any]]:
    """
    For each BB84 basis, show what happens when the wrong basis is measured.

    Rectilinear basis (Z): {|0⟩, |1⟩}  measured in X-basis → maximally mixed
    Diagonal basis (X):    {|+⟩, |−⟩}  measured in Z-basis → maximally mixed

    This basis mismatch effect is the physical foundation of BB84 security.
    """
    results = []

    pairs = [
        ("Rectilinear", ["|0⟩", "|1⟩"], "correct_basis", "z", "wrong_basis", "x"),
        ("Diagonal",    ["|+⟩", "|−⟩"], "correct_basis", "x", "wrong_basis", "z"),
    ]

    for basis_name, state_labels, cb_key, cb_basis, wb_key, wb_basis in pairs:
        states_qs = [BB84_STATES[lbl] for lbl in state_labels]

        for qs, label in zip(states_qs, state_labels):
            dm = DensityMatrix.from_pure(qs)
            cb_p0, cb_p1 = dm.measure_prob(cb_basis)
            wb_p0, wb_p1 = dm.measure_prob(wb_basis)
            results.append({
                "basis":         basis_name,
                "state":         label,
                "density_matrix": dm,
                "correct_basis": cb_basis.upper(),
                "correct_probs": {"P(0)": cb_p0, "P(1)": cb_p1},
                "wrong_basis":   wb_basis.upper(),
                "wrong_probs":   {"P(0)": wb_p0, "P(1)": wb_p1},
                "wrong_is_random": np.isclose(wb_p0, 0.5, atol=0.01),
            })
    return results
