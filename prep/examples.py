"""
prep_examples.py
================
Solved worked examples for the QKD preparation phase.

Each example returns a structured dict:
  {
    "title":        str,
    "symbol":       str,     -- compact mathematical notation
    "state" / "density_matrix" / ...,  -- computed objects
    "steps":        List[Step],        -- ordered solution steps
    "key_takeaway": str,
  }

Each Step dict:
  {
    "step":        int,
    "heading":     str,
    "math":        str,   -- LaTeX-compatible or plain text formula
    "result":      str,   -- concrete numerical answer
    "explanation": str,   -- physical intuition
  }
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List

from prep.hilbert import QubitState
from prep.density  import DensityMatrix


# ── Helper ────────────────────────────────────────────────────────────────────

def _fmt(z: complex, tol: float = 1e-9) -> str:
    """Format complex number: show imaginary only if non-zero."""
    if abs(z.imag) < tol:
        return f"{z.real:.4f}"
    if abs(z.real) < tol:
        return f"{z.imag:.4f}i"
    return f"{z.real:.4f} + {z.imag:.4f}i"


def _mat_str(m: np.ndarray) -> str:
    """Pretty 2×2 matrix string."""
    return (
        f"[[{_fmt(m[0,0])}, {_fmt(m[0,1])}],\n"
        f" [{_fmt(m[1,0])}, {_fmt(m[1,1])}]]"
    )


# ── Example 1: Pure state |+⟩ ────────────────────────────────────────────────

def example_plus_state() -> Dict[str, Any]:
    """
    Worked example: the |+⟩ superposition state.

    Covers: state vector → density matrix → purity → Bloch vector →
            measurement probabilities in two bases.
    """
    state = QubitState.plus()
    dm    = DensityMatrix.from_pure(state)
    props = dm.verify_properties()
    z0, z1 = dm.measure_prob("z")
    x0, x1 = dm.measure_prob("x")
    rx, ry, rz = dm.bloch_vector()

    steps: List[Dict[str, Any]] = [
        {
            "step": 1,
            "heading": "Write the state vector",
            "math": r"|\psi\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle",
            "result": f"α = {_fmt(state.alpha)},   β = {_fmt(state.beta)}",
            "explanation": (
                "Equal amplitudes 1/√2 ≈ 0.7071 for both |0⟩ and |1⟩. "
                "The state lies on the equator of the Bloch sphere."
            ),
        },
        {
            "step": 2,
            "heading": "Compute the density matrix  ρ = |ψ⟩⟨ψ|",
            "math": r"\rho = |\psi\rangle\langle\psi| = \frac{1}{2}\begin{pmatrix}1 & 1\\1 & 1\end{pmatrix}",
            "result": _mat_str(dm.matrix),
            "explanation": (
                "The off-diagonal elements (ρ₀₁ = ρ₁₀ = 0.5) are the coherences — "
                "the quantum signature that tells us this is a genuine superposition, "
                "not a classical mixture."
            ),
        },
        {
            "step": 3,
            "heading": "Verify  Tr(ρ) = 1",
            "math": r"\text{Tr}(\rho) = \rho_{00} + \rho_{11}",
            "result": f"= {dm.matrix[0,0].real:.4f} + {dm.matrix[1,1].real:.4f} = {dm.trace():.4f}  ✓",
            "explanation": "The diagonal elements are the Z-basis probabilities and must sum to 1.",
        },
        {
            "step": 4,
            "heading": "Compute purity  Tr(ρ²)",
            "math": r"\text{Tr}(\rho^2) = \text{Tr}\!\left(\frac{1}{4}\begin{pmatrix}2&2\\2&2\end{pmatrix}\right)",
            "result": f"= {dm.purity():.4f}  → Pure state ✓",
            "explanation": (
                "Purity = 1.0 confirms this is a pure quantum state. "
                "Every pure state has Tr(ρ²) = 1 regardless of which state it is."
            ),
        },
        {
            "step": 5,
            "heading": "Find the Bloch vector  rₖ = Tr(ρ σₖ)",
            "math": (
                r"r_x = \text{Tr}(\rho\,\sigma_x),\quad"
                r"r_y = \text{Tr}(\rho\,\sigma_y),\quad"
                r"r_z = \text{Tr}(\rho\,\sigma_z)"
            ),
            "result": f"r = ({rx:.4f}, {ry:.4f}, {rz:.4f})   |r| = {dm.bloch_radius():.4f}",
            "explanation": (
                "|+⟩ points along the +X axis of the Bloch sphere. "
                "rz = 0 means equal Z-basis probabilities. rx = 1 means "
                "the qubit is perfectly aligned with the X-axis eigenstates."
            ),
        },
        {
            "step": 6,
            "heading": "Measurement probabilities (Born rule)",
            "math": r"P(k) = \langle k|\rho|k\rangle = \text{Tr}(\Pi_k\,\rho)",
            "result": (
                f"Z-basis:  P(0) = {z0:.4f},  P(1) = {z1:.4f}\n"
                f"X-basis:  P(+) = {x0:.4f},  P(−) = {x1:.4f}"
            ),
            "explanation": (
                "Z-basis gives 50 / 50 — maximum Z uncertainty. "
                "X-basis gives P(+) = 1.0 — zero X uncertainty. "
                "This anisotropy is the hallmark of quantum uncertainty."
            ),
        },
    ]

    return {
        "title":     "Worked Example 1 — Pure Superposition |+⟩",
        "symbol":    "|+⟩ = (|0⟩ + |1⟩)/√2",
        "state":     state,
        "density_matrix": dm,
        "steps":     steps,
        "key_takeaway": (
            "The |+⟩ state has MAXIMUM uncertainty in Z-basis and ZERO uncertainty "
            "in X-basis. This anisotropy is a pure quantum effect: coherence in the "
            "density matrix (non-zero off-diagonal elements) cannot be explained by "
            "classical probability."
        ),
    }


# ── Example 2: Mixed state — classical 50/50 mixture ─────────────────────────

def example_classical_mixture() -> Dict[str, Any]:
    """
    Worked example: classical 50/50 mixture → maximally mixed state ρ = I/2.
    """
    states_qs = [QubitState.ket0(), QubitState.ket1()]
    dm = DensityMatrix.from_ensemble(states_qs, [0.5, 0.5])
    mm = DensityMatrix.maximally_mixed()
    z0, z1 = dm.measure_prob("z")
    x0, x1 = dm.measure_prob("x")
    rx, ry, rz = dm.bloch_vector()

    steps: List[Dict[str, Any]] = [
        {
            "step": 1,
            "heading": "Define the ensemble",
            "math": r"\rho = p_0|0\rangle\langle 0| + p_1|1\rangle\langle 1|,\quad p_0=p_1=\tfrac{1}{2}",
            "result": "States: {|0⟩, |1⟩},   Probabilities: {0.5, 0.5}",
            "explanation": (
                "A coin was flipped. Heads → |0⟩, Tails → |1⟩. "
                "The qubit IS in one definite state — we simply don't know which."
            ),
        },
        {
            "step": 2,
            "heading": "Compute the weighted sum",
            "math": (
                r"\rho = \frac{1}{2}\begin{pmatrix}1&0\\0&0\end{pmatrix}"
                r" + \frac{1}{2}\begin{pmatrix}0&0\\0&1\end{pmatrix}"
                r" = \frac{1}{2}\begin{pmatrix}1&0\\0&1\end{pmatrix}"
            ),
            "result": _mat_str(dm.matrix),
            "explanation": (
                "The off-diagonal elements are zero. There is no quantum coherence. "
                "Classical mixing always destroys (or never creates) off-diagonal terms."
            ),
        },
        {
            "step": 3,
            "heading": "Identify: this equals I/2 (maximally mixed)",
            "math": r"\rho = \frac{I}{2} = \frac{1}{2}\begin{pmatrix}1&0\\0&1\end{pmatrix}",
            "result": f"Matches maximally mixed: {np.allclose(dm.matrix, mm.matrix)} ✓",
            "explanation": (
                "Any 50/50 classical mixture of two orthogonal states "
                "produces the same maximally mixed state I/2."
            ),
        },
        {
            "step": 4,
            "heading": "Compute purity  Tr(ρ²)",
            "math": r"\text{Tr}(\rho^2) = \text{Tr}\!\left(\frac{I}{4}\right) = \frac{1}{2}",
            "result": f"= {dm.purity():.4f}  → Mixed state (minimum purity for qubit)",
            "explanation": (
                "Purity = 0.5 is the minimum possible for a qubit. "
                "This state carries zero quantum information — maximum disorder."
            ),
        },
        {
            "step": 5,
            "heading": "Bloch vector",
            "math": r"\mathbf{r} = \bigl(\text{Tr}(\rho\,\sigma_x),\,\text{Tr}(\rho\,\sigma_y),\,\text{Tr}(\rho\,\sigma_z)\bigr)",
            "result": f"r = ({rx:.4f}, {ry:.4f}, {rz:.4f})   |r| = {dm.bloch_radius():.4f}",
            "explanation": (
                "The zero Bloch vector places this state at the centre of the sphere. "
                "Every basis measurement gives 50 / 50 — the state has no preferred direction."
            ),
        },
        {
            "step": 6,
            "heading": "Measurement probabilities",
            "math": r"P(k) = \text{Tr}(\Pi_k\,\rho) = \frac{1}{2}",
            "result": (
                f"Z-basis:  P(0) = {z0:.4f},  P(1) = {z1:.4f}\n"
                f"X-basis:  P(+) = {x0:.4f},  P(−) = {x1:.4f}"
            ),
            "explanation": (
                "EVERY basis gives 50 / 50. This is how it differs from |+⟩: "
                "the pure state gives P(+) = 1.0 in X-basis while the mixed state gives 0.5."
            ),
        },
    ]

    return {
        "title":     "Worked Example 2 — Classical 50/50 Mixture → I/2",
        "symbol":    "ρ = ½|0⟩⟨0| + ½|1⟩⟨1| = I/2",
        "density_matrix": dm,
        "steps":     steps,
        "key_takeaway": (
            "The maximally mixed state gives 50/50 in EVERY basis — it has no "
            "preferred direction in Hilbert space. Unlike the |+⟩ superposition, "
            "it cannot be identified by measuring in any single basis. "
            "Purity = 0.5 and zero coherence are the defining signatures."
        ),
    }


# ── Example 3: All four BB84 preparation states ───────────────────────────────

def example_bb84_states() -> Dict[str, Any]:
    """
    Worked example: all four BB84 qubit states Alice prepares.
    """
    configs = [
        (0, 0, "|0⟩", r"|0\rangle",        "Rectilinear (+)", "North pole, +Z"),
        (1, 0, "|1⟩", r"|1\rangle",        "Rectilinear (+)", "South pole, −Z"),
        (0, 1, "|+⟩", r"|+\rangle",        "Diagonal (×)",   "Equator, +X"),
        (1, 1, "|−⟩", r"|{-}\rangle",      "Diagonal (×)",   "Equator, −X"),
    ]

    states_info = []
    for bit, basis, label, latex_lbl, basis_name, bloch_pos in configs:
        qs   = QubitState.from_bb84(bit, basis)
        dm   = DensityMatrix.from_pure(qs)
        z0, z1 = dm.measure_prob("z")
        x0, x1 = dm.measure_prob("x")
        rx, ry, rz = dm.bloch_vector()
        states_info.append({
            "label":       label,
            "latex":       latex_lbl,
            "bit":         bit,
            "basis":       basis,
            "basis_name":  basis_name,
            "bloch_pos":   bloch_pos,
            "state":       qs,
            "density_matrix": dm,
            "purity":      dm.purity(),
            "bloch_vector": (rx, ry, rz),
            "bloch_radius": dm.bloch_radius(),
            "coherence":   dm.coherence(),
            "z_basis":     {"P(0)": z0, "P(1)": z1},
            "x_basis":     {"P(+)": x0, "P(-)": x1},
        })

    key_insights = [
        "All 4 states are PURE (purity = 1.0) — Alice always knows exactly what she prepares.",
        "|0⟩ and |1⟩ are orthogonal: ⟨0|1⟩ = 0. Perfectly distinguishable in Z-basis.",
        "|+⟩ and |−⟩ are orthogonal: ⟨+|−⟩ = 0. Perfectly distinguishable in X-basis.",
        "Measuring |0⟩ in X-basis gives 50/50 — no information about bit value leaks.",
        "The two bases are mutually unbiased: knowing the state in one gives no info in the other.",
        "This basis ambiguity is the quantum foundation of BB84 security.",
    ]

    return {
        "title":        "Worked Example 3 — BB84 Preparation States",
        "states":       states_info,
        "key_insights": key_insights,
        "key_takeaway": (
            "The four BB84 states form two pairs of mutually unbiased bases. "
            "Any attempt by Eve to measure in the wrong basis destroys the state's "
            "purity and introduces detectable errors into the sifted key."
        ),
    }


# ── Example 4: Density matrix properties verification ────────────────────────

def example_properties_verification() -> Dict[str, Any]:
    """
    Worked example: systematically verify all four density matrix properties
    for both a pure and a mixed state side by side.
    """
    dm_pure  = DensityMatrix.from_pure(QubitState.plus())
    dm_mixed = DensityMatrix.from_ensemble(
        [QubitState.ket0(), QubitState.ket1()], [0.7, 0.3]
    )

    props_pure  = dm_pure.verify_properties()
    props_mixed = dm_mixed.verify_properties()

    properties = [
        {
            "name":       "Hermitian  (ρ = ρ†)",
            "symbol":     r"\rho = \rho^\dagger",
            "meaning":    (
                "The density matrix equals its conjugate transpose. "
                "This is required so that all expectation values ⟨O⟩ = Tr(ρO) are real."
            ),
            "pure":  f"Passed: {props_pure['hermitian']['passed']}",
            "mixed": f"Passed: {props_mixed['hermitian']['passed']}",
        },
        {
            "name":       "Unit trace  (Tr ρ = 1)",
            "symbol":     r"\text{Tr}(\rho) = 1",
            "meaning":    (
                "The sum of all diagonal elements equals 1. "
                "This is the quantum normalisation condition — total probability = 1."
            ),
            "pure":  f"Tr = {props_pure['unit_trace']['value']:.6f}",
            "mixed": f"Tr = {props_mixed['unit_trace']['value']:.6f}",
        },
        {
            "name":       "Positive semi-definite  (PSD)",
            "symbol":     r"\langle\psi|\rho|\psi\rangle \geq 0\;\forall|\psi\rangle",
            "meaning":    (
                "All eigenvalues ≥ 0. Equivalently, every measurement outcome "
                "probability is non-negative. Negative eigenvalues would be unphysical."
            ),
            "pure":  f"Eigenvalues: {[round(e,4) for e in props_pure['positive_semidefinite']['eigenvalues']]}",
            "mixed": f"Eigenvalues: {[round(e,4) for e in props_mixed['positive_semidefinite']['eigenvalues']]}",
        },
        {
            "name":       "Purity  (Tr ρ²)",
            "symbol":     r"\frac{1}{2} \leq \text{Tr}(\rho^2) \leq 1",
            "meaning":    (
                "Equals 1 for pure states, strictly less than 1 for mixed states. "
                "The minimum value 1/d (d = 2 for qubit) is achieved by I/2."
            ),
            "pure":  f"Tr(ρ²) = {props_pure['purity']['value']:.6f}  (pure ✓)",
            "mixed": f"Tr(ρ²) = {props_mixed['purity']['value']:.6f}  (mixed)",
        },
    ]

    return {
        "title":      "Worked Example 4 — Density Matrix Properties",
        "properties": properties,
        "pure_dm":    dm_pure,
        "mixed_dm":   dm_mixed,
        "pure_label": "|+⟩ (pure state)",
        "mixed_label": "0.7|0⟩⟨0| + 0.3|1⟩⟨1| (mixed state)",
        "key_takeaway": (
            "Every valid density matrix satisfies all four properties simultaneously. "
            "The purity is the single most useful discriminant: "
            "pure state → Tr(ρ²) = 1; any mixing → Tr(ρ²) < 1."
        ),
    }


# ── Registry of all examples ──────────────────────────────────────────────────

EXAMPLES: List[Dict[str, Any]] = []

def get_all_examples() -> List[Dict[str, Any]]:
    """Return all solved examples in order."""
    return [
        example_plus_state(),
        example_classical_mixture(),
        example_bb84_states(),
        example_properties_verification(),
    ]
