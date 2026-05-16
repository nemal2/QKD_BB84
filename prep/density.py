"""
prep_density.py
===============
Density operator (density matrix) for QKD preparation phase.

Provides:
  DensityMatrix  — 2×2 complex Hermitian operator ρ with:
    • Pure state constructor  (ρ = |ψ⟩⟨ψ|)
    • Mixed state constructor (ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|)
    • Maximally mixed state  (ρ = I/2)
    • Bloch-vector constructor
    • Full property verification
    • Purity, eigenvalues, expectation values
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any

from prep.hilbert import QubitState, PAULI_X, PAULI_Y, PAULI_Z, IDENTITY


class DensityMatrix:
    """
    Qubit density operator ρ — a 2×2 positive semi-definite Hermitian matrix
    with unit trace.

    The four defining properties:
      1.  Hermitian:           ρ = ρ†
      2.  Positive semi-def:   ⟨ψ|ρ|ψ⟩ ≥ 0   for all |ψ⟩
      3.  Unit trace:          Tr(ρ) = 1
      4.  Purity bound:        1/2 ≤ Tr(ρ²) ≤ 1   (for a qubit)

    For a pure state:   ρ² = ρ,  Tr(ρ²) = 1
    For a mixed state:  ρ² ≠ ρ,  Tr(ρ²) < 1
    """

    def __init__(self, matrix: np.ndarray):
        self._rho = np.asarray(matrix, dtype=complex)
        if self._rho.shape != (2, 2):
            raise ValueError("Density matrix must be 2×2.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def matrix(self) -> np.ndarray:
        return self._rho.copy()

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_pure(cls, state: QubitState) -> DensityMatrix:
        """
        Pure state density matrix:  ρ = |ψ⟩⟨ψ|

        Properties: purity = 1, Bloch vector has |r| = 1,
        off-diagonal elements (coherences) are generally non-zero.
        """
        return cls(state.density_matrix())

    @classmethod
    def from_ensemble(
        cls,
        states: List[QubitState],
        probs: List[float],
    ) -> DensityMatrix:
        """
        Classical mixture (ensemble):  ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

        'We know the system is in one of these states but not which one.'
        This is classical ignorance — the uncertainty is in our knowledge,
        not in the physical state of the qubit.

        Parameters
        ----------
        states : list of QubitState
        probs  : probabilities (must sum to 1)
        """
        if len(states) != len(probs):
            raise ValueError("Number of states and probabilities must match.")
        if abs(sum(probs) - 1.0) > 1e-9:
            raise ValueError(f"Probabilities must sum to 1 (got {sum(probs):.6f}).")
        if any(p < 0 for p in probs):
            raise ValueError("All probabilities must be non-negative.")
        rho = np.zeros((2, 2), dtype=complex)
        for state, p in zip(states, probs):
            rho += p * state.density_matrix()
        return cls(rho)

    @classmethod
    def maximally_mixed(cls) -> DensityMatrix:
        """
        Maximally mixed state:  ρ = I/2

        Represents maximum uncertainty — we know nothing about the qubit.
        Purity = 0.5 (minimum for a qubit).
        Bloch vector = (0, 0, 0) — center of sphere.
        """
        return cls(IDENTITY / 2.0)

    @classmethod
    def from_bloch_vector(cls, rx: float, ry: float, rz: float) -> DensityMatrix:
        """
        Construct from Bloch vector components.

            ρ = (I + rx·X + ry·Y + rz·Z) / 2

        Valid when |r|² = rx² + ry² + rz² ≤ 1.
        Pure state ↔ |r| = 1.   Mixed state ↔ |r| < 1.
        """
        r_sq = rx ** 2 + ry ** 2 + rz ** 2
        if r_sq > 1.0 + 1e-9:
            raise ValueError(
                f"|r|² = {r_sq:.4f} > 1. Bloch vector must lie inside the unit sphere."
            )
        rho = (IDENTITY + rx * PAULI_X + ry * PAULI_Y + rz * PAULI_Z) / 2.0
        return cls(rho)

    # ── Core quantities ───────────────────────────────────────────────────────

    def trace(self) -> float:
        """Tr(ρ)  — should equal 1.0"""
        return float(np.trace(self._rho).real)

    def purity(self) -> float:
        """
        Tr(ρ²)

        Range for a qubit: [0.5, 1.0]
          1.0 → pure state (maximum knowledge)
          0.5 → maximally mixed (minimum knowledge)
        """
        return float(np.trace(self._rho @ self._rho).real)

    def is_pure(self, tol: float = 1e-9) -> bool:
        """True when purity ≈ 1 (within numerical tolerance)."""
        return abs(self.purity() - 1.0) < tol

    def eigenvalues(self) -> np.ndarray:
        """
        Real eigenvalues of ρ, sorted ascending.
        For a valid density matrix: all eigenvalues ∈ [0, 1] and sum to 1.
        """
        return np.sort(np.linalg.eigvalsh(self._rho).real)

    def bloch_vector(self) -> Tuple[float, float, float]:
        """
        Bloch vector  r = (rx, ry, rz)  where  rₖ = Tr(ρ σₖ).

        Interpretation:
          |r| = 1   →  pure state (surface of Bloch sphere)
          |r| < 1   →  mixed state (inside Bloch sphere)
          |r| = 0   →  maximally mixed (center of sphere)
        """
        rx = float(np.trace(self._rho @ PAULI_X).real)
        ry = float(np.trace(self._rho @ PAULI_Y).real)
        rz = float(np.trace(self._rho @ PAULI_Z).real)
        return rx, ry, rz

    def bloch_radius(self) -> float:
        """Length of Bloch vector |r|. Range: [0, 1]."""
        rx, ry, rz = self.bloch_vector()
        return float(np.sqrt(rx ** 2 + ry ** 2 + rz ** 2))

    def coherence(self) -> float:
        """
        Off-diagonal magnitude |ρ₀₁| — a measure of quantum coherence.
        Zero for classical mixtures in the computational basis.
        """
        return float(abs(self._rho[0, 1]))

    def measure_prob(self, basis: str = "z") -> Tuple[float, float]:
        """
        Measurement probabilities via Born rule:  P(k) = Tr(Πₖ ρ)

        basis : 'z' → {|0⟩, |1⟩},  'x' → {|+⟩, |−⟩},  'y' → {|+i⟩, |−i⟩}
        Returns (P(0), P(1))
        """
        if basis == "z":
            p0 = float(self._rho[0, 0].real)
        elif basis == "x":
            bra = np.array([1, 1], dtype=complex) / np.sqrt(2)
            p0 = float((bra.conj() @ self._rho @ bra).real)
        elif basis == "y":
            bra = np.array([1, 1j], dtype=complex) / np.sqrt(2)
            p0 = float((bra.conj() @ self._rho @ bra).real)
        else:
            raise ValueError(f"Unknown basis '{basis}'. Use 'z', 'x', or 'y'.")
        p0 = float(np.clip(p0, 0.0, 1.0))
        return p0, 1.0 - p0

    def expectation(self, operator: np.ndarray) -> complex:
        """⟨O⟩ = Tr(ρ O)"""
        return complex(np.trace(self._rho @ np.asarray(operator, dtype=complex)))

    def simulate_measurements(
        self, n_shots: int = 1000, basis: str = "z", seed: int | None = None
    ) -> np.ndarray:
        """
        Sample n_shots measurement outcomes (0 or 1) from Born-rule probabilities.
        Returns an integer array of 0s and 1s.
        """
        p0, p1 = self.measure_prob(basis)
        rng = np.random.default_rng(seed)
        return rng.choice([0, 1], size=n_shots, p=[p0, p1])

    # ── Property verification ─────────────────────────────────────────────────

    def verify_properties(self) -> Dict[str, Any]:
        """
        Check all four defining properties of a density matrix.
        Returns a dict with 'passed' flags, numeric values, and descriptions.
        """
        eigs = self.eigenvalues()
        rho  = self._rho
        return {
            "hermitian": {
                "passed":      bool(np.allclose(rho, rho.conj().T, atol=1e-9)),
                "description": "ρ = ρ†  (conjugate transpose equals itself)",
                "detail":      "Guarantees real eigenvalues and real expectation values.",
            },
            "unit_trace": {
                "passed":      bool(abs(self.trace() - 1.0) < 1e-9),
                "value":       self.trace(),
                "description": "Tr(ρ) = 1  (total probability normalisation)",
                "detail":      "Sum of all measurement outcome probabilities equals 1.",
            },
            "positive_semidefinite": {
                "passed":      bool(np.all(eigs >= -1e-9)),
                "eigenvalues": eigs.tolist(),
                "description": "All eigenvalues ≥ 0  (⟨ψ|ρ|ψ⟩ ≥ 0 for all |ψ⟩)",
                "detail":      "Ensures every probability is non-negative.",
            },
            "purity": {
                "value":       self.purity(),
                "is_pure":     self.is_pure(),
                "bloch_radius": self.bloch_radius(),
                "description": "Tr(ρ²) ∈ [0.5, 1.0] for a qubit",
                "detail":      "Equal to 1 for pure states; strictly < 1 for mixed states.",
            },
        }

    def summary(self) -> Dict[str, Any]:
        """Compact summary of all key quantities."""
        rx, ry, rz = self.bloch_vector()
        eigs = self.eigenvalues()
        z0, z1 = self.measure_prob("z")
        x0, x1 = self.measure_prob("x")
        return {
            "matrix":       self._rho,
            "purity":       self.purity(),
            "is_pure":      self.is_pure(),
            "trace":        self.trace(),
            "eigenvalues":  eigs,
            "bloch_vector": (rx, ry, rz),
            "bloch_radius": self.bloch_radius(),
            "coherence":    self.coherence(),
            "z_basis":      {"P(0)": z0, "P(1)": z1},
            "x_basis":      {"P(+)": x0, "P(-)": x1},
        }

    def __repr__(self) -> str:
        kind = "pure" if self.is_pure() else f"mixed, purity={self.purity():.3f}"
        return (
            f"DensityMatrix [{kind}]\n"
            f"{np.array2string(self._rho, precision=4, suppress_small=True)}"
        )
