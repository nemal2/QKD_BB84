"""
prep_hilbert.py
===============
Hilbert space fundamentals for QKD preparation phase.

Provides:
  QubitState  — single qubit |ψ⟩ = α|0⟩ + β|1⟩ with full math support
  PAULI_*     — Pauli matrices used throughout quantum mechanics
  Factory methods for all BB84 encoding states
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ── Pauli matrices and identity ───────────────────────────────────────────────
PAULI_X = np.array([[0, 1],  [1,  0]], dtype=complex)
PAULI_Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0],  [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


class QubitState:
    """
    Single qubit state  |ψ⟩ = α|0⟩ + β|1⟩

    Invariant: |α|² + |β|² = 1  (enforced on construction).

    The Bloch sphere parameterization is:
        α = cos(θ/2)
        β = e^(iφ) sin(θ/2)
    where θ ∈ [0, π] is the polar angle and φ ∈ [0, 2π) is the azimuthal angle.
    """

    def __init__(self, alpha: complex, beta: complex, *, _skip_norm: bool = False):
        self._alpha = complex(alpha)
        self._beta  = complex(beta)
        if not _skip_norm:
            norm = np.sqrt(abs(self._alpha) ** 2 + abs(self._beta) ** 2)
            if norm < 1e-12:
                raise ValueError("Cannot create a zero-norm qubit state.")
            self._alpha /= norm
            self._beta  /= norm

    # ── Basic accessors ───────────────────────────────────────────────────────

    @property
    def alpha(self) -> complex:
        """Amplitude for |0⟩"""
        return self._alpha

    @property
    def beta(self) -> complex:
        """Amplitude for |1⟩"""
        return self._beta

    @property
    def vector(self) -> np.ndarray:
        """State as a column vector [α, β]ᵀ"""
        return np.array([self._alpha, self._beta], dtype=complex)

    # ── Derived quantities ────────────────────────────────────────────────────

    def density_matrix(self) -> np.ndarray:
        """ρ = |ψ⟩⟨ψ|  (2×2 complex matrix)"""
        v = self.vector.reshape(2, 1)
        return v @ v.conj().T

    def bloch_vector(self) -> Tuple[float, float, float]:
        """
        Bloch sphere coordinates (rx, ry, rz).

        Computed as  rₖ = Tr(ρ σₖ)  for k ∈ {x, y, z}.
        For any pure state: |r| = 1 exactly.
        """
        rho = self.density_matrix()
        rx = float(np.trace(rho @ PAULI_X).real)
        ry = float(np.trace(rho @ PAULI_Y).real)
        rz = float(np.trace(rho @ PAULI_Z).real)
        return rx, ry, rz

    def bloch_angles(self) -> Tuple[float, float]:
        """
        Polar angle θ and azimuthal angle φ on the Bloch sphere.
        Returns (theta_rad, phi_rad).
        """
        rx, ry, rz = self.bloch_vector()
        theta = float(np.arccos(np.clip(rz, -1.0, 1.0)))
        phi   = float(np.arctan2(ry, rx)) % (2 * np.pi)
        return theta, phi

    def measure_prob(self, basis: str = "z") -> Tuple[float, float]:
        """
        Born-rule measurement probabilities in a given basis.

        Parameters
        ----------
        basis : 'z'  — computational basis  {|0⟩, |1⟩}
                'x'  — diagonal basis        {|+⟩, |−⟩}
                'y'  — circular basis        {|+i⟩, |−i⟩}

        Returns
        -------
        (P(outcome 0), P(outcome 1))
        """
        rho = self.density_matrix()
        if basis == "z":
            p0 = float(rho[0, 0].real)
        elif basis == "x":
            bra_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
            p0 = float((bra_plus.conj() @ rho @ bra_plus).real)
        elif basis == "y":
            bra_yi = np.array([1, 1j], dtype=complex) / np.sqrt(2)
            p0 = float((bra_yi.conj() @ rho @ bra_yi).real)
        else:
            raise ValueError(f"Unknown basis '{basis}'. Use 'z', 'x', or 'y'.")
        p0 = float(np.clip(p0, 0.0, 1.0))
        return p0, 1.0 - p0

    def inner_product(self, other: QubitState) -> complex:
        """⟨self|other⟩"""
        return complex(np.dot(self.vector.conj(), other.vector))

    def simulate_measurements(self, n_shots: int = 1000, basis: str = "z",
                              seed: int | None = None) -> np.ndarray:
        """
        Sample n_shots measurement outcomes (0 or 1) using Born-rule probabilities.
        Returns an array of 0s and 1s.
        """
        p0, p1 = self.measure_prob(basis)
        rng = np.random.default_rng(seed)
        return rng.choice([0, 1], size=n_shots, p=[p0, p1])

    def __repr__(self) -> str:
        a, b = self._alpha, self._beta
        return f"|ψ⟩ = ({a:.4f})|0⟩ + ({b:.4f})|1⟩"

    def latex(self) -> str:
        """LaTeX string for the state."""
        a, b = self._alpha, self._beta
        a_str = f"{a.real:.3f}" if abs(a.imag) < 1e-9 else f"{a:.3f}"
        b_str = f"{b.real:.3f}" if abs(b.imag) < 1e-9 else f"{b:.3f}"
        return rf"|\psi\rangle = ({a_str})|0\rangle + ({b_str})|1\rangle"

    # ── Factory constructors ──────────────────────────────────────────────────

    @classmethod
    def ket0(cls) -> QubitState:
        """Computational basis  |0⟩  — north pole of Bloch sphere"""
        return cls(1.0, 0.0, _skip_norm=True)

    @classmethod
    def ket1(cls) -> QubitState:
        """Computational basis  |1⟩  — south pole of Bloch sphere"""
        return cls(0.0, 1.0, _skip_norm=True)

    @classmethod
    def plus(cls) -> QubitState:
        """|+⟩ = (|0⟩ + |1⟩)/√2  — +X axis of Bloch sphere"""
        s = 1 / np.sqrt(2)
        return cls(s, s, _skip_norm=True)

    @classmethod
    def minus(cls) -> QubitState:
        """|−⟩ = (|0⟩ − |1⟩)/√2  — −X axis of Bloch sphere"""
        s = 1 / np.sqrt(2)
        return cls(s, -s, _skip_norm=True)

    @classmethod
    def plus_i(cls) -> QubitState:
        """|+i⟩ = (|0⟩ + i|1⟩)/√2  — +Y axis of Bloch sphere"""
        s = 1 / np.sqrt(2)
        return cls(s, 1j * s, _skip_norm=True)

    @classmethod
    def minus_i(cls) -> QubitState:
        """|−i⟩ = (|0⟩ − i|1⟩)/√2  — −Y axis of Bloch sphere"""
        s = 1 / np.sqrt(2)
        return cls(s, -1j * s, _skip_norm=True)

    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> QubitState:
        """
        Construct from Bloch sphere polar (θ) and azimuthal (φ) angles.

            |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩

        θ ∈ [0, π]   — 0 → |0⟩, π → |1⟩, π/2 → equatorial
        φ ∈ [0, 2π)  — 0 → |+⟩ direction, π/2 → |+i⟩ direction
        """
        alpha = complex(np.cos(theta / 2))
        beta  = complex(np.exp(1j * phi) * np.sin(theta / 2))
        return cls(alpha, beta, _skip_norm=True)

    @classmethod
    def from_bb84(cls, bit: int, basis: int) -> QubitState:
        """
        Alice's BB84 encoding.

        bit=0, basis=0 (Rectilinear) → |0⟩
        bit=1, basis=0 (Rectilinear) → |1⟩
        bit=0, basis=1 (Diagonal)    → |+⟩
        bit=1, basis=1 (Diagonal)    → |−⟩
        """
        if basis == 0:
            return cls.ket0() if bit == 0 else cls.ket1()
        else:
            return cls.plus() if bit == 0 else cls.minus()


# ── Named standard states for convenience ────────────────────────────────────

BB84_STATES: dict[str, QubitState] = {
    "|0⟩": QubitState.ket0(),
    "|1⟩": QubitState.ket1(),
    "|+⟩": QubitState.plus(),
    "|−⟩": QubitState.minus(),
}

CARDINAL_STATES: dict[str, QubitState] = {
    "|0⟩":  QubitState.ket0(),
    "|1⟩":  QubitState.ket1(),
    "|+⟩":  QubitState.plus(),
    "|−⟩":  QubitState.minus(),
    "|+i⟩": QubitState.plus_i(),
    "|−i⟩": QubitState.minus_i(),
}
