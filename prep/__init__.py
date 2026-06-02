"""
prep
====
QKD BB84 preparation phase — educational module.

Submodules
──────────
  hilbert      — QubitState, Bloch sphere math, Pauli matrices, BB84 factory methods
  density      — DensityMatrix (pure / mixed / ensemble) with property verification
  uncertainty  — Classical ignorance vs quantum uncertainty, trace distance, fidelity
  examples     — Solved worked examples with step-by-step calculations
  viz          — Plotly visualisations (Bloch sphere, density heatmap, gauges)
  page         — Streamlit UI — call render_prep_page() to embed
"""

from prep.page import render_prep_page

__all__ = ["render_prep_page"]
