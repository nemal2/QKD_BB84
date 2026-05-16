"""
prep_viz.py
===========
Plotly visualisations for QKD preparation phase educational module.

All public functions return go.Figure objects ready for st.plotly_chart().

Functions
─────────
  plot_bloch_sphere()           — 3-D Bloch sphere with state vectors
  plot_density_matrix()         — dual heatmap (Re part + Im part)
  plot_measurement_probs()      — grouped bar chart across bases
  plot_purity_gauge()           — gauge chart for Tr(ρ²)
  plot_eigenvalue_bar()         — bar chart of density matrix eigenvalues
  plot_comparison_bloch()       — two states side by side on same sphere
  plot_wigner_qubit()           — Q-function (Husimi) on sphere surface
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple, Union

from prep.hilbert import QubitState
from prep.density  import DensityMatrix

# ── Colour palette (matches qkd_app.py) ──────────────────────────────────────
_C = {
    "|0⟩":  "#2563EB",   # blue
    "|1⟩":  "#DC2626",   # red
    "|+⟩":  "#16A34A",   # green
    "|−⟩":  "#EA580C",   # orange
    "|+i⟩": "#7C3AED",   # purple
    "|−i⟩": "#DB2777",   # pink
    "default": ["#2563EB","#DC2626","#16A34A","#EA580C","#7C3AED","#DB2777"],
}

_FONT_BODY = "Outfit, system-ui, sans-serif"
_FONT_MONO = "JetBrains Mono, monospace"

# ── Private helpers ───────────────────────────────────────────────────────────

def _sphere_xyz(n: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(n), np.cos(v))
    return x, y, z


def _add_sphere_frame(fig: go.Figure, row=None, col=None):
    """Add unit-sphere surface + equator/meridian grid lines to fig."""
    kw = dict(row=row, col=col) if row else {}

    sx, sy, sz = _sphere_xyz()
    fig.add_trace(go.Surface(
        x=sx, y=sy, z=sz,
        colorscale=[[0, "rgba(37,99,235,0.05)"], [1, "rgba(37,99,235,0.05)"]],
        showscale=False, hoverinfo="skip", opacity=0.65,
        lighting={"ambient": 1.0, "diffuse": 0},
        lightposition={"x": 0, "y": 0, "z": 1},
    ), **kw)

    ang = np.linspace(0, 2 * np.pi, 300)
    for xs, ys, zs in [
        (np.cos(ang), np.sin(ang), np.zeros_like(ang)),
        (np.cos(ang), np.zeros_like(ang), np.sin(ang)),
        (np.zeros_like(ang), np.cos(ang), np.sin(ang)),
    ]:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines", line={"color": "#E5E7EB", "width": 1},
            hoverinfo="skip", showlegend=False,
        ), **kw)

    # Axis stubs
    for xs, ys, zs in [
        ([0, 0], [0, 0], [-1.3, 1.3]),
        ([-1.3, 1.3], [0, 0], [0, 0]),
        ([0, 0], [-1.3, 1.3], [0, 0]),
    ]:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines", line={"color": "#D1D5DB", "width": 2},
            hoverinfo="skip", showlegend=False,
        ), **kw)

    # Pole and equator labels
    for x, y, z, txt in [
        (0, 0, 1.42, "|0⟩"), (0, 0, -1.42, "|1⟩"),
        (1.42, 0, 0, "|+⟩"), (-1.42, 0, 0, "|−⟩"),
        (0, 1.42, 0, "|+i⟩"), (0, -1.42, 0, "|−i⟩"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z], mode="text",
            text=[txt], textfont={"size": 10, "color": "#9CA3AF", "family": _FONT_BODY},
            hoverinfo="skip", showlegend=False,
        ), **kw)


def _state_colour(label: str, idx: int = 0) -> str:
    return _C.get(label, _C["default"][idx % len(_C["default"])])


# ── Public API ────────────────────────────────────────────────────────────────

def plot_bloch_sphere(
    states: List[Union[QubitState, DensityMatrix]],
    labels: List[str],
    title: str = "Bloch Sphere",
) -> go.Figure:
    """
    3-D Bloch sphere with one arrow per state/density-matrix.

    Pure states land on the surface (|r| = 1).
    Mixed states land inside (|r| < 1).
    The maximally mixed state sits at the origin.
    """
    fig = go.Figure()
    _add_sphere_frame(fig)

    for idx, (obj, label) in enumerate(zip(states, labels)):
        colour = _state_colour(label, idx)

        if isinstance(obj, QubitState):
            rx, ry, rz = obj.bloch_vector()
        else:
            rx, ry, rz = obj.bloch_vector()

        # Arrow shaft + tip marker
        fig.add_trace(go.Scatter3d(
            x=[0, rx], y=[0, ry], z=[0, rz],
            mode="lines+markers",
            line={"color": colour, "width": 7},
            marker={"size": [2, 9], "color": colour},
            name=label,
            hovertemplate=f"{label}<br>r = ({rx:.3f}, {ry:.3f}, {rz:.3f})<extra></extra>",
        ))
        # Text label near tip
        scale = 1.18
        fig.add_trace(go.Scatter3d(
            x=[rx * scale], y=[ry * scale], z=[rz * scale],
            mode="text", text=[label],
            textfont={"size": 12, "color": colour, "family": _FONT_MONO},
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        title={"text": title, "font": {"size": 16, "family": _FONT_BODY}},
        height=520,
        scene={
            "xaxis": {"showticklabels": False, "title": "X  (σₓ)"},
            "yaxis": {"showticklabels": False, "title": "Y  (σᵧ)"},
            "zaxis": {"showticklabels": False, "title": "Z  (σᵤ)"},
            "bgcolor": "#F9FAFB",
            "camera": {"eye": {"x": 1.5, "y": 1.1, "z": 0.8}},
        },
        paper_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        legend={"font": {"family": _FONT_BODY, "size": 12}},
    )
    return fig


def plot_density_matrix(
    dm: DensityMatrix,
    title: str = "Density Matrix  ρ",
) -> go.Figure:
    """
    Side-by-side heatmaps of Re(ρ) and Im(ρ) with annotated values.
    """
    rho   = dm.matrix
    real  = rho.real
    imag  = rho.imag
    ticks = ["⟨0|", "⟨1|"]
    rows  = ["|0⟩", "|1⟩"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Re(ρ) — Real Part", "Im(ρ) — Imaginary Part"],
        horizontal_spacing=0.14,
    )

    for col_idx, (data, subtitle) in enumerate([(real, "Re"), (imag, "Im")], start=1):
        cb_x = 0.44 if col_idx == 1 else 1.0
        fig.add_trace(go.Heatmap(
            z=data, x=ticks, y=rows,
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in data],
            texttemplate="%{text}",
            textfont={"size": 18, "family": _FONT_MONO},
            showscale=True,
            colorbar={"x": cb_x, "thickness": 14, "len": 0.85},
            hovertemplate=f"ρ[%{{y}}, %{{x}}] = %{{z:.4f}}<extra>{subtitle}(ρ)</extra>",
        ), row=1, col=col_idx)

    fig.update_layout(
        title={"text": title, "font": {"size": 15, "family": _FONT_BODY}},
        height=290,
        paper_bgcolor="white",
        plot_bgcolor="#F9FAFB",
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )
    fig.update_xaxes(tickfont={"size": 14, "family": _FONT_MONO})
    fig.update_yaxes(tickfont={"size": 14, "family": _FONT_MONO})
    return fig


def plot_measurement_probs(
    data: List[dict],
    title: str = "Measurement Probabilities",
) -> go.Figure:
    """
    Grouped bar chart: P(0)/P(1) for each state in Z-basis and X-basis.

    data items must have keys: 'label', 'z_basis' {P(0), P(1)},
                                             'x_basis' {P(+), P(-)}
    """
    labels = [d["label"] for d in data]
    z_p0   = [d["z_basis"]["P(0)"] for d in data]
    z_p1   = [d["z_basis"]["P(1)"] for d in data]
    x_p0   = [d["x_basis"].get("P(+)", d["x_basis"].get("P(0)", 0)) for d in data]
    x_p1   = [d["x_basis"].get("P(-)", d["x_basis"].get("P(1)", 0)) for d in data]

    colours = [_state_colour(l, i) for i, l in enumerate(labels)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Z-basis  {|0⟩, |1⟩}", "X-basis  {|+⟩, |−⟩}"],
        shared_yaxes=True,
    )

    for col_idx, (p0_list, p1_list, lbl0, lbl1) in enumerate([
        (z_p0, z_p1, "P(0)", "P(1)"),
        (x_p0, x_p1, "P(+)", "P(−)"),
    ], start=1):
        fig.add_trace(go.Bar(
            name=lbl0, x=labels, y=p0_list,
            marker_color="#2563EB", opacity=0.85,
            text=[f"{v:.2f}" for v in p0_list],
            textposition="outside",
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)
        fig.add_trace(go.Bar(
            name=lbl1, x=labels, y=p1_list,
            marker_color="#DC2626", opacity=0.85,
            text=[f"{v:.2f}" for v in p1_list],
            textposition="outside",
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

    fig.update_layout(
        barmode="group",
        title={"text": title, "font": {"size": 15, "family": _FONT_BODY}},
        height=370,
        yaxis={"title": "Probability", "range": [0, 1.25]},
        paper_bgcolor="white",
        plot_bgcolor="#F9FAFB",
        margin={"l": 40, "r": 20, "t": 65, "b": 40},
        legend={"font": {"family": _FONT_BODY, "size": 12}},
    )
    return fig


def plot_purity_gauge(
    dm: DensityMatrix,
    label: str = "",
) -> go.Figure:
    """Gauge chart for Tr(ρ²) ranging from 0.5 (maximally mixed) to 1 (pure)."""
    purity = dm.purity()
    colour = "#16A34A" if purity > 0.95 else ("#EA580C" if purity > 0.6 else "#DC2626")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=purity,
        delta={"reference": 0.5, "valueformat": ".4f",
               "prefix": "Δ from mixed: "},
        number={"valueformat": ".4f", "font": {"size": 30, "family": _FONT_MONO}},
        title={"text": f"Purity  Tr(ρ²)  {label}", "font": {"size": 13, "family": _FONT_BODY}},
        gauge={
            "axis": {"range": [0.5, 1.0], "tickwidth": 1, "nticks": 6},
            "bar": {"color": colour, "thickness": 0.28},
            "bgcolor": "#F9FAFB",
            "borderwidth": 0,
            "steps": [
                {"range": [0.5, 0.65], "color": "#FEE2E2"},
                {"range": [0.65, 0.85], "color": "#FEF3C7"},
                {"range": [0.85, 1.0],  "color": "#D1FAE5"},
            ],
            "threshold": {
                "line": {"color": "#111827", "width": 3},
                "thickness": 0.8,
                "value": purity,
            },
        },
    ))
    fig.update_layout(
        height=240,
        paper_bgcolor="white",
        margin={"l": 25, "r": 25, "t": 65, "b": 20},
    )
    return fig


def plot_eigenvalue_bar(
    dm: DensityMatrix,
    title: str = "Eigenvalues of ρ",
) -> go.Figure:
    """
    Bar chart of the two eigenvalues of the density matrix.

    For a pure state:   eigenvalues are {1, 0}.
    For maximally mixed: eigenvalues are {0.5, 0.5}.
    """
    eigs   = dm.eigenvalues()
    labels = ["λ₁  (smaller)", "λ₂  (larger)"]
    colours = ["#6B7280", "#2563EB"]

    fig = go.Figure(go.Bar(
        x=labels, y=eigs.tolist(),
        marker_color=colours,
        text=[f"{e:.4f}" for e in eigs],
        textposition="outside",
        textfont={"size": 14, "family": _FONT_MONO},
        hovertemplate="Eigenvalue: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "family": _FONT_BODY}},
        height=260,
        yaxis={"title": "Eigenvalue", "range": [0, 1.15]},
        paper_bgcolor="white",
        plot_bgcolor="#F9FAFB",
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
        shapes=[{"type": "line", "x0": -0.5, "x1": 1.5, "y0": 0.5, "y1": 0.5,
                 "line": {"dash": "dot", "color": "#9CA3AF", "width": 1.5}}],
        annotations=[{"x": 1.5, "y": 0.5, "xref": "x", "yref": "y",
                       "text": "Max-mixed", "showarrow": False,
                       "font": {"size": 10, "color": "#6B7280"}, "xanchor": "left"}],
    )
    return fig


def plot_bloch_sphere_comparison(
    pair: List[Tuple[Union[QubitState, DensityMatrix], str]],
    title: str = "Classical Ignorance  vs  Quantum Uncertainty",
) -> go.Figure:
    """
    Two sub-spheres side by side for comparing two states.
    pair = [(obj_a, label_a), (obj_b, label_b)]
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[pair[0][1], pair[1][1]],
    )

    for col_idx, (obj, label) in enumerate(pair, start=1):
        colour = _state_colour(label, col_idx - 1)

        if isinstance(obj, QubitState):
            rx, ry, rz = obj.bloch_vector()
        else:
            rx, ry, rz = obj.bloch_vector()
        radius = float(np.sqrt(rx**2 + ry**2 + rz**2))

        # Sphere surface
        sx, sy, sz = _sphere_xyz(40)
        fig.add_trace(go.Surface(
            x=sx, y=sy, z=sz,
            colorscale=[[0, "rgba(37,99,235,0.05)"], [1, "rgba(37,99,235,0.05)"]],
            showscale=False, hoverinfo="skip", opacity=0.6,
        ), row=1, col=col_idx)

        # Grid circles
        ang = np.linspace(0, 2 * np.pi, 200)
        for xs, ys, zs in [
            (np.cos(ang), np.sin(ang), np.zeros_like(ang)),
            (np.cos(ang), np.zeros_like(ang), np.sin(ang)),
        ]:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines", line={"color": "#E5E7EB", "width": 1},
                hoverinfo="skip", showlegend=False,
            ), row=1, col=col_idx)

        # State vector
        fig.add_trace(go.Scatter3d(
            x=[0, rx], y=[0, ry], z=[0, rz],
            mode="lines+markers",
            line={"color": colour, "width": 8},
            marker={"size": [2, 10], "color": colour},
            name=label,
            hovertemplate=f"r = ({rx:.3f},{ry:.3f},{rz:.3f})<br>|r| = {radius:.3f}<extra></extra>",
        ), row=1, col=col_idx)

        # Pole labels
        sc_kw = dict(row=1, col=col_idx)
        for x2, y2, z2, t in [(0,0,1.4,"|0⟩"),(0,0,-1.4,"|1⟩"),
                                (1.4,0,0,"|+⟩"),(-1.4,0,0,"|−⟩")]:
            fig.add_trace(go.Scatter3d(
                x=[x2], y=[y2], z=[z2], mode="text",
                text=[t], textfont={"size": 9, "color": "#9CA3AF"},
                hoverinfo="skip", showlegend=False,
            ), **sc_kw)

    scene_cfg = {
        "xaxis": {"showticklabels": False, "title": "X"},
        "yaxis": {"showticklabels": False, "title": "Y"},
        "zaxis": {"showticklabels": False, "title": "Z"},
        "bgcolor": "#F9FAFB",
        "camera": {"eye": {"x": 1.4, "y": 1.2, "z": 0.7}},
    }
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "family": _FONT_BODY}},
        height=450,
        scene=scene_cfg,
        scene2=scene_cfg,
        paper_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        showlegend=True,
        legend={"font": {"family": _FONT_BODY, "size": 12}},
    )
    return fig


def plot_measurement_histogram(
    outcomes: np.ndarray,
    basis: str = "z",
    title: str = "Simulated Measurement Outcomes",
) -> go.Figure:
    """
    Bar chart of 0/1 outcome counts from simulated shots.
    outcomes: integer array of 0s and 1s.
    """
    n_shots = len(outcomes)
    n0 = int(np.sum(outcomes == 0))
    n1 = n_shots - n0

    if basis == "z":
        lbl0, lbl1 = "|0⟩", "|1⟩"
    elif basis == "x":
        lbl0, lbl1 = "|+⟩", "|−⟩"
    else:
        lbl0, lbl1 = "|+i⟩", "|−i⟩"

    fig = go.Figure([
        go.Bar(
            x=[lbl0, lbl1], y=[n0, n1],
            marker_color=["#2563EB", "#DC2626"],
            text=[f"{n0} ({n0/n_shots:.1%})", f"{n1} ({n1/n_shots:.1%})"],
            textposition="outside",
            textfont={"size": 13, "family": _FONT_MONO},
        )
    ])
    fig.update_layout(
        title={"text": f"{title}  (N={n_shots} shots)", "font": {"size": 14, "family": _FONT_BODY}},
        height=300,
        yaxis={"title": "Count", "range": [0, n_shots * 1.2]},
        paper_bgcolor="white",
        plot_bgcolor="#F9FAFB",
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
    )
    return fig


def plot_purity_spectrum(
    dms: List[DensityMatrix],
    labels: List[str],
    title: str = "Purity Comparison",
) -> go.Figure:
    """
    Horizontal bar chart showing purity of multiple density matrices.
    """
    purities = [dm.purity() for dm in dms]
    colours  = ["#16A34A" if p > 0.95 else "#EA580C" if p > 0.6 else "#DC2626"
                for p in purities]

    fig = go.Figure(go.Bar(
        x=purities, y=labels, orientation="h",
        marker_color=colours,
        text=[f"{p:.4f}" for p in purities],
        textposition="outside",
        textfont={"size": 13, "family": _FONT_MONO},
        hovertemplate="Purity: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=1.0, line_dash="dot", line_color="#16A34A",
                  annotation_text="Pure (1.0)", annotation_position="top")
    fig.add_vline(x=0.5, line_dash="dot", line_color="#DC2626",
                  annotation_text="Max-mixed (0.5)", annotation_position="bottom")
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "family": _FONT_BODY}},
        height=max(180, 80 + 55 * len(dms)),
        xaxis={"title": "Tr(ρ²)", "range": [0.4, 1.15]},
        paper_bgcolor="white",
        plot_bgcolor="#F9FAFB",
        margin={"l": 120, "r": 60, "t": 55, "b": 40},
    )
    return fig
