"""
qkd_app.py  ·  BB84 QKD Simulator
University of Ruhuna · Dept. of Computer Engineering
Run:  streamlit run qkd_app.py
"""

from __future__ import annotations

import json
import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bb84_config import SimulationConfig, SimulationResult
from bb84_fast import fast_run_simulation as _fast_sim
from bb84_runner import PRESET_SCENARIOS
from bb84_runner import run_simulation as _qiskit_sim


def _run(cfg: SimulationConfig) -> SimulationResult:
    if st.session_state.get("use_fast", True):
        return _fast_sim(cfg)
    return _qiskit_sim(cfg, verbose=False)


st.set_page_config(
    page_title="BB84 QKD Simulator",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu, footer, .stDeployButton, [data-testid="stSidebar"],
header[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 2.5rem 4rem !important; max-width: 1360px !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
html, body, .stApp { font-family: 'Outfit', system-ui, sans-serif !important; color: #111827; }

/* Metric */
[data-testid="stMetric"] {
    background: #F9FAFB; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 11px !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: .07em; color: #6B7280 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important; font-weight: 600 !important; color: #111827 !important;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] { font-size: 12px !important; color: #6B7280 !important; }

/* Tabs (Analysis/Research/Compare sub-tabs) */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #E5E7EB !important; gap: 0 !important; background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px !important; font-weight: 500 !important; color: #6B7280 !important;
    padding: 10px 20px !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] { color: #111827 !important; border-bottom-color: #2563EB !important; }

/* Primary button */
div[data-testid="stButton"] button[kind="primary"] {
    background: #2563EB !important; border: none !important; color: #fff !important;
    font-weight: 600 !important; font-size: 14px !important; border-radius: 7px !important;
    letter-spacing: .02em !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover { opacity: .88 !important; }

/* Secondary / default buttons */
div[data-testid="stButton"] button {
    border: 1px solid #E5E7EB !important; background: #fff !important;
    color: #374151 !important; font-size: 12px !important; border-radius: 6px !important;
    font-weight: 500 !important;
}
div[data-testid="stButton"] button:hover {
    border-color: #2563EB !important; color: #2563EB !important; background: #EFF6FF !important;
}

/* Download buttons */
div[data-testid="stDownloadButton"] button {
    background: #fff !important; border: 1px solid #E5E7EB !important;
    color: #374151 !important; font-size: 12px !important; border-radius: 6px !important;
}
div[data-testid="stDownloadButton"] button:hover { border-color: #2563EB !important; color: #2563EB !important; }

/* Bit cells */
.brow { display: flex; flex-wrap: wrap; gap: 3px; margin: 4px 0 12px; }
.bc {
    width: 22px; height: 22px; border-radius: 4px;
    display: inline-flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600;
}
.bc0 { background: #F0FDF4; color: #15803D; border: 1px solid #BBF7D0; }
.bc1 { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
.bce { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }

/* Nav button active indicator via primary */
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "page": "guide",
    "result": None,
    "sweep_result": None,
    "comparison_results": None,
    "use_fast": True,
    "last_runtime": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Plotly theme ──────────────────────────────────────────────────────────────
_PL = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#6B7280", family="Outfit, sans-serif", size=11),
    margin=dict(t=40, b=36, l=10, r=10),
    hoverlabel=dict(
        bgcolor="#1F2937", bordercolor="#374151", font_color="#F9FAFB", font_size=12
    ),
    xaxis=dict(gridcolor="#F3F4F6", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#F3F4F6", showline=False, zeroline=False),
)
C_BLUE, C_GREEN, C_AMBER, C_RED = "#2563EB", "#059669", "#D97706", "#DC2626"
C_TEAL, C_PURPLE = "#0891B2", "#7C3AED"


def _sec(r: SimulationResult):
    s = r.qber_result.security_status
    if "SECURE" in s:
        return "SECURE", C_GREEN, "#F0FDF4", "#BBF7D0"
    if "WARNING" in s:
        return "WARNING", C_AMBER, "#FFFBEB", "#FDE68A"
    return "ABORT", C_RED, "#FEF2F2", "#FECACA"


# ── Top navigation ────────────────────────────────────────────────────────────
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

_pages = [
    ("Guide", "guide"),
    ("Simulator", "sim"),
    ("Analysis", "analysis"),
    ("Research", "research"),
    ("Compare", "compare"),
]

nh, *_nav_cols, n_end = st.columns([2.8] + [1] * 5 + [1.4])

with nh:
    st.markdown(
        "<div style='font-family:Outfit,sans-serif;font-size:17px;font-weight:700;"
        "color:#111827;line-height:1.2;padding:6px 0;'>BB84 QKD Simulator<br>"
        "<span style='font-size:11px;font-weight:400;color:#9CA3AF;'>"
        "University of Ruhuna</span></div>",
        unsafe_allow_html=True,
    )

for col, (label, key) in zip(_nav_cols, _pages):
    with col:
        active = st.session_state["page"] == key
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state["page"] = key
            st.rerun()

with n_end:
    use_fast = st.toggle(
        "⚡ Fast mode",
        value=st.session_state.use_fast,
        help="numpy density matrix (~200× faster). Off = Qiskit AerSimulator.",
    )
    st.session_state["use_fast"] = use_fast

st.markdown(
    "<div style='border-top:1px solid #E5E7EB;margin:10px 0 28px;'></div>",
    unsafe_allow_html=True,
)

page = st.session_state["page"]
r: Optional[SimulationResult] = st.session_state.result


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: GUIDE
# ═════════════════════════════════════════════════════════════════════════════

if page == "guide":
    # Hero
    st.markdown(
        "<h1 style='font-family:Outfit,sans-serif;font-size:36px;font-weight:700;"
        "color:#111827;letter-spacing:-.02em;margin:0 0 10px;'>"
        "Quantum Key Distribution<br>"
        "<span style='color:#2563EB;'>BB84 Protocol Simulator</span></h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:16px;color:#6B7280;max-width:680px;line-height:1.7;"
        "margin:0 0 32px;'>"
        "A complete research and educational tool for simulating the BB84 quantum "
        "key distribution protocol. Explore how quantum cryptography works, "
        "test noise models, and detect eavesdropping attacks.</p>",
        unsafe_allow_html=True,
    )

    if st.button("Open Simulator  →", type="primary"):
        st.session_state["page"] = "sim"
        st.rerun()

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    # How the protocol works — 4 steps
    st.markdown(
        "<h2 style='font-size:20px;font-weight:700;color:#111827;margin-bottom:18px;'>"
        "How the BB84 protocol works</h2>",
        unsafe_allow_html=True,
    )

    step_cols = st.columns(4, gap="medium")
    steps_data = [
        (
            "01",
            "Alice prepares qubits",
            "Alice randomly picks a bit (0/1) and a basis — "
            "rectilinear (+) or diagonal (×). She encodes each bit as a qubit state: "
            "|0⟩, |1⟩, |+⟩, or |−⟩.",
        ),
        (
            "02",
            "Quantum channel",
            "Qubits travel over the quantum channel. "
            "Real channels introduce photon loss, depolarizing noise, "
            "T1/T2 relaxation, or fiber attenuation. "
            "An eavesdropper (Eve) may intercept.",
        ),
        (
            "03",
            "Bob measures & sifting",
            "Bob picks a random basis and measures. "
            "After transmission, Alice and Bob publicly compare bases. "
            "Only matching-basis qubits are kept — the sifted key (~50%).",
        ),
        (
            "04",
            "QBER check & key",
            "A sample of sifted bits is compared to estimate the "
            "Quantum Bit Error Rate (QBER). "
            "Low QBER → secure key. "
            "High QBER → eavesdropping detected, abort.",
        ),
    ]
    for col, (num, title, desc) in zip(step_cols, steps_data):
        with col:
            st.markdown(
                f"<div style='background:#F9FAFB;border:1px solid #E5E7EB;"
                f"border-top:3px solid #2563EB;border-radius:10px;padding:20px 18px 18px;height:100%;'>"
                f"<div style='font-size:10px;font-weight:700;letter-spacing:.12em;"
                f"text-transform:uppercase;color:#2563EB;margin-bottom:10px;'>Step {num}</div>"
                f"<div style='font-size:15px;font-weight:600;color:#111827;margin-bottom:10px;'>{title}</div>"
                f"<div style='font-size:13px;color:#6B7280;line-height:1.65;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    # Security thresholds
    st.markdown(
        "<h2 style='font-size:20px;font-weight:700;color:#111827;margin-bottom:18px;'>"
        "Security decision thresholds</h2>",
        unsafe_allow_html=True,
    )
    th1, th2, th3 = st.columns(3, gap="medium")
    for col, (color, border, bg, level, qber, desc) in zip(
        [th1, th2, th3],
        [
            (
                C_GREEN,
                "#BBF7D0",
                "#F0FDF4",
                "SECURE",
                "QBER < 5%",
                "Channel is clean. Key can be used for secure communication.",
            ),
            (
                C_AMBER,
                "#FDE68A",
                "#FFFBEB",
                "WARNING",
                "QBER 5–11%",
                "Elevated noise or partial interception. Proceed with caution.",
            ),
            (
                C_RED,
                "#FECACA",
                "#FEF2F2",
                "ABORT",
                "QBER ≥ 11%",
                "Channel is compromised. Eve detected. Discard key and restart.",
            ),
        ],
    ):
        with col:
            st.markdown(
                f"<div style='background:{bg};border:1px solid {border};"
                f"border-left:4px solid {color};border-radius:10px;padding:20px 18px;'>"
                f"<div style='font-size:10px;font-weight:700;letter-spacing:.12em;"
                f"text-transform:uppercase;color:{color};margin-bottom:6px;'>{level}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:18px;"
                f"font-weight:600;color:{color};margin-bottom:10px;'>{qber}</div>"
                f"<div style='font-size:13px;color:#374151;line-height:1.6;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    # Quick start guide
    st.markdown(
        "<h2 style='font-size:20px;font-weight:700;color:#111827;margin-bottom:18px;'>"
        "How to use this tool</h2>",
        unsafe_allow_html=True,
    )
    qs1, qs2, qs3 = st.columns(3, gap="medium")
    for col, (num, title, desc) in zip(
        [qs1, qs2, qs3],
        [
            (
                "1",
                "Configure & run",
                "Go to the **Simulator** tab. "
                "Choose a quick preset (e.g. Ideal, Eve 100%, Fiber 50km) "
                "or set parameters manually. Click **Run Simulation**.",
            ),
            (
                "2",
                "Read the dashboard",
                "The Dashboard shows QBER, sifted key size, "
                "final key length, and key agreement rate. "
                "The status banner tells you if the channel is secure.",
            ),
            (
                "3",
                "Explore & research",
                "Use **Analysis** for detailed QBER charts. "
                "Use **Research Sweeps** to plot QBER vs Eve probability "
                "or noise level. Use **Compare** for multi-scenario analysis.",
            ),
        ],
    ):
        with col:
            st.markdown(
                f"<div style='display:flex;gap:14px;align-items:flex-start;"
                f"padding:18px;background:#fff;border:1px solid #E5E7EB;border-radius:10px;'>"
                f"<div style='width:28px;height:28px;background:#EFF6FF;border-radius:50%;"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:13px;font-weight:700;color:#2563EB;flex-shrink:0;'>{num}</div>"
                f"<div><div style='font-size:14px;font-weight:600;color:#111827;"
                f"margin-bottom:6px;'>{title}</div>"
                f"<div style='font-size:13px;color:#6B7280;line-height:1.65;'>{desc}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<h2 style='font-size:20px;font-weight:700;color:#111827;margin-bottom:18px;'>"
        "Noise models available</h2>",
        unsafe_allow_html=True,
    )
    nm1, nm2, nm3, nm4, nm5 = st.columns(5, gap="small")
    for col, (name, desc) in zip(
        [nm1, nm2, nm3, nm4, nm5],
        [
            ("Depolarizing", "Random Pauli errors on each gate with probability p"),
            ("Amplitude Damp", "T1 energy relaxation (|1⟩→|0⟩ decay)"),
            ("Phase Damp", "T2 dephasing without energy loss"),
            ("Combined T1+T2", "Thermal relaxation combining T1 and T2"),
            ("Fiber Loss", "Photon absorption (0.2 dB/km attenuation)"),
        ],
    ):
        with col:
            st.markdown(
                f"<div style='padding:14px 16px;background:#F9FAFB;border:1px solid #E5E7EB;"
                f"border-radius:8px;text-align:center;'>"
                f"<div style='font-size:13px;font-weight:600;color:#111827;margin-bottom:6px;'>{name}</div>"
                f"<div style='font-size:11.5px;color:#6B7280;line-height:1.5;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;padding:20px;'>"
        "<span style='font-size:13px;color:#9CA3AF;'>"
        "Simulation engine: numpy vectorised density matrices  ·  "
        "~200× faster than Qiskit AerSimulator  ·  "
        "Identical physics via Kraus operators</span></div>",
        unsafe_allow_html=True,
    )
    _, btn_c, _ = st.columns([3, 2, 3])
    with btn_c:
        if st.button("Open Simulator  →", type="primary", use_container_width=True):
            st.session_state["page"] = "sim"
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════

elif page == "sim":
    st.markdown(
        "<h2 style='font-size:22px;font-weight:700;color:#111827;margin:0 0 6px;'>"
        "Simulator</h2>"
        "<p style='font-size:13px;color:#9CA3AF;margin:0 0 22px;'>"
        "Configure parameters below, then click Run Simulation.</p>",
        unsafe_allow_html=True,
    )

    # ── CONTROL PANEL ─────────────────────────────────────────────────────
    with st.container(border=True):
        # Quick presets row
        st.markdown(
            "<div style='font-size:11px;font-weight:600;letter-spacing:.08em;"
            "text-transform:uppercase;color:#9CA3AF;margin-bottom:10px;'>"
            "Quick Presets</div>",
            unsafe_allow_html=True,
        )
        preset_map = {
            "Ideal": SimulationConfig(n_qubits=600, label="Ideal"),
            "Eve 100%": SimulationConfig(
                n_qubits=600, eve_present=True, eve_intercept_prob=1.0, label="Eve 100%"
            ),
            "Eve 50%": SimulationConfig(
                n_qubits=600, eve_present=True, eve_intercept_prob=0.5, label="Eve 50%"
            ),
            "Depolar": SimulationConfig(
                n_qubits=600,
                noise_enabled=True,
                noise_model="depolarizing",
                depolar_prob=0.05,
                label="Depolarizing",
            ),
            "Amp. Damp": SimulationConfig(
                n_qubits=600,
                noise_enabled=True,
                noise_model="amplitude_damp",
                t1_ns=10_000,
                gate_time_ns=50,
                label="Amp.Damp",
            ),
            "Fiber 50km": SimulationConfig(
                n_qubits=800,
                noise_enabled=True,
                noise_model="fiber_loss",
                channel_length_km=50,
                label="Fiber 50km",
            ),
        }
        preset_clicked = None
        pc = st.columns(len(preset_map))
        for i, name in enumerate(preset_map):
            if pc[i].button(name, key=f"pre_{name}", use_container_width=True):
                preset_clicked = name

        st.divider()

        # Main parameters
        mp1, mp2, mp3, mp4 = st.columns([2, 1, 1, 1])
        with mp1:
            n_qubits = st.number_input(
                "Number of qubits", min_value=10, max_value=50_000,
                value=600, step=50, key="s_n",
                help="Type any value or use ▲/▼. Range: 10 – 50,000.",
            )
            n_qubits = int(n_qubits)
        with mp2:
            sample_pct = st.slider("QBER sample (%)", 5, 30, 10, 1, key="s_sp")
            sample_fraction = sample_pct / 100.0
        with mp3:
            s1, s2 = st.columns([1, 1])
            use_seed = s1.checkbox("Fixed seed", value=True, key="s_fseed")
            seed_val = s2.number_input(
                "Seed",
                value=42,
                step=1,
                key="s_sv",
                disabled=not use_seed,
                label_visibility="visible",
            )
            seed = int(seed_val) if use_seed else None
        with mp4:
            sim_label = st.text_input("Run label", value="Run 1", key="s_lbl")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Eve + Noise (side by side)
        ec, nc = st.columns(2, gap="large")

        # Eve
        with ec:
            st.markdown(
                "<div style='font-size:12px;font-weight:600;color:#374151;"
                "margin-bottom:8px;'>Eavesdropper (Eve)</div>",
                unsafe_allow_html=True,
            )
            eve_present = st.checkbox("Enable intercept-resend attack", key="s_eve")
            eve_intercept_prob = 1.0
            if eve_present:
                eve_intercept_prob = st.slider(
                    "Intercept probability", 0.0, 1.0, 1.0, 0.05, key="s_evep"
                )
                eq = eve_intercept_prob * 25
                if eq >= 11:
                    st.error(f"Expected QBER ≈ {eq:.1f}%  →  ABORT level")
                elif eq >= 5:
                    st.warning(f"Expected QBER ≈ {eq:.1f}%  →  WARNING level")
                else:
                    st.success(f"Expected QBER ≈ {eq:.1f}%  →  SECURE level")

        # Noise
        with nc:
            st.markdown(
                "<div style='font-size:12px;font-weight:600;color:#374151;"
                "margin-bottom:8px;'>Channel Noise</div>",
                unsafe_allow_html=True,
            )
            noise_enabled = st.checkbox("Enable noise model", key="s_noise")
            noise_models_selected: list = []
            noise_model = "depolarizing"
            depolar_prob = 0.01
            t1_ns = 100_000.0
            t2_ns = 50_000.0
            gate_time_ns = 50.0
            channel_length_km = 0.0

            _NM_LABELS = {
                "depolarizing": "Depolarizing (Pauli)",
                "amplitude_damp": "Amplitude Damping (T1)",
                "phase_damp": "Phase Damping (T2)",
                "combined": "Combined T1+T2",
                "fiber_loss": "Fiber Loss",
            }

            if noise_enabled:
                noise_models_selected = st.multiselect(
                    "Noise models — select multiple to combine",
                    list(_NM_LABELS.keys()),
                    default=["depolarizing"],
                    format_func=lambda x: _NM_LABELS[x],
                    key="s_nm_multi",
                )
                noise_model = noise_models_selected[0] if noise_models_selected else "depolarizing"

                if "depolarizing" in noise_models_selected:
                    depolar_prob = st.slider(
                        "Gate error prob (depolarizing)",
                        0.001, 0.20, 0.05, 0.001,
                        format="%.3f", key="s_dp",
                    )
                    st.caption(f"p/3 = {depolar_prob / 3:.5f} per Pauli")

                _needs_t1   = any(m in noise_models_selected for m in ["amplitude_damp", "combined"])
                _needs_t2   = any(m in noise_models_selected for m in ["phase_damp", "combined"])
                _needs_gate = any(m in noise_models_selected for m in ["amplitude_damp", "phase_damp", "combined"])

                if _needs_t1:
                    t1_us = st.slider("T1 relaxation (µs)", 1.0, 500.0, 10.0, 1.0, key="s_t1m")
                    t1_ns = t1_us * 1000
                if _needs_t2:
                    t2_us = st.slider("T2 dephasing (µs)", 0.5, 200.0, 5.0, 0.5, key="s_t2m")
                    t2_ns = min(t2_us * 1000, 2.0 * t1_ns - 1.0)
                    if t2_us * 1000 > 2 * t1_ns:
                        st.warning("T2 clamped to 2·T1")
                if _needs_gate:
                    gate_time_ns = st.slider("Gate time (ns)", 10, 200, 50, 5, key="s_gtm")
                    if _needs_t1:
                        st.caption(f"γ(T1) = {1.0 - math.exp(-gate_time_ns / t1_ns):.6f}")
                    if _needs_t2:
                        st.caption(f"λ(T2) = {1.0 - math.exp(-gate_time_ns / t2_ns):.6f}")

                if "fiber_loss" in noise_models_selected:
                    channel_length_km = st.slider(
                        "Channel length (km)", 0.0, 200.0, 50.0, 5.0, key="s_kmm"
                    )
                    if channel_length_km > 0:
                        survive = 10 ** (-0.2 * channel_length_km / 10)
                        st.caption(
                            f"P(survive) = {survive:.4f}  ·  "
                            f"Loss = {(1 - survive) * 100:.1f}%"
                        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

        # Run row
        _, run_col, rt_col = st.columns([3, 1, 1])
        with run_col:
            run_clicked = st.button(
                "Run Simulation", type="primary", use_container_width=True
            )
        with rt_col:
            if st.session_state.last_runtime is not None:
                bk = "numpy" if use_fast else "Qiskit"
                st.markdown(
                    f"<div style='text-align:right;padding-top:6px;'>"
                    f"<span style='font-size:12px;color:#6B7280;'>"
                    f"Last: <span style='font-family:JetBrains Mono,monospace;'>"
                    f"{st.session_state.last_runtime:.3f}s [{bk}]</span></span></div>",
                    unsafe_allow_html=True,
                )

    # ── Run logic ─────────────────────────────────────────────────────────
    if run_clicked or preset_clicked is not None:
        _active_noise_models = noise_models_selected  # from UI multiselect
        if preset_clicked is not None:
            pc_cfg = preset_map[preset_clicked]
            n_qubits = pc_cfg.n_qubits
            eve_present = pc_cfg.eve_present
            eve_intercept_prob = pc_cfg.eve_intercept_prob
            noise_enabled = pc_cfg.noise_enabled
            noise_model = pc_cfg.noise_model
            depolar_prob = pc_cfg.depolar_prob
            t1_ns = pc_cfg.t1_ns
            t2_ns = pc_cfg.t2_ns
            gate_time_ns = pc_cfg.gate_time_ns
            channel_length_km = pc_cfg.channel_length_km
            sim_label = pc_cfg.label
            _active_noise_models = []  # presets use single noise_model

        cfg = SimulationConfig(
            n_qubits=n_qubits,
            eve_present=eve_present,
            eve_intercept_prob=eve_intercept_prob,
            noise_enabled=noise_enabled,
            noise_model=noise_model,
            noise_models=_active_noise_models,
            depolar_prob=depolar_prob,
            t1_ns=t1_ns,
            t2_ns=t2_ns,
            gate_time_ns=gate_time_ns,
            channel_length_km=channel_length_km,
            sample_fraction=sample_fraction,
            seed=seed,
            label=sim_label,
        )
        try:
            with st.status("Running simulation…", expanded=True) as _s:
                st.write(f"Transmitting {cfg.n_qubits:,} qubits…")
                t0 = time.time()
                result = _run(cfg)
                elapsed = time.time() - t0
                st.write(
                    f"Sifted: {result.n_sifted:,} bits  ({result.sifted_key_rate:.1%})"
                )
                st.write(
                    f"QBER: {result.qber_result.qber * 100:.2f}%  —  "
                    f"{result.qber_result.security_status.strip()}"
                )
                _s.update(
                    label=f"Complete  ·  {elapsed:.3f} s",
                    state="complete",
                    expanded=False,
                )
            st.session_state.result = result
            st.session_state.last_runtime = elapsed
            r = result
        except Exception as e:
            st.error(f"Simulation error: {e}")

    # ── Dashboard results ──────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if r is None:
        st.markdown(
            "<div style='text-align:center;padding:60px 0;"
            "border:1px dashed #E5E7EB;border-radius:12px;margin-top:8px;'>"
            "<div style='font-size:16px;font-weight:600;color:#374151;margin-bottom:8px;'>"
            "No results yet</div>"
            "<div style='font-size:13px;color:#9CA3AF;'>"
            "Choose a preset or configure parameters above, "
            "then click <strong>Run Simulation</strong>.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        status_label, status_color, status_bg, status_border = _sec(r)
        qr = r.qber_result

        # Status banner
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:16px;"
            f"background:{status_bg};border:1px solid {status_border};"
            f"border-left:4px solid {status_color};"
            f"border-radius:10px;padding:14px 22px;margin-bottom:22px;'>"
            f"<span style='font-size:10px;font-weight:700;letter-spacing:.1em;"
            f"text-transform:uppercase;padding:3px 10px;border-radius:4px;"
            f"background:rgba(255,255,255,.6);color:{status_color};'>{status_label}</span>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:22px;"
            f"font-weight:600;color:{status_color};'>{qr.qber * 100:.2f}%</span>"
            f"<span style='font-size:13px;color:#6B7280;'>QBER</span>"
            f"<span style='width:1px;height:20px;background:rgba(0,0,0,.1);'></span>"
            f"<span style='font-size:13px;color:#374151;'>"
            f"95% CI  [{qr.confidence_low * 100:.1f}%, {qr.confidence_high * 100:.1f}%]</span>"
            f"<span style='width:1px;height:20px;background:rgba(0,0,0,.1);'></span>"
            f"<span style='font-size:13px;color:#6B7280;'>"
            f"{qr.errors} errors / {qr.sample_size} sampled bits</span>"
            f"<span style='margin-left:auto;font-size:12px;color:#9CA3AF;'>{r.config.label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # KPI row 1
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transmitted", f"{r.n_transmitted:,}", "qubits sent")
        k2.metric("Sifted key", f"{r.n_sifted:,}", f"{r.sifted_key_rate:.1%} retained")
        k3.metric(
            "Final key", f"{r.key_length:,}", f"{r.key_generation_rate:.4f} bits/qubit"
        )
        k4.metric(
            "QBER",
            f"{qr.qber * 100:.2f}%",
            f"{qr.errors}/{qr.sample_size} errors",
            delta_color="inverse",
        )
        st.write("")
        k5, k6, k7, k8 = st.columns(4)
        k5.metric(
            "Key agreement",
            f"{r.key_agreement_rate * 100:.2f}%",
            "pre-error-correction",
        )
        k6.metric(
            "QBER sample",
            f"{qr.sample_size:,}",
            f"{r.config.sample_fraction:.0%} of sifted",
        )
        k7.metric(
            "Eve intercept",
            f"{r.eve_interception_rate * 100:.1f}%"
            if r.eve_interception_rate > 0
            else "Not detected",
        )
        k8.metric(
            "Runtime",
            f"{st.session_state.last_runtime or r.runtime_seconds:.3f} s",
            "numpy" if use_fast else "Qiskit",
        )

        st.divider()

        col_l, col_r = st.columns([3, 2], gap="large")

        with col_l:
            st.markdown("**Key bit sequence — first 80 bits**")
            n_show = min(80, r.key_length)
            a_bits, b_bits = r.alice_final_key[:n_show], r.bob_final_key[:n_show]
            mismatches = sum(a != b for a, b in zip(a_bits, b_bits))

            def _cells(a_list, b_list, party):
                out = ""
                for a, b in zip(a_list, b_list):
                    v = a if party == "alice" else b
                    cls = (
                        "bc0"
                        if (a == b and v == 0)
                        else "bc1"
                        if (a == b and v == 1)
                        else "bce"
                    )
                    out += f'<span class="bc {cls}">{v}</span>'
                return out

            st.markdown(
                "<small style='color:#9CA3AF;font-weight:600;'>Alice</small>"
                f"<div class='brow'>{_cells(a_bits, b_bits, 'alice')}</div>"
                "<small style='color:#9CA3AF;font-weight:600;'>Bob</small>"
                f"<div class='brow'>{_cells(a_bits, b_bits, 'bob')}</div>"
                "<small style='color:#9CA3AF;'>"
                "🟢 0 agree &nbsp; 🔵 1 agree &nbsp; 🔴 mismatch"
                f"  ·  <strong>{mismatches}/{n_show}</strong> differ</small>",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown("**Key bit pipeline**")
            fig_f = go.Figure(
                go.Funnel(
                    y=["Transmitted", "Sifted", "After QBER sample", "Final key"],
                    x=[
                        r.n_transmitted,
                        r.n_sifted,
                        r.n_sifted - qr.sample_size,
                        r.key_length,
                    ],
                    textinfo="value+percent initial",
                    marker=dict(color=[C_BLUE, C_TEAL, C_AMBER, C_GREEN]),
                    textfont=dict(size=11, color="#fff"),
                    connector=dict(line=dict(color="#E5E7EB", width=1)),
                )
            )
            fig_f.update_layout(
                **{**_PL, "height": 200, "margin": dict(t=4, b=4, l=4, r=4)}
            )
            st.plotly_chart(fig_f, use_container_width=True)

        with col_r:
            st.markdown("**Run configuration**")
            cfg_r = r.config
            _nm_desc = {
                "depolarizing": f"Depolar p={cfg_r.depolar_prob:.3f}",
                "amplitude_damp": f"AmpDamp T1={cfg_r.t1_ns / 1000:.0f}µs",
                "phase_damp": f"PhaseDamp T2={cfg_r.t2_ns / 1000:.0f}µs",
                "combined": "T1+T2",
                "fiber_loss": f"Fiber {cfg_r.channel_length_km:.0f}km",
            }
            ch = "Ideal"
            if cfg_r.noise_enabled:
                _mlist = cfg_r.noise_models if cfg_r.noise_models else [cfg_r.noise_model]
                ch = " + ".join(_nm_desc.get(m, m) for m in _mlist)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Parameter": [
                            "Label",
                            "Qubits",
                            "Attack",
                            "Channel",
                            "QBER sample",
                            "Seed",
                        ],
                        "Value": [
                            cfg_r.label,
                            f"{cfg_r.n_qubits:,}",
                            "None"
                            if not cfg_r.eve_present
                            else f"Eve {cfg_r.eve_intercept_prob:.0%}",
                            ch,
                            f"{cfg_r.sample_fraction:.0%}",
                            str(cfg_r.seed),
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.write("")
            st.markdown("**Export**")
            st.download_button(
                "Alice key (.txt)",
                "\n".join(str(b) for b in r.alice_final_key),
                "alice_key.txt",
                "text/plain",
                use_container_width=True,
            )
            st.download_button(
                "Bob key (.txt)",
                "\n".join(str(b) for b in r.bob_final_key),
                "bob_key.txt",
                "text/plain",
                use_container_width=True,
            )
            st.download_button(
                "Results (.json)",
                json.dumps(
                    {
                        "label": r.config.label,
                        "qber": r.qber_result.qber,
                        "n_transmitted": r.n_transmitted,
                        "n_sifted": r.n_sifted,
                        "key_length": r.key_length,
                        "key_agreement_rate": r.key_agreement_rate,
                        "eve_interception_rate": r.eve_interception_rate,
                        "runtime_seconds": r.runtime_seconds,
                    },
                    indent=2,
                ),
                "qkd_results.json",
                "application/json",
                use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "analysis":
    st.markdown(
        "<h2 style='font-size:22px;font-weight:700;color:#111827;margin:0 0 6px;'>"
        "Analysis</h2>",
        unsafe_allow_html=True,
    )

    if r is None:
        st.info("No results yet — go to **Simulator** and run a simulation first.")
        if st.button("Go to Simulator"):
            st.session_state["page"] = "sim"
            st.rerun()
    else:
        qr = r.qber_result
        status_label, status_color, status_bg, status_border = _sec(r)

        ag, am = st.columns([1, 2], gap="large")
        with ag:
            st.markdown("**QBER gauge**")
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=qr.qber * 100,
                    number={
                        "suffix": "%",
                        "font": {
                            "size": 30,
                            "color": status_color,
                            "family": "JetBrains Mono, monospace",
                        },
                    },
                    gauge={
                        "axis": {
                            "range": [0, 35],
                            "tickfont": {"size": 9, "color": "#9CA3AF"},
                            "tickwidth": 1,
                        },
                        "bar": {"color": status_color, "thickness": 0.22},
                        "bgcolor": "#F9FAFB",
                        "borderwidth": 1,
                        "bordercolor": "#E5E7EB",
                        "steps": [
                            {"range": [0, 5], "color": "#F0FDF4"},
                            {"range": [5, 11], "color": "#FFFBEB"},
                            {"range": [11, 35], "color": "#FEF2F2"},
                        ],
                        "threshold": {
                            "line": {"color": C_RED, "width": 2},
                            "thickness": 0.7,
                            "value": 11,
                        },
                    },
                    title={
                        "text": f"<b>{status_label}</b>",
                        "font": {"color": status_color, "size": 12},
                    },
                )
            )
            fig_g.update_layout(
                **{**_PL, "height": 250, "margin": dict(t=30, b=10, l=30, r=30)}
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.markdown(
                f"95% CI: **{qr.confidence_low * 100:.2f}% – {qr.confidence_high * 100:.2f}%**  \n"
                f"Errors: **{qr.errors}** / {qr.sample_size} sampled bits  \n"
                f"Sample fraction: **{r.config.sample_fraction:.0%}** of sifted key"
            )

        with am:
            st.markdown("**QBER with 95% confidence interval  +  key bit composition**")
            fig_m = make_subplots(
                1,
                2,
                subplot_titles=["QBER (%)", "Key bit ratio"],
                specs=[[{"type": "xy"}, {"type": "domain"}]],
            )
            fig_m.add_trace(
                go.Bar(
                    x=["QBER"],
                    y=[qr.qber * 100],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[(qr.confidence_high - qr.qber) * 100],
                        arrayminus=[(qr.qber - qr.confidence_low) * 100],
                        color="#9CA3AF",
                        thickness=1.5,
                        width=8,
                    ),
                    marker_color=status_color,
                    showlegend=False,
                ),
                1,
                1,
            )
            fig_m.add_hline(
                y=11,
                line_dash="dot",
                line_color=C_RED,
                annotation_text="Abort 11%",
                annotation_font_size=9,
                row=1,
                col=1,
            )
            fig_m.add_hline(
                y=5,
                line_dash="dot",
                line_color=C_AMBER,
                annotation_text="Warning 5%",
                annotation_font_size=9,
                row=1,
                col=1,
            )
            n0, n1_ = r.alice_final_key.count(0), r.alice_final_key.count(1)
            fig_m.add_trace(
                go.Pie(
                    labels=["Zeros", "Ones"],
                    values=[n0, n1_],
                    marker_colors=[C_GREEN, C_BLUE],
                    hole=0.52,
                    textfont_size=10,
                ),
                1,
                2,
            )
            fig_m.update_layout(
                **{
                    **_PL,
                    "height": 280,
                    "margin": dict(t=40, b=10, l=10, r=10),
                    "legend": dict(font_size=10),
                }
            )
            fig_m.update_yaxes(range=[0, max(35, qr.qber * 100 + 8)], row=1, col=1)
            st.plotly_chart(fig_m, use_container_width=True)

        st.divider()
        ca, cb = st.columns(2, gap="large")
        with ca:
            st.markdown("**Basis matching (sifting)**")
            fig_b = go.Figure(
                go.Pie(
                    labels=["Matched (kept)", "Discarded"],
                    values=[r.n_sifted, r.n_transmitted - r.n_sifted],
                    marker_colors=[C_GREEN, "#E5E7EB"],
                    hole=0.52,
                    textfont_size=10,
                )
            )
            fig_b.update_layout(
                **{
                    **_PL,
                    "height": 230,
                    "margin": dict(t=10, b=10, l=10, r=10),
                    "legend": dict(font_size=10),
                }
            )
            st.plotly_chart(fig_b, use_container_width=True)
        with cb:
            st.markdown("**Error distribution in QBER sample**")
            fig_e = go.Figure(
                go.Bar(
                    x=["Correct", "Errors"],
                    y=[qr.sample_size - qr.errors, qr.errors],
                    marker_color=[C_GREEN, C_RED],
                    text=[qr.sample_size - qr.errors, qr.errors],
                    textposition="outside",
                    textfont_size=11,
                )
            )
            fig_e.update_layout(
                **{**_PL, "height": 230, "margin": dict(t=10, b=10, l=10, r=10)}
            )
            st.plotly_chart(fig_e, use_container_width=True)

        st.divider()

        # ── Key quality metrics ───────────────────────────────────────
        st.markdown("**Key quality & quantum measurement**")
        key = r.alice_final_key
        n_key = len(key)
        p1 = sum(key) / n_key if n_key else 0.5
        p0 = 1 - p1
        entropy = (
            -(p0 * math.log2(p0) if p0 > 0 else 0)
            - (p1 * math.log2(p1) if p1 > 0 else 0)
        )
        # Lag-1 autocorrelation
        arr = np.array(key, dtype=float)
        ac1 = 0.0
        if n_key > 2 and arr.var() > 0:
            m = arr.mean()
            ac1 = float(np.mean((arr[:-1] - m) * (arr[1:] - m)) / arr.var())

        qm1, qm2, qm3, qm4 = st.columns(4)
        qm1.metric(
            "Shannon entropy",
            f"{entropy:.4f} bits/bit",
            f"{'ideal' if abs(entropy - 1.0) < 0.05 else f'{(1 - entropy) * 100:.1f}% below max'}",
        )
        qm2.metric(
            "Bit balance  (0s / 1s)",
            f"{p0 * 100:.1f}% / {p1 * 100:.1f}%",
            "ideal ≈ 50/50",
        )
        qm3.metric(
            "Lag-1 autocorrelation",
            f"{ac1:.4f}",
            "ideal = 0  (independent bits)",
        )
        qm4.metric(
            "Shots per qubit",
            "1  (single-shot)",
            "No-cloning theorem",
        )

        st.write("")

        # ── Entropy + autocorrelation charts ─────────────────────────
        ea1, ea2 = st.columns(2, gap="large")

        with ea1:
            st.markdown("**Shannon entropy vs ideal**")
            fig_ent = go.Figure()
            fig_ent.add_trace(go.Bar(
                x=["This key", "Ideal (random)", "Worst case (constant)"],
                y=[entropy, 1.0, 0.0],
                marker_color=[status_color, C_GREEN, C_RED],
                text=[f"{entropy:.4f}", "1.0000", "0.0000"],
                textposition="outside",
                textfont_size=11,
            ))
            fig_ent.update_layout(
                **{**_PL, "height": 240,
                   "margin": dict(t=10, b=10, l=10, r=10),
                   "yaxis": {**_PL.get("yaxis", {}), "range": [0, 1.15]}},
            )
            st.plotly_chart(fig_ent, use_container_width=True)
            st.caption(
                "H = 1.0 bit/bit → perfect randomness. "
                "Low entropy means bits are predictable."
            )

        with ea2:
            st.markdown("**Autocorrelation (lags 1–10)**")
            max_lag = min(10, n_key - 2)
            lags, acs = [], []
            for lag in range(1, max_lag + 1):
                if arr.var() > 0:
                    c = float(np.mean((arr[:-lag] - arr.mean()) * (arr[lag:] - arr.mean())) / arr.var())
                else:
                    c = 0.0
                lags.append(lag)
                acs.append(c)
            fig_ac = go.Figure()
            fig_ac.add_trace(go.Bar(
                x=lags, y=acs,
                marker_color=[C_RED if abs(v) > 0.05 else C_GREEN for v in acs],
                showlegend=False,
            ))
            fig_ac.add_hline(y=0.05,  line_dash="dot", line_color=C_AMBER,
                             annotation_text="+0.05", annotation_font_size=8)
            fig_ac.add_hline(y=-0.05, line_dash="dot", line_color=C_AMBER)
            fig_ac.update_layout(
                **{**_PL, "height": 240,
                   "margin": dict(t=10, b=10, l=10, r=10),
                   "xaxis": {**_PL.get("xaxis", {}), "title": "Lag",
                              "tickmode": "linear", "dtick": 1},
                   "yaxis": {**_PL.get("yaxis", {}), "title": "Autocorr."}},
            )
            st.plotly_chart(fig_ac, use_container_width=True)
            st.caption(
                "Values near 0 = bits are independent across positions. "
                "Green = safe (|r| < 0.05), Red = correlated."
            )

        # ── Security margin chart ─────────────────────────────────────
        st.write("")
        st.markdown("**Security margin — distance from thresholds**")
        qber_pct = qr.qber * 100
        sm_labels = ["QBER", "→ Warning (5%)", "→ Abort (11%)"]
        sm_values = [qber_pct, max(0, 5 - qber_pct), max(0, 11 - qber_pct)]
        sm_colors = [status_color, C_AMBER, C_RED]
        fig_sm = go.Figure()
        fig_sm.add_trace(go.Bar(
            name="QBER", x=["Security margin"],
            y=[qber_pct], marker_color=status_color,
            text=[f"QBER: {qber_pct:.2f}%"], textposition="inside",
        ))
        fig_sm.add_trace(go.Bar(
            name="Margin to Warning", x=["Security margin"],
            y=[max(0, 5 - qber_pct)], marker_color="#FDE68A",
            text=[f"+{max(0, 5 - qber_pct):.2f}% to warning"],
            textposition="inside", textfont_color="#92400E",
        ))
        fig_sm.add_trace(go.Bar(
            name="Margin to Abort", x=["Security margin"],
            y=[max(0, 11 - qber_pct)], marker_color="#FECACA",
            text=[f"+{max(0, 11 - qber_pct):.2f}% to abort"],
            textposition="inside", textfont_color="#991B1B",
        ))
        fig_sm.update_layout(
            **{**_PL, "barmode": "stack", "height": 140,
               "margin": dict(t=4, b=4, l=4, r=4),
               "showlegend": True,
               "legend": dict(orientation="h", y=1.2, font_size=10),
               "xaxis": {**_PL.get("xaxis", {}), "showticklabels": False},
               "yaxis": {**_PL.get("yaxis", {}), "range": [0, 15],
                         "title": "QBER %"}},
        )
        st.plotly_chart(fig_sm, use_container_width=True)

        # ── Shots info card ───────────────────────────────────────────
        st.write("")
        st.markdown(
            "<div style='background:#F0F9FF;border:1px solid #BAE6FD;"
            "border-left:4px solid #0891B2;border-radius:10px;padding:14px 20px;"
            "display:flex;gap:20px;align-items:flex-start;'>"
            "<div>"
            "<div style='font-size:12px;font-weight:700;letter-spacing:.08em;"
            "text-transform:uppercase;color:#0891B2;margin-bottom:6px;'>"
            "Quantum measurement — shots = 1</div>"
            "<div style='font-size:13px;color:#0C4A6E;line-height:1.7;'>"
            "Each qubit is measured <strong>exactly once</strong> (shots = 1). "
            "This is a fundamental constraint of quantum mechanics: "
            "measuring a quantum state collapses it irreversibly. "
            "You cannot re-run the measurement to gather statistics — "
            "the single outcome is all you get. "
            "This is what makes BB84 eavesdropping-detectable: "
            "Eve cannot clone and re-measure qubits without introducing errors."
            "</div></div></div>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH SWEEPS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "research":
    st.markdown(
        "<h2 style='font-size:22px;font-weight:700;color:#111827;margin:0 0 6px;'>"
        "Research Sweeps</h2>"
        "<p style='font-size:13px;color:#9CA3AF;margin:0 0 22px;'>"
        "Sweep a parameter across a range and observe how QBER changes.</p>",
        unsafe_allow_html=True,
    )

    sw_l, sw_m = st.columns([1, 3], gap="large")
    with sw_l:
        st.markdown("**Experiment**")
        experiment = st.radio(
            "",
            [
                "QBER vs Eve intercept probability",
                "QBER vs depolarizing noise",
                "QBER vs fiber length",
            ],
            label_visibility="collapsed",
        )
        st.write("")
        sw_n = st.slider("Qubits / point", 200, 1000, 500, 50, key="sw_n")
        sw_steps = st.slider("Data points", 5, 20, 10, 1, key="sw_steps")
        st.write("")
        run_sweep = st.button("Run Sweep", type="primary", use_container_width=True)

    with sw_m:
        sweep_data = st.session_state.sweep_result
        if run_sweep:
            xs, qbers, ci_l, ci_h, key_rates = [], [], [], [], []
            if experiment == "QBER vs Eve intercept probability":
                x_vals, x_label, is_fiber = (
                    np.linspace(0, 1, sw_steps + 1),
                    "Eve intercept prob",
                    False,
                )
                with st.status("Sweeping…", expanded=True) as _ss:
                    for p in x_vals:
                        cfg = SimulationConfig(
                            n_qubits=sw_n,
                            eve_present=(p > 0),
                            eve_intercept_prob=float(p),
                            noise_enabled=False,
                            seed=42,
                        )
                        res = _run(cfg)
                        xs.append(float(p))
                        qbers.append(res.qber_result.qber * 100)
                        ci_l.append(res.qber_result.confidence_low * 100)
                        ci_h.append(res.qber_result.confidence_high * 100)
                        key_rates.append(res.key_generation_rate * 100)
                        st.write(f"p={p:.2f}  →  QBER {qbers[-1]:.1f}%")
                    _ss.update(label="Done", state="complete", expanded=False)
            elif experiment == "QBER vs depolarizing noise":
                x_vals, x_label, is_fiber = (
                    np.linspace(0, 0.15, sw_steps + 1),
                    "Depolarizing prob (p)",
                    False,
                )
                with st.status("Sweeping…", expanded=True) as _ss:
                    for p in x_vals:
                        cfg = SimulationConfig(
                            n_qubits=sw_n,
                            noise_enabled=(p > 0),
                            noise_model="depolarizing",
                            depolar_prob=float(p),
                            seed=42,
                        )
                        res = _run(cfg)
                        xs.append(float(p))
                        qbers.append(res.qber_result.qber * 100)
                        ci_l.append(res.qber_result.confidence_low * 100)
                        ci_h.append(res.qber_result.confidence_high * 100)
                        key_rates.append(res.key_generation_rate * 100)
                        st.write(f"p={p:.3f}  →  QBER {qbers[-1]:.1f}%")
                    _ss.update(label="Done", state="complete", expanded=False)
            else:
                x_vals, x_label, is_fiber = (
                    np.linspace(0, 150, sw_steps + 1),
                    "Fiber length (km)",
                    True,
                )
                with st.status("Sweeping…", expanded=True) as _ss:
                    for km in x_vals:
                        cfg = SimulationConfig(
                            n_qubits=sw_n,
                            noise_enabled=True,
                            noise_model="fiber_loss",
                            channel_length_km=float(km),
                            seed=42,
                        )
                        res = _run(cfg)
                        xs.append(float(km))
                        qbers.append(res.qber_result.qber * 100)
                        ci_l.append(res.qber_result.confidence_low * 100)
                        ci_h.append(res.qber_result.confidence_high * 100)
                        key_rates.append(res.sifted_key_rate * 100)
                        st.write(f"{km:.0f}km  →  survive {res.sifted_key_rate:.1%}")
                    _ss.update(label="Done", state="complete", expanded=False)
            st.session_state.sweep_result = {
                "x": xs,
                "qbers": qbers,
                "ci_l": ci_l,
                "ci_h": ci_h,
                "key_rates": key_rates,
                "x_label": x_label,
                "is_fiber": is_fiber,
                "experiment": experiment,
            }
            sweep_data = st.session_state.sweep_result

        if sweep_data is None:
            st.info("Select an experiment on the left and click **Run Sweep**.")
        else:
            sw = sweep_data
            is_fiber = sw.get("is_fiber", False)
            specs = (
                [[{"type": "xy"}, {"type": "xy"}]] if is_fiber else [[{"type": "xy"}]]
            )
            fig_sw = make_subplots(
                1,
                2 if is_fiber else 1,
                specs=specs,
                subplot_titles=["QBER (%)", "Sifted rate (%)"]
                if is_fiber
                else ["QBER (%)"],
            )
            fig_sw.add_trace(
                go.Scatter(
                    x=sw["x"],
                    y=sw["qbers"],
                    mode="lines+markers",
                    name="QBER (%)",
                    line=dict(color=C_BLUE, width=2),
                    marker=dict(size=6),
                ),
                1,
                1,
            )
            fig_sw.add_trace(
                go.Scatter(
                    x=sw["x"],
                    y=sw["ci_h"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                1,
                1,
            )
            fig_sw.add_trace(
                go.Scatter(
                    x=sw["x"],
                    y=sw["ci_l"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(37,99,235,.08)",
                    line=dict(width=0),
                    name="95% CI",
                ),
                1,
                1,
            )
            if sw["experiment"] == "QBER vs Eve intercept probability":
                fig_sw.add_trace(
                    go.Scatter(
                        x=sw["x"],
                        y=[0.25 * p * 100 for p in sw["x"]],
                        mode="lines",
                        name="Theory (0.25×p)",
                        line=dict(color=C_RED, dash="dot", width=1.5),
                    ),
                    1,
                    1,
                )
            fig_sw.add_hline(
                y=11,
                line_dash="dot",
                line_color=C_RED,
                annotation_text="Abort 11%",
                annotation_font_size=9,
                row=1,
                col=1,
            )
            fig_sw.add_hline(
                y=5,
                line_dash="dot",
                line_color=C_AMBER,
                annotation_text="Warning 5%",
                annotation_font_size=9,
                row=1,
                col=1,
            )
            if is_fiber:
                fig_sw.add_trace(
                    go.Scatter(
                        x=sw["x"],
                        y=sw["key_rates"],
                        mode="lines+markers",
                        name="Sifted rate (%)",
                        line=dict(color=C_GREEN, width=2),
                        marker=dict(size=6),
                    ),
                    1,
                    2,
                )
            fig_sw.update_layout(**{**_PL, "height": 400, "legend": dict(font_size=10)})
            fig_sw.update_xaxes(title_text=sw["x_label"], title_font_size=11)
            fig_sw.update_yaxes(title_text="QBER (%)", row=1, col=1, title_font_size=11)
            if is_fiber:
                fig_sw.update_yaxes(
                    title_text="Rate (%)", row=1, col=2, title_font_size=11
                )
            st.plotly_chart(fig_sw, use_container_width=True)
            with st.expander("Raw data"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            sw["x_label"]: sw["x"],
                            "QBER (%)": [f"{q:.2f}" for q in sw["qbers"]],
                            "CI low": [f"{c:.2f}" for c in sw["ci_l"]],
                            "CI high": [f"{c:.2f}" for c in sw["ci_h"]],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIO COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

elif page == "compare":
    st.markdown(
        "<h2 style='font-size:22px;font-weight:700;color:#111827;margin:0 0 6px;'>"
        "Scenario Comparison</h2>"
        "<p style='font-size:13px;color:#9CA3AF;margin:0 0 22px;'>"
        "Run multiple preset scenarios side-by-side.</p>",
        unsafe_allow_html=True,
    )

    ct, crt = st.columns([3, 1], gap="large")
    with ct:
        sel = st.multiselect(
            "Scenarios",
            [s[0] for s in PRESET_SCENARIOS],
            default=[s[0] for s in PRESET_SCENARIOS][:4],
        )
        cmp_n = st.slider("Qubits per scenario", 200, 800, 600, 50, key="cmp_n")
    with crt:
        st.write("")
        run_cmp = st.button("Run Comparison", type="primary", use_container_width=True)

    if run_cmp:
        if len(sel) < 2:
            st.warning("Select at least 2 scenarios.")
        else:
            cmp_res = []
            with st.status("Running…", expanded=True) as _cs:
                for name, cfg in [(n, c) for n, c in PRESET_SCENARIOS if n in sel]:
                    cfg2 = SimulationConfig(
                        n_qubits=cmp_n,
                        eve_present=cfg.eve_present,
                        eve_intercept_prob=cfg.eve_intercept_prob,
                        noise_enabled=cfg.noise_enabled,
                        noise_model=cfg.noise_model,
                        depolar_prob=cfg.depolar_prob,
                        t1_ns=cfg.t1_ns,
                        t2_ns=cfg.t2_ns,
                        gate_time_ns=cfg.gate_time_ns,
                        channel_length_km=cfg.channel_length_km,
                        label=cfg.label,
                        seed=42,
                    )
                    res = _run(cfg2)
                    cmp_res.append((name, res))
                    st.write(
                        f"{name}  →  QBER {res.qber_result.qber * 100:.1f}%  ·  "
                        f"{res.qber_result.security_status.strip()}"
                    )
                _cs.update(label="Done", state="complete", expanded=False)
            st.session_state.comparison_results = cmp_res

    cmp_data = st.session_state.comparison_results
    if cmp_data:

        def _bc(res):
            s = res.qber_result.security_status
            return C_GREEN if "SECURE" in s else C_AMBER if "WARNING" in s else C_RED

        res_ls = [d[1] for d in cmp_data]
        names = [d[0] for d in cmp_data]
        colors = [_bc(r2) for r2 in res_ls]
        qbers = [r2.qber_result.qber * 100 for r2 in res_ls]
        ci_lo = [
            (r2.qber_result.qber - r2.qber_result.confidence_low) * 100 for r2 in res_ls
        ]
        ci_hi = [
            (r2.qber_result.confidence_high - r2.qber_result.qber) * 100
            for r2 in res_ls
        ]
        short = [n[:16] + "…" if len(n) > 16 else n for n in names]

        rows = []
        for name, res2 in cmp_data:
            s = res2.qber_result.security_status.strip()
            rows.append(
                {
                    "Scenario": name,
                    "QBER (%)": f"{res2.qber_result.qber * 100:.2f}",
                    "Key bits": res2.key_length,
                    "Agreement (%)": f"{res2.key_agreement_rate * 100:.1f}",
                    "Status": "Secure"
                    if "SECURE" in s
                    else "Warning"
                    if "WARNING" in s
                    else "Abort",
                    "Runtime (s)": f"{res2.runtime_seconds:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.write("")

        fig_cmp = make_subplots(
            1, 3, subplot_titles=["QBER (%)", "Final key (bits)", "Agreement (%)"]
        )
        for ci, (y_vals, y_err) in enumerate(
            [
                (qbers, [ci_lo, ci_hi]),
                ([r2.key_length for r2 in res_ls], None),
                ([r2.key_agreement_rate * 100 for r2 in res_ls], None),
            ],
            start=1,
        ):
            kw = {}
            if y_err:
                kw["error_y"] = dict(
                    type="data",
                    symmetric=False,
                    array=y_err[1],
                    arrayminus=y_err[0],
                    color="#9CA3AF",
                    thickness=1.2,
                    width=6,
                )
            fig_cmp.add_trace(
                go.Bar(
                    x=short,
                    y=y_vals,
                    marker_color=colors,
                    showlegend=False,
                    text=[
                        f"{v:.1f}" if isinstance(v, float) else str(v) for v in y_vals
                    ],
                    textposition="outside",
                    textfont_size=9,
                    **kw,
                ),
                1,
                ci,
            )
            if ci == 1:
                fig_cmp.add_hline(y=11, line_dash="dot", line_color=C_RED, row=1, col=1)
                fig_cmp.add_hline(
                    y=5, line_dash="dot", line_color=C_AMBER, row=1, col=1
                )
        fig_cmp.update_layout(
            **{**_PL, "height": 400, "margin": dict(t=50, b=60, l=10, r=10)}
        )
        fig_cmp.update_xaxes(tickangle=-20, tickfont_size=9)
        fig_cmp.update_yaxes(range=[0, max(35, max(qbers) + 10)], row=1, col=1)
        fig_cmp.update_yaxes(range=[0, 115], row=1, col=3)
        st.plotly_chart(fig_cmp, use_container_width=True)
