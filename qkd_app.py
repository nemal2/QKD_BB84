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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bb84_config import SimulationConfig, SimulationResult
from bb84_noise import NoiseModelType
from bb84_runner import PRESET_SCENARIOS
from bb84_runner import run_simulation as _run
from bb84_zne import run_zne_analysis, ZNEResult
from reconciliation import LDPCReconciler

_ZNE_SUPPORTED_MODELS = (
    NoiseModelType.DEPOLARIZING,
    NoiseModelType.AMPLITUDE_DAMPING,
    NoiseModelType.PHASE_DAMPING,
)
_LDPC_BLOCK_LENS = [20, 40, 80, 160, 320, 640]


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
    "comparison_results": None,
    "last_runtime": None,
    "zne_result": None,
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


@st.cache_resource
def get_ldpc_reconciler(block_len: int, seed: int, calibrate: bool) -> LDPCReconciler:
    """Cached across reruns — LDPCReconciler construction (and optional
    calibration) rebuilds up to 13 sparse parity-check matrices, not
    cheap to redo on every click with the same settings."""
    rec = LDPCReconciler(n=block_len, seed=seed)
    if calibrate:
        rec.calibrate()
    return rec


# ── Top navigation ────────────────────────────────────────────────────────────
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

_pages = [
    ("Guide", "guide"),
    ("Simulator", "sim"),
    ("Analysis", "analysis"),
    ("Compare", "compare"),
]

nh, *_nav_cols = st.columns([2.8] + [1] * 4)

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
        (
            "05",
            "Reconciliation & noise mitigation",
            "LDPC syndrome decoding corrects remaining bit errors between "
            "Alice's and Bob's keys. Zero-Noise Extrapolation (ZNE) reruns "
            "the channel at scaled noise levels and extrapolates back to "
            "estimate the noiseless QBER. Privacy amplification is not yet "
            "implemented.",
        ),
    ]
    step_cols = st.columns(len(steps_data), gap="medium")
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
                "Explore & compare",
                "Use **Analysis** for detailed QBER charts and confidence intervals. "
                "Use **Compare** to run multiple preset scenarios side-by-side "
                "and see how different noise models and attack strategies affect the key.",
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
        "Simulation engine: Qiskit AerSimulator  ·  "
        "Physically accurate noise models via Kraus operators</span></div>",
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
                n_qubits=1200,
                sample_fraction=0.25,
                noise_enabled=True,
                noise_model="amplitude_damping",
                t1_ns=3_000,          # 3 µs — was 10 µs (gamma ~0.05%, invisible)
                t2_ns=3_000,          # must satisfy T2 <= 2*T1; pinned to T1
                gate_time_ns=50,
                label="Amp.Damp",
            ),
            "Fiber 50km": SimulationConfig(
                n_qubits=800,
                noise_enabled=True,
                noise_model="fibre_loss",
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
            n_qubits = st.slider("Number of qubits", 100, 2000, 1000, 50, key="s_n")
        with mp2:
            sample_pct = st.slider("QBER sample (%)", 5, 30, 20, 1, key="s_sp")
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
            # BUGFIX: this must default to IDEAL, not "depolarizing".
            # bb84_noise.QuantumChannel.from_config() only respects
            # `noise_enabled` when config.noise_model is None — any concrete
            # string here (even the old "depolarizing" placeholder) makes it
            # apply that model unconditionally, so unchecking the box did
            # nothing. IDEAL is a real, safe no-op noise model.
            noise_model = NoiseModelType.IDEAL
            depolar_prob = 0.01
            t1_ns = 100_000.0
            t2_ns = 50_000.0
            gate_time_ns = 50.0
            channel_length_km = 0.0

            if noise_enabled:
                noise_model = st.selectbox(
                    "Model",
                    [
                        "depolarizing",
                        "amplitude_damping",
                        "phase_damping",
                        "combined",
                        "fibre_loss",
                    ],
                    key="s_nm",
                    format_func=lambda x: {
                        "depolarizing": "Depolarizing (Pauli)",
                        "amplitude_damping": "Amplitude Damping (T1)",
                        "phase_damping": "Phase Damping (T2)",
                        "combined": "Combined T1 + T2",
                        "fibre_loss": "Fiber Loss",
                    }[x],
                )
                # Rough resolution floor: with this many sifted+sampled bits,
                # a true error rate well below ~1/sqrt(sample_size) will
                # usually look like "no effect" in a single run, buried in
                # the Wilson CI. Warn instead of leaving people guessing.
                _approx_sample_n = max(1, int(n_qubits * 0.5 * sample_fraction))
                _res_floor = 1.0 / math.sqrt(_approx_sample_n)

                def _effect_caption(expected_qber: float, label: str) -> None:
                    pct = expected_qber * 100
                    if expected_qber < _res_floor:
                        st.caption(
                            f"{label} ≈ {pct:.2f}% — below this run's ~"
                            f"{_res_floor * 100:.1f}% statistical resolution "
                            f"(~{_approx_sample_n} sampled bits). A single run "
                            f"will likely look flat; raise the noise, qubits, "
                            f"or QBER sample % to see a clear trend."
                        )
                    else:
                        st.caption(f"{label} ≈ {pct:.2f}%")

                if noise_model == "depolarizing":
                    depolar_prob = st.slider(
                        "Gate error prob",
                        0.001,
                        0.20,
                        0.05,
                        0.001,
                        format="%.3f",
                        key="s_dp",
                    )
                    st.caption(f"p/3 = {depolar_prob / 3:.5f} per Pauli")
                    # Sifted qubits see 0, 1, or 2 noisy gates depending on
                    # basis/bit (avg 1.25); each exposure flips the bit with
                    # prob ~2p/3 (X or Y error).
                    _p_flip = 2 * depolar_prob / 3
                    _expected = 1 - (1 - _p_flip) ** 1.25
                    _effect_caption(_expected, "Expected QBER")
                elif noise_model == "amplitude_damping":
                    # Rescaled from the old 1-500 µs range: at 50 ns gate
                    # time, anything above ~50 µs is indistinguishable from
                    # ideal (gamma < 0.1%). 0.1-50 µs keeps the slider's
                    # low end usable (gamma up to ~40%) and moves the
                    # default into a range with a real, checkable effect.
                    t1_us = st.slider("T1 (µs)", 0.1, 50.0, 3.0, 0.1, key="s_t1")
                    gate_time_ns = st.slider(
                        "Gate time (ns)", 10, 200, 50, 5, key="s_gtad"
                    )
                    t1_ns = t1_us * 1000
                    # t2_ns is unused by amplitude damping's Kraus operators,
                    # but SimulationConfig always validates T2 <= 2*T1 —
                    # pin it to T1 so an unrelated default can't trip that
                    # check for this model.
                    t2_ns = t1_ns
                    gamma = 1.0 - math.exp(-gate_time_ns / t1_ns)
                    st.caption(f"γ = {gamma:.6f}")
                    _effect_caption(gamma * 1.25, "Expected QBER")
                elif noise_model == "phase_damping":
                    # Rescaled for the same reason as T1 above.
                    t2_us = st.slider("T2 (µs)", 0.05, 50.0, 3.0, 0.05, key="s_t2p")
                    gate_time_ns = st.slider(
                        "Gate time (ns)", 10, 200, 50, 5, key="s_gtpd"
                    )
                    t2_ns = t2_us * 1000
                    # t1_ns is unused by phase damping's Kraus operators,
                    # but SimulationConfig always validates T2 <= 2*T1 —
                    # pin it to T2 so an unrelated default can't trip that
                    # check for this model.
                    t1_ns = t2_ns
                    lam = 1.0 - math.exp(-gate_time_ns / t2_ns)
                    st.caption(f"λ = {lam:.6f}")
                    # Pure dephasing only shows up as errors for qubits
                    # measured in the diagonal (X) basis, i.e. ~half of
                    # sifted bits, and roughly halves the visible flip rate.
                    _effect_caption(lam * 0.5 * 1.25, "Expected QBER")
                elif noise_model == "combined":
                    # Rescaled for the same reason as T1/T2 above.
                    t1_us = st.slider("T1 (µs)", 0.1, 50.0, 4.0, 0.1, key="s_t1c")
                    t2_us = st.slider("T2 (µs)", 0.05, 25.0, 3.0, 0.05, key="s_t2c")
                    gate_time_ns = st.slider(
                        "Gate time (ns)", 10, 200, 50, 5, key="s_gtc"
                    )
                    t1_ns = t1_us * 1000
                    t2_ns = min(t2_us * 1000, 2.0 * t1_ns - 1.0)
                    if t2_us * 1000 > 2 * t1_ns:
                        st.warning("T2 clamped to 2·T1")
                    _gamma_c = 1.0 - math.exp(-gate_time_ns / t1_ns)
                    _lam_c = 1.0 - math.exp(-gate_time_ns / t2_ns)
                    _effect_caption(
                        (_gamma_c + _lam_c * 0.5) * 1.25, "Expected QBER"
                    )
                elif noise_model == "fibre_loss":
                    channel_length_km = st.slider(
                        "Channel length (km)", 0.0, 200.0, 50.0, 5.0, key="s_km"
                    )
                    st.caption(
                        "Fibre loss drops photons before they reach Bob — it "
                        "lowers the sifted/final key **rate**, not the QBER. "
                        "Expect the QBER to stay flat as you increase distance; "
                        "that's correct, not a bug."
                    )
                    if channel_length_km > 0:
                        survive = 10 ** (-0.2 * channel_length_km / 10)
                        st.caption(
                            f"P(survive) = {survive:.4f}  ·  "
                            f"Loss = {(1 - survive) * 100:.1f}%"
                        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

        # Error correction (LDPC)
        st.markdown(
            "<div style='font-size:12px;font-weight:600;color:#374151;"
            "margin-bottom:8px;'>Error Correction (LDPC)</div>",
            unsafe_allow_html=True,
        )
        lc1, lc2, lc3 = st.columns([1, 1, 2])
        with lc1:
            ldpc_enabled = st.checkbox("Enable LDPC reconciliation", key="s_ldpc")
        with lc2:
            ldpc_block_len = st.selectbox(
                "Block length",
                _LDPC_BLOCK_LENS,
                index=_LDPC_BLOCK_LENS.index(160),
                key="s_ldpc_bl",
                disabled=not ldpc_enabled,
            )
        with lc3:
            ldpc_calibrate = st.checkbox(
                "Calibrate code rates (slower, more accurate)",
                key="s_ldpc_cal",
                disabled=not ldpc_enabled,
            )
        if ldpc_enabled:
            st.caption(
                "Reconciles Alice's and Bob's key in fixed blocks after QBER "
                "sampling, using LDPC syndrome decoding. Exact block "
                "accounting (blocks used, bits leaked) appears in the "
                "results below once the final key length is known — keys "
                "shorter than 20 bits skip reconciliation entirely."
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
                st.markdown(
                    f"<div style='text-align:right;padding-top:6px;'>"
                    f"<span style='font-size:12px;color:#6B7280;'>"
                    f"Last: <span style='font-family:JetBrains Mono,monospace;'>"
                    f"{st.session_state.last_runtime:.3f}s [Qiskit]</span></span></div>",
                    unsafe_allow_html=True,
                )

    # ── Run logic ─────────────────────────────────────────────────────────
    if run_clicked or preset_clicked is not None:
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

        # Defensive belt-and-braces: SimulationConfig always enforces the
        # physical bound T2 <= 2*T1, but only the "combined" noise branch
        # above actively keeps both sliders in that relationship. Clamp
        # here so any future model/UI path that forgets this degrades
        # gracefully with a warning instead of raising ValueError.
        if t2_ns > 2 * t1_ns:
            st.warning(
                f"T2 ({t2_ns / 1000:.1f} µs) exceeded the physical bound "
                f"2×T1 ({2 * t1_ns / 1000:.1f} µs) — clamped to stay valid."
            )
            t2_ns = 2 * t1_ns - 1.0

        cfg = SimulationConfig(
            n_qubits=n_qubits,
            eve_present=eve_present,
            eve_intercept_prob=eve_intercept_prob,
            noise_enabled=noise_enabled,
            noise_model=noise_model,
            depolar_prob=depolar_prob,
            t1_ns=t1_ns,
            t2_ns=t2_ns,
            gate_time_ns=gate_time_ns,
            channel_length_km=channel_length_km,
            sample_fraction=sample_fraction,
            seed=seed,
            label=sim_label,
            ldpc_enabled=ldpc_enabled,
            ldpc_block_len=ldpc_block_len,
            ldpc_calibrate=ldpc_calibrate,
        )
        try:
            with st.status("Running simulation…", expanded=True) as _s:
                st.write(f"Transmitting {cfg.n_qubits:,} qubits…")
                t0 = time.time()
                ldpc_reconciler = None
                if cfg.ldpc_enabled:
                    ldpc_reconciler = get_ldpc_reconciler(
                        cfg.ldpc_block_len,
                        cfg.ldpc_seed if cfg.ldpc_seed is not None
                        else (cfg.seed if cfg.seed is not None else 0),
                        cfg.ldpc_calibrate,
                    )
                result = _run(cfg, verbose=False, ldpc_reconciler=ldpc_reconciler)
                elapsed = time.time() - t0
                st.write(
                    f"Sifted: {result.n_sifted:,} bits  ({result.sifted_key_rate:.1%})"
                )
                st.write(
                    f"QBER: {result.qber_result.qber * 100:.2f}%  —  "
                    f"{result.qber_result.security_status.strip()}"
                )
                if result.ldpc_result is not None:
                    lr = result.ldpc_result
                    st.write(
                        f"LDPC: {lr.n_blocks} blocks reconciled  ·  "
                        f"net key {lr.net_key_bits} bits  ·  "
                        f"{'no undetected errors' if not lr.any_undetected_error else 'undetected error flagged'}"
                    )
                _s.update(
                    label=f"Complete  ·  {elapsed:.3f} s",
                    state="complete",
                    expanded=False,
                )
            st.session_state.result = result
            st.session_state.last_runtime = elapsed
            st.session_state.zne_result = None
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
            "Qiskit",
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
            funnel_y = ["Transmitted", "Sifted", "After QBER sample", "Final key"]
            funnel_x = [
                r.n_transmitted,
                r.n_sifted,
                r.n_sifted - qr.sample_size,
                r.key_length,
            ]
            funnel_colors = [C_BLUE, C_TEAL, C_AMBER, C_GREEN]
            if r.ldpc_result is not None:
                funnel_y.append("Reconciled key")
                funnel_x.append(r.ldpc_result.net_key_bits)
                funnel_colors.append(C_PURPLE)
            fig_f = go.Figure(
                go.Funnel(
                    y=funnel_y,
                    x=funnel_x,
                    textinfo="value+percent initial",
                    marker=dict(color=funnel_colors),
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
            ch = "Ideal"
            if cfg_r.noise_enabled:
                ch = {
                    "depolarizing": f"Depolarizing  p={cfg_r.depolar_prob:.3f}",
                    "amplitude_damping": f"Amp. damp  T1={cfg_r.t1_ns / 1000:.0f} µs",
                    "phase_damping": f"Phase damp  T2={cfg_r.t2_ns / 1000:.0f} µs",
                    "combined": "T1+T2",
                    "fibre_loss": f"Fiber {cfg_r.channel_length_km:.0f} km",
                }.get(cfg_r.noise_model, cfg_r.noise_model)
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
            if r.ldpc_result is not None:
                st.download_button(
                    "Reconciled Alice key (.txt)",
                    "\n".join(str(b) for b in r.ldpc_result.reconciled_alice_key),
                    "reconciled_alice_key.txt",
                    "text/plain",
                    use_container_width=True,
                )
                st.download_button(
                    "Reconciled Bob key (.txt)",
                    "\n".join(str(b) for b in r.ldpc_result.reconciled_bob_key),
                    "reconciled_bob_key.txt",
                    "text/plain",
                    use_container_width=True,
                )

            results_json = {
                "label": r.config.label,
                "qber": r.qber_result.qber,
                "n_transmitted": r.n_transmitted,
                "n_sifted": r.n_sifted,
                "key_length": r.key_length,
                "key_agreement_rate": r.key_agreement_rate,
                "eve_interception_rate": r.eve_interception_rate,
                "runtime_seconds": r.runtime_seconds,
            }
            if r.ldpc_result is not None:
                lr = r.ldpc_result
                results_json["ldpc"] = {
                    "block_len": lr.block_len,
                    "n_blocks": lr.n_blocks,
                    "remainder_bits": lr.remainder_bits,
                    "failed_block_bits": lr.failed_block_bits,
                    "net_key_bits": lr.net_key_bits,
                    "total_leaked_bits": lr.total_leaked_bits,
                    "any_undetected_error": lr.any_undetected_error,
                }
            st.download_button(
                "Results (.json)",
                json.dumps(results_json, indent=2),
                "qkd_results.json",
                "application/json",
                use_container_width=True,
            )

        # ── LDPC reconciliation results ─────────────────────────────────
        if r.ldpc_result is not None:
            lr = r.ldpc_result
            st.divider()
            st.markdown("**LDPC reconciliation**")
            st.caption(
                "Net key bits is a Shannon-cost estimate of what would remain "
                "after (hypothetical) privacy amplification — not yet "
                "implemented, so the reconciled key exported above is the "
                "full error-corrected key, not a shortened one. "
                "\"Actually correct\" is checked against Alice's key directly, "
                "which is only possible in simulation."
            )
            lk1, lk2, lk3, lk4 = st.columns(4)
            lk1.metric("Blocks reconciled", f"{lr.n_blocks}",
                       f"of {lr.n_blocks} attempted")
            lk2.metric("Net key bits", f"{lr.net_key_bits:,}",
                       f"leaked {lr.total_leaked_bits}")
            leak_rate = (lr.total_leaked_bits / lr.total_input_bits * 100
                         if lr.total_input_bits else 0.0)
            lk3.metric("Leak rate", f"{leak_rate:.1f}%", "of input bits")
            if lr.any_undetected_error:
                lk4.metric("Undetected error", "Yes", "decoder wrong, unflagged")
            elif not lr.all_blocks_correct:
                lk4.metric("Undetected error", "No", "but a block failed safely")
            else:
                lk4.metric("Undetected error", "No", "all blocks verified correct")

            if lr.n_blocks > 0:
                block_colors = []
                for b in lr.blocks:
                    if b.claimed_success and not b.actually_correct:
                        block_colors.append(C_RED)
                    elif not b.actually_correct:
                        block_colors.append(C_AMBER)
                    else:
                        block_colors.append(C_GREEN)
                fig_ldpc = make_subplots(specs=[[{"secondary_y": False}]])
                fig_ldpc.add_trace(go.Bar(
                    x=list(range(1, lr.n_blocks + 1)),
                    y=[b.leaked_bits for b in lr.blocks],
                    name="Leaked bits",
                    marker_color=block_colors,
                    marker_opacity=0.55,
                ))
                fig_ldpc.add_trace(go.Bar(
                    x=list(range(1, lr.n_blocks + 1)),
                    y=[(lr.block_len - b.leaked_bits) if b.actually_correct else 0
                       for b in lr.blocks],
                    name="Net bits",
                    marker_color=block_colors,
                ))
                fig_ldpc.update_layout(
                    **{**_PL, "height": 240, "barmode": "stack",
                       "margin": dict(t=20, b=30, l=10, r=10),
                       "legend": dict(font_size=10, orientation="h", y=1.12)}
                )
                fig_ldpc.update_xaxes(title_text="Block", tickfont_size=9)
                st.plotly_chart(fig_ldpc, use_container_width=True)
                st.caption(
                    "Green = correctly reconciled  ·  Amber = decoder safely "
                    "flagged failure  ·  Red = undetected error (claimed "
                    "success but wrong)."
                )

        # ── ZNE analysis (separate action, own qubit-count control) ─────
        st.divider()
        st.markdown("**Zero-Noise Extrapolation (ZNE)**")

        zne_blocked_reason = None
        if not r.config.noise_enabled:
            zne_blocked_reason = (
                "ZNE scales *hardware* noise and extrapolates back to f=0 — "
                "there's no noise enabled on this run for it to scale. Enable "
                "a noise model on the Simulator page above (Eve's "
                "intercept-resend attack alone won't move with the "
                "noise-scale factor, so running ZNE without noise just "
                "repeats the same QBER at every f)."
            )
        elif r.config.noise_model not in _ZNE_SUPPORTED_MODELS:
            if r.config.noise_model == "combined":
                zne_blocked_reason = (
                    "ZNE is disabled for the Combined T1+T2 model. Linear "
                    "extrapolation gave no reliable correction under combined "
                    "T1/T2 noise in evaluation — run depolarizing, amplitude "
                    "damping, or phase damping individually instead."
                )
            else:
                zne_blocked_reason = (
                    "ZNE requires a scalable Kraus noise model (depolarizing, "
                    "amplitude damping, or phase damping). The last run used "
                    f"`{r.config.noise_model or 'ideal'}`, for which "
                    "noise-scaling has no defined meaning (photon loss or a "
                    "noiseless channel)."
                )

        if zne_blocked_reason:
            st.info(zne_blocked_reason)
        else:
            with st.container(border=True):
                zc1, zc2, zc3, zc4 = st.columns([2, 1, 1, 1])
                with zc1:
                    zne_f_scales = st.multiselect(
                        "Noise scale factors",
                        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                        default=[1.0, 1.5, 2.0, 2.5, 3.0],
                        key="s_zne_f",
                    )
                with zc2:
                    zne_n_qubits = st.slider(
                        "Qubits per point", 200, 1200, 600, 100, key="s_zne_nq"
                    )
                    st.caption("Separate from the main run — keeps the sweep fast.")
                with zc3:
                    zne_n_seeds = st.slider("Seeds per point", 3, 10, 5, 1, key="s_zne_ns")
                with zc4:
                    zne_method = st.selectbox(
                        "Fit method", ["linear", "exponential"], key="s_zne_m",
                        help=(
                            "Linear is the recommended default. The "
                            "exponential ansatz is unreliable at this "
                            "f-resolution and silently falls back to linear "
                            "when it doesn't converge cleanly."
                        ),
                    )
                zne_bootstrap = st.checkbox(
                    "Compute bootstrap confidence interval (slower)", key="s_zne_boot"
                )
                if r.config.eve_present:
                    st.caption(
                        "Eve's intercept-resend attack is active. The f=0 "
                        "intercept will still include Eve's contribution — "
                        "ZNE removes *hardware* noise, not eavesdropping. "
                        "Read it as \"QBER attributable to Eve once hardware "
                        "noise is extrapolated away,\" not as a clean-channel "
                        "estimate."
                    )
                run_zne_clicked = st.button(
                    "Run ZNE Analysis", type="primary", key="s_zne_run"
                )

            if run_zne_clicked:
                if len(zne_f_scales) < 2:
                    st.warning("Select at least 2 noise scale factors.")
                else:
                    zne_base_dict = {**r.config.__dict__, "n_qubits": zne_n_qubits}
                    # Defensive belt-and-braces (same reasoning as the main
                    # run's clamp above): r.config was already valid when
                    # the main run built it, but copying its __dict__ here
                    # is a second, independent construction path, so if a
                    # future noise-model branch or a stale cached module
                    # ever produces a mismatched (t1_ns, t2_ns) pair, clamp
                    # it instead of letting SimulationConfig raise and kill
                    # the whole ZNE sweep.
                    if zne_base_dict["t2_ns"] > 2 * zne_base_dict["t1_ns"]:
                        st.warning(
                            f"T2 ({zne_base_dict['t2_ns'] / 1000:.1f} µs) exceeded "
                            f"the physical bound 2×T1 "
                            f"({2 * zne_base_dict['t1_ns'] / 1000:.1f} µs) for the "
                            f"ZNE base config — clamped to stay valid."
                        )
                        zne_base_dict["t2_ns"] = 2 * zne_base_dict["t1_ns"] - 1.0
                    zne_base_cfg = SimulationConfig(**zne_base_dict)
                    try:
                        with st.status("Running ZNE sweep…", expanded=True) as _zs:
                            st.write(
                                f"{len(sorted(set(zne_f_scales)))} scale factors × "
                                f"{zne_n_seeds} seeds = "
                                f"{len(set(zne_f_scales)) * zne_n_seeds} simulations…"
                            )
                            zne_result = run_zne_analysis(
                                zne_base_cfg, sorted(set(zne_f_scales)),
                                n_seeds=zne_n_seeds, method=zne_method,
                                bootstrap=zne_bootstrap,
                            )
                            _zs.update(
                                label=f"Complete  ·  {zne_result.runtime_seconds:.2f} s",
                                state="complete", expanded=False,
                            )
                        st.session_state.zne_result = zne_result
                    except (ValueError, KeyError, TypeError) as e:
                        st.error(f"ZNE error: {e}")

            zr_state: Optional[ZNEResult] = st.session_state.zne_result
            if zr_state is not None and zr_state.noise_model == r.config.noise_model:
                f_sorted = sorted(zr_state.per_f_qber)
                means = [zr_state.per_f_qber[f][0] for f in f_sorted]
                lo = [means[i] - zr_state.per_f_qber[f_sorted[i]][1] for i in range(len(f_sorted))]
                hi = [zr_state.per_f_qber[f_sorted[i]][2] - means[i] for i in range(len(f_sorted))]

                fig_zne = go.Figure()
                fig_zne.add_trace(go.Scatter(
                    x=f_sorted, y=means, mode="markers",
                    error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo,
                                 color="#9CA3AF", thickness=1.2, width=6),
                    marker=dict(color=C_BLUE, size=9),
                    name="Measured QBER",
                ))
                fit_x = [0.0] + f_sorted
                fit_y = [zr_state.linear_intercept + zr_state.linear_slope * f for f in fit_x]
                fig_zne.add_trace(go.Scatter(
                    x=fit_x, y=fit_y, mode="lines",
                    line=dict(color=C_BLUE, dash="dash", width=1.5),
                    name="Linear fit",
                ))
                fig_zne.add_trace(go.Scatter(
                    x=[0.0], y=[zr_state.recommended_estimate], mode="markers",
                    marker=dict(color=C_GREEN, size=13, symbol="star"),
                    name="ZNE estimate (f=0)",
                ))
                if 1.0 in zr_state.per_f_qber:
                    fig_zne.add_trace(go.Scatter(
                        x=[1.0], y=[zr_state.qber_at_f1], mode="markers",
                        marker=dict(color=C_AMBER, size=11, symbol="diamond"),
                        name="Raw (no ZNE)",
                    ))
                fig_zne.update_layout(
                    **{**_PL, "height": 320,
                       "margin": dict(t=20, b=30, l=10, r=10),
                       "legend": dict(font_size=10, orientation="h", y=1.1)}
                )
                fig_zne.update_xaxes(title_text="Noise scale factor (f)")
                fig_zne.update_yaxes(title_text="QBER (%)")
                st.plotly_chart(fig_zne, use_container_width=True)

                # Was exponential requested but silently fell back to linear?
                exp_fell_back = (
                    zne_method == "exponential" and not zr_state.exponential["converged"]
                )
                fit_used_label = (
                    "linear fit (exponential requested, fell back)"
                    if exp_fell_back else f"{zne_method} fit"
                )

                zk1, zk2, zk3 = st.columns(3)
                zk1.metric("Raw QBER (f=1)", f"{zr_state.qber_at_f1:.2f}%",
                           "what you'd report without ZNE")
                zk2.metric("ZNE estimate (f=0)", f"{zr_state.recommended_estimate:.2f}%",
                           fit_used_label)
                if zr_state.bootstrap_ci is not None:
                    bmean, blo, bhi = zr_state.bootstrap_ci
                    zk3.metric("Bootstrap 95% CI", f"[{blo:.2f}, {bhi:.2f}]%",
                               f"mean {bmean:.2f}%")
                else:
                    zk3.metric("Exponential converged",
                               "Yes" if zr_state.exponential["converged"] else "No")

                if exp_fell_back:
                    st.caption(
                        "⚠️ The exponential fit didn't converge cleanly (its "
                        "intercept uncertainty was too large relative to its "
                        "own estimate) — the number above is the linear fit "
                        "instead, consistent with the exponential ansatz being "
                        "unreliable at this f-resolution."
                    )

                # Bias-reduction readout.
                if zr_state.qber_at_f1 > 1e-9:
                    reduction_pct = (
                        (zr_state.qber_at_f1 - zr_state.recommended_estimate)
                        / zr_state.qber_at_f1 * 100
                    )
                    if reduction_pct < 10:
                        st.caption(
                            f"Bias reduction here is only {reduction_pct:.1f}% — "
                            "at low base-noise settings ZNE has little to "
                            "correct for; this matches evaluation showing no "
                            "net benefit at the lowest tested noise level."
                        )
                    else:
                        st.caption(f"Bias reduction vs. raw QBER: {reduction_pct:.1f}%.")

                st.info(
                    "**Positioning:** report the ZNE estimate *alongside* raw "
                    "QBER, never in place of it. The corrected estimate is "
                    "systematically lower than raw QBER by construction — "
                    "privacy amplification must still be sized off a "
                    "conservative bound, not this number."
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

        zr_state: Optional[ZNEResult] = st.session_state.zne_result
        zne_applies = zr_state is not None and zr_state.noise_model == r.config.noise_model
        if r.ldpc_result is not None or zne_applies:
            st.divider()
            st.markdown("**Reconciliation & Mitigation**")
            st.caption(
                "Compares the raw measured QBER against the ZNE-extrapolated "
                "zero-noise estimate and the post-LDPC key agreement, where "
                "available — run LDPC/ZNE on the Simulator page first."
            )
            rm_cols = st.columns(3)
            rm_cols[0].metric(
                "Raw QBER", f"{qr.qber * 100:.2f}%",
                r.qber_result.security_status.strip(),
            )
            if zne_applies:
                rm_cols[1].metric(
                    "ZNE estimate (f=0)", f"{zr_state.recommended_estimate:.2f}%",
                    "noise-mitigated"
                    if zr_state.recommended_estimate < qr.qber * 100
                    else "no improvement",
                )
            else:
                rm_cols[1].metric("ZNE estimate", "—", "not run for this config")
            if r.ldpc_result is not None:
                lr = r.ldpc_result
                agree_label = (
                    "all blocks match" if lr.all_blocks_correct
                    else "undetected error" if lr.any_undetected_error
                    else "some blocks failed"
                )
                has_reconciled_key = len(lr.reconciled_alice_key) > 0
                rm_cols[2].metric(
                    "Post-LDPC agreement",
                    "100.0%" if (has_reconciled_key and lr.keys_match) else "—",
                    agree_label if has_reconciled_key else "no blocks succeeded",
                )
            else:
                rm_cols[2].metric("Post-LDPC agreement", "—", "not run for this config")


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
                    res = _run(cfg2, verbose=False)
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