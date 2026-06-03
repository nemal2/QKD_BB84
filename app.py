"""
BB84 QKD Simulator - Enhanced Web Interface
============================================
Interactive Streamlit UI for BB84 Phase 1 & 3 simulations.
Supports classical noise models, Eve attacks, and Phase 3 quantum channel losses.

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from bb84_config import SimulationConfig, SimulationResult
from bb84_runner import run_simulation, run_comparison, PRESET_SCENARIOS
from bb84_plots import plot_comparison

# Define scenario subsets locally
PHASE1_SCENARIOS = PRESET_SCENARIOS[:3]
PHASE3_SCENARIOS = PRESET_SCENARIOS[3:]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BB84 QKD Simulator - Phase 1 & 3",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔐 BB84 Quantum Key Distribution Simulator")
st.markdown("**Interactive interface for Phase 1 & 3 simulations** — Classical channels, Quantum noise models, Eve attacks")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - NAVIGATION & CACHING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_cached_simulation(config_dict: str, seed: int) -> SimulationResult:
    """Cache simulation results by configuration."""
    import json
    config_data = json.loads(config_dict)
    config = SimulationConfig(**config_data)
    return run_simulation(config, verbose=False)

def run_cached_simulation(config: SimulationConfig) -> SimulationResult:
    """Run simulation with caching based on config parameters."""
    import json
    config_dict = json.dumps({
        'n_qubits': config.n_qubits,
        'eve_present': config.eve_present,
        'eve_intercept_prob': config.eve_intercept_prob,
        'noise_enabled': config.noise_enabled,
        'noise_model': config.noise_model,
        'depolar_prob': config.depolar_prob,
        't1_ns': config.t1_ns,
        't2_ns': config.t2_ns,
        'gate_time_ns': config.gate_time_ns,
        'channel_length_km': config.channel_length_km,
        'sample_fraction': config.sample_fraction,
        'seed': config.seed if config.seed is not None else 42,
        'label': config.label,
    }, default=str)
    return get_cached_simulation(config_dict, config.seed if config.seed is not None else 42)

page = st.sidebar.radio(
    "📋 Navigation",
    ["Home", "Phase 1 Simulations", "Phase 3 (Noise Models)", "Preset Comparison", "Security Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📚 Documentation**")
st.sidebar.markdown("- Phase 1: Classical BB84 protocol")
st.sidebar.markdown("- Phase 3: Quantum channel noise & fiber loss")
st.sidebar.markdown("- Phase 4: Attack models (coming soon)")

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════

if page == "Home":
    st.header("Welcome to BB84 Quantum Key Distribution Simulator")
    
    st.markdown("""
    This simulator implements the **BB84 Quantum Key Distribution Protocol**, a foundational quantum cryptography 
    algorithm that enables two parties (Alice and Bob) to establish a shared secret key secure against eavesdropping.
    """)
    
    st.subheader("🔐 What is BB84?")
    st.markdown("""
    The BB84 protocol works in 4 steps:
    1. **Qubit Preparation**: Alice randomly prepares qubits in one of two bases (rectilinear or diagonal)
    2. **Transmission**: Alice sends qubits to Bob through a quantum channel (potentially intercepted by Eve)
    3. **Measurement**: Bob randomly measures each qubit in one of two bases
    4. **Basis Reconciliation**: Alice and Bob publicly compare bases (not bits) and keep only matches
    
    The final shared key is distilled from bits where both parties used the same basis.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Phase 1: Baseline")
        st.markdown("""
        - **Ideal Channel**: No noise, no attacks
        - **Eve Attacks**: Intercept-resend at variable rates (0-100%)
        - **QBER Estimation**: Detects eavesdropping via quantum bit errors
        - **Key Agreement**: Measures legitimate key generation
        """)
    
    with col2:
        st.subheader("⚛️ Phase 3: Quantum Noise")
        st.markdown("""
        - **Depolarizing**: Random Pauli errors (uniform)
        - **Amplitude Damping (T1)**: Energy relaxation |1⟩→|0⟩
        - **Phase Damping (T2)**: Coherence loss via dephasing
        - **Combined**: Realistic T1+T2 noise model
        - **Fiber Loss**: Distance-based photon loss
        """)
    
    with col3:
        st.subheader("⚡ Performance Features")
        st.markdown("""
        - **Batch Processing**: 3-5x speedup vs sequential
        - **Statevector Simulator**: 10x faster than qasm method
        - **Smart Caching**: Instant re-runs of same config
        - **Optimized Eve**: Pre-batched intercept calculations
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Key Metrics Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **QBER (Quantum Bit Error Rate)**
        - Fraction of sampled bits that disagree between Alice and Bob
        - Ideal: ~0%, With Eve: ~25%, Threshold for abort: ~11%
        - Used to detect eavesdropping attempts
        
        **Sifted Key Rate**
        - Fraction of transmitted qubits that survive sifting
        - Theoretical ideal: 50% (half of bases match)
        - Actual: ~50% ± noise effects
        """)
    
    with col2:
        st.markdown("""
        **Key Agreement Rate**
        - Fraction of final key bits that match between Alice and Bob
        - Ideal (no noise, no Eve): ~100%
        - With noise: 99-99.9% (errors detected & removed)
        - Indicates protocol correctness
        
        **Eve Interception Rate**
        - Percentage of qubits Eve actually measured
        - Eve's action introduces detectable QBER increase
        - Higher intercept → easier detection
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 - SINGLE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Phase 1 Simulations":
    st.header("Phase 1: Single Simulation")
    st.markdown("Configure and run a single BB84 simulation with optional eavesdropping attacks.")
    
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔧 Protocol Parameters")
            st.markdown("**Configure the BB84 transmission:**")
            
            n_qubits = st.slider(
                "Total Qubits", 
                100, 5000, 600, step=100,
                help="Total quantum bits Alice transmits. More qubits = longer key but slower simulation. Range: 100-5000. Typical: 600-2000."
            )
            
            eve_present = st.checkbox(
                "Eve Present?", 
                value=False,
                help="Enable eavesdropping: Eve attempts intercept-resend attack on qubits"
            )
            
            if eve_present:
                eve_intercept_prob = st.slider(
                    "Eve Intercept Rate", 
                    0.0, 1.0, 1.0, step=0.1,
                    help="Fraction of qubits Eve intercepts [0.0-1.0]. 0.0=no attack, 1.0=intercept all qubits. Higher = more detectable QBER."
                )
            else:
                eve_intercept_prob = 0.0
            
            sample_fraction = st.slider(
                "QBER Sample Fraction", 
                0.05, 0.50, 0.10, step=0.05,
                help="Fraction of sifted key used for QBER estimation [0.05-0.50]. Higher = better statistics but shorter final key. Typical: 0.10-0.15."
            )
        
        with col2:
            st.subheader("🔐 Quantum Channel Noise")
            st.markdown("**Add realistic quantum noise:**")
            
            noise_enabled = st.checkbox(
                "Enable Noise?", 
                value=False,
                help="Add channel noise to simulate real quantum systems"
            )
            
            if noise_enabled:
                noise_model = st.selectbox(
                    "Noise Model",
                    ["depolarizing", "amplitude_damp", "phase_damp", "combined", "fiber_loss"],
                    help="Type of quantum noise to simulate"
                )
                
                if noise_model == "depolarizing":
                    st.info("**Depolarizing**: Random Pauli errors with uniform probability")
                    depolar_prob = st.slider(
                        "Error Probability", 
                        0.0, 0.1, 0.01, step=0.005,
                        help="Probability of random qubit flip [0.0-0.1]. Typical: 0.01-0.05 (1-5% error rate)."
                    )
                    t1_ns, t2_ns, gate_time_ns, channel_length_km = 0, 0, 0, 0
                
                elif noise_model == "amplitude_damp":
                    st.info("**Amplitude Damping (T1)**: Energy relaxation |1⟩ decays to |0⟩ over time")
                    t1_us = st.slider(
                        "T1 Relaxation Time (µs)", 
                        1, 500, 10, step=5,
                        help="Energy relaxation timescale (microseconds). Superconducting qubits: 10-500 µs. Lower T1 = more noise."
                    )
                    gate_time_ns = st.slider(
                        "Gate Time (ns)", 
                        1, 100, 50, step=5,
                        help="Duration of quantum gates (nanoseconds). Typical: 20-100 ns. Affects noise accumulation per operation."
                    )
                    t1_ns = t1_us * 1000
                    depolar_prob, t2_ns, channel_length_km = 0, 0, 0
                
                elif noise_model == "phase_damp":
                    st.info("**Phase Damping (T2)**: Pure dephasing - coherence decay via environmental interaction")
                    t2_us = st.slider(
                        "T2 Dephasing Time (µs)", 
                        1, 500, 50, step=5,
                        help="Dephasing timescale (microseconds). Superconducting qubits: 5-500 µs. Lower T2 = faster coherence loss."
                    )
                    gate_time_ns = st.slider(
                        "Gate Time (ns)", 
                        1, 100, 50, step=5,
                        help="Gate operation time (nanoseconds). Affects how much phase noise accumulates."
                    )
                    t2_ns = t2_us * 1000
                    depolar_prob, t1_ns, channel_length_km = 0, 0, 0
                
                elif noise_model == "combined":
                    st.info("**Combined T1+T2**: Realistic model with both energy and phase relaxation")
                    t1_us = st.slider(
                        "T1 (Energy Relaxation, µs)", 
                        1, 500, 10, step=5,
                        help="T1 timescale (microseconds). Energy decay |1⟩→|0⟩."
                    )
                    t2_us = st.slider(
                        "T2 (Phase Dephasing, µs)", 
                        1, 500, 8, step=5,
                        help="T2 timescale (microseconds). Must be T2 ≤ 2×T1. Usually shorter than T1."
                    )
                    gate_time_ns = st.slider(
                        "Gate Time (ns)", 
                        1, 100, 50, step=5,
                        help="Quantum gate duration (nanoseconds)."
                    )
                    t1_ns = t1_us * 1000
                    t2_ns = t2_us * 1000
                    depolar_prob, channel_length_km = 0, 0
                
                elif noise_model == "fiber_loss":
                    st.info("**Fiber Loss**: Distance-dependent photon loss in quantum channel")
                    channel_length_km = st.slider(
                        "Channel Distance (km)", 
                        1, 500, 50, step=10,
                        help="Quantum channel length in kilometers. Longer distance = higher loss. Typical: 50-300 km."
                    )
                    depolar_prob, t1_ns, t2_ns, gate_time_ns = 0, 0, 0, 0
            else:
                noise_model = "depolarizing"
                depolar_prob = t1_ns = t2_ns = gate_time_ns = channel_length_km = 0
        
        with col3:
            st.subheader("🎲 Reproducibility & Randomness")
            st.markdown("**Control randomness for reproducibility:**")
            
            seed_type = st.radio(
                "Seed Type", 
                ["Random", "Fixed"],
                help="Random: Different results each run. Fixed: Same results for debugging/documentation."
            )
            if seed_type == "Fixed":
                seed = st.number_input(
                    "Random Seed", 
                    0, 10000, 42, step=1,
                    help="Seed for reproducible randomness [0-10000]. Same seed = same qubit/basis sequences. Default: 42."
                )
            else:
                seed = None
    
    # Run Simulation
    if st.button("▶️ Run Simulation", key="single_run", type="primary"):
        config = SimulationConfig(
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
            label=f"Custom ({n_qubits} qubits)"
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing simulation...")
        progress_bar.progress(10)
        
        with st.spinner("🔬 Running simulation..."):
            progress_bar.progress(30)
            status_text.text("📡 Quantum transmission in progress...")
            result = run_cached_simulation(config)
            progress_bar.progress(100)
            status_text.text("✅ Simulation complete!")
        
        st.success("✓ Simulation completed successfully!")
        
        # Results Display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Qubits Transmitted", result.n_transmitted)
        with col2:
            st.metric("Sifted Key Bits", result.n_sifted)
        with col3:
            st.metric("Sifted Rate", f"{result.sifted_key_rate:.1%}")
        with col4:
            st.metric("Final Key Length", result.key_length)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 QBER Analysis")
            st.write(f"**QBER**: {result.qber_result.qber:.2%}")
            st.write(f"**95% CI**: [{result.qber_result.confidence_low:.2%}, {result.qber_result.confidence_high:.2%}]")
            st.write(f"**Errors**: {result.qber_result.errors} / {result.qber_result.sample_size}")
            
            status_color = "🟢" if result.qber_result.security_status == "SECURE ✓" else "🔴"
            st.write(f"{status_color} **Status**: {result.qber_result.security_status}")
        
        with col2:
            st.subheader("🔑 Key Statistics")
            st.write(f"**Key Length**: {result.key_length} bits")
            key_rate = result.key_length / result.n_transmitted if result.n_transmitted > 0 else 0
            st.write(f"**Key Rate**: {key_rate:.3f} bits/qubit")
            st.write(f"**Agreement**: {result.key_agreement_rate:.2%}")
        
        with col3:
            st.subheader("👁️ Eve Analysis")
            if eve_present:
                st.write(f"**Intercept Rate**: {result.eve_interception_rate:.1%}")
                st.write(f"**Detection QBER**: < 4%?")
                detected = "Detected ⚠️" if result.qber_result.qber > 0.04 else "Undetected ✓"
                st.write(f"{detected}")
            else:
                st.write("Eve not present")
        
        st.markdown(f"**Runtime**: {result.runtime_seconds:.2f}s")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 - NOISE MODELS EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Phase 3 (Noise Models)":
    st.header("Phase 3: Quantum Channel Noise")
    st.markdown("Explore how different noise models affect QBER and security.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        noise_model = st.selectbox(
            "Select Noise Model",
            ["depolarizing", "amplitude_damp", "phase_damp", "combined", "fiber_loss"],
            help="Describe different physical noise mechanisms"
        )
    
    with col2:
        n_qubits = st.slider("Qubits", 200, 2000, 800, step=100)
    
    # Configure based on noise model
    if noise_model == "depolarizing":
        st.markdown("**Depolarizing Channel**: Random Pauli errors with uniform probability")
        depolar_prob = st.slider("Error Probability (p)", 0.0, 0.2, 0.05, step=0.01)
        config = SimulationConfig(
            n_qubits=n_qubits,
            noise_enabled=True,
            noise_model="depolarizing",
            depolar_prob=depolar_prob,
            label=f"Depolar p={depolar_prob}"
        )
    
    elif noise_model == "amplitude_damp":
        st.markdown("**Amplitude Damping (T1)**: Energy relaxation |1⟩→|0⟩ over time")
        t1_us = st.slider("T1 Relaxation Time (µs)", 1, 500, 10, step=5)
        gate_time_ns = st.slider("Gate Time (ns)", 1, 100, 50, step=5)
        config = SimulationConfig(
            n_qubits=n_qubits,
            noise_enabled=True,
            noise_model="amplitude_damp",
            t1_ns=t1_us * 1000,
            gate_time_ns=gate_time_ns,
            label=f"AmpDamp T1={t1_us}µs"
        )
    
    elif noise_model == "phase_damp":
        st.markdown("**Phase Damping (T2)**: Pure dephasing, coherence loss")
        t2_us = st.slider("T2 Dephasing Time (µs)", 1, 500, 50, step=5)
        gate_time_ns = st.slider("Gate Time (ns)", 1, 100, 50, step=5)
        config = SimulationConfig(
            n_qubits=n_qubits,
            noise_enabled=True,
            noise_model="phase_damp",
            t2_ns=t2_us * 1000,
            gate_time_ns=gate_time_ns,
            label=f"PhaseDamp T2={t2_us}µs"
        )
    
    elif noise_model == "combined":
        st.markdown("**Combined (T1+T2)**: Both energy and phase relaxation")
        t1_us = st.slider("T1 Relaxation (µs)", 1, 500, 10, step=5)
        t2_us = st.slider("T2 Dephasing (µs)", 1, 500, 8, step=5)
        gate_time_ns = st.slider("Gate Time (ns)", 1, 100, 50, step=5)
        config = SimulationConfig(
            n_qubits=n_qubits,
            noise_enabled=True,
            noise_model="combined",
            t1_ns=t1_us * 1000,
            t2_ns=t2_us * 1000,
            gate_time_ns=gate_time_ns,
            label=f"Combined T1={t1_us}µs T2={t2_us}µs"
        )
    
    elif noise_model == "fiber_loss":
        st.markdown("**Fiber Loss**: Distance-based photon loss in quantum channel")
        distance_km = st.slider("Distance (km)", 1, 500, 50, step=10)
        config = SimulationConfig(
            n_qubits=n_qubits,
            noise_enabled=True,
            noise_model="fiber_loss",
            channel_length_km=distance_km,
            label=f"Fiber {distance_km}km"
        )
    
    if st.button("🔬 Analyze Noise Impact", type="primary"):
        with st.spinner("Running simulation..."):
            result = run_cached_simulation(config)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("QBER", f"{result.qber_result.qber:.2%}")
        with col2:
            st.metric("Key Length", result.key_length)
        with col3:
            st.metric("Security", result.qber_result.security_status)
        
        st.info(f"💾 **Runtime**: {result.runtime_seconds:.2f} seconds")

# ══════════════════════════════════════════════════════════════════════════════
# PRESET COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Preset Comparison":
    st.header("📊 Comparison: Standard Scenarios")
    st.markdown("Run and compare multiple pre-configured scenarios.")
    
    scenario_type = st.radio(
        "Select Scenario Set",
        ["Phase 1 Only", "Phase 3 Noise Models", "All Scenarios"]
    )
    
    if scenario_type == "Phase 1 Only":
        scenarios = PHASE1_SCENARIOS
    elif scenario_type == "Phase 3 Noise Models":
        scenarios = PHASE3_SCENARIOS
    else:
        scenarios = PRESET_SCENARIOS
    
    st.markdown(f"**Running {len(scenarios)} scenarios...**")
    
    if st.button("▶️ Run All Scenarios", type="primary"):
        progress = st.progress(0)
        results_list = []
        
        for idx, (name, config) in enumerate(scenarios):
            progress.progress((idx + 1) / len(scenarios))
            result = run_cached_simulation(config)
            results_list.append((name, result))
        
        st.success(f"✓ Completed {len(scenarios)} simulations!")
        
        # Create comparison table
        comparison_data = []
        for name, result in results_list:
            comparison_data.append({
                "Scenario": name,
                "Qubits": result.n_transmitted,
                "Key Length": result.key_length,
                "Key Rate": f"{result.key_length / result.n_transmitted:.3f}",
                "QBER": f"{result.qber_result.qber:.2%}",
                "Status": result.qber_result.security_status,
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        
        for name, result in results_list:
            fig.add_trace(go.Bar(
                name=name,
                x=["Key Rate", "QBER"],
                y=[result.key_length / result.n_transmitted * 100, result.qber_result.qber * 100],
            ))
        
        fig.update_layout(
            title="Scenario Comparison: Key Rate vs QBER",
            barmode="group",
            xaxis_title="Metric",
            yaxis_title="Percentage (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECURITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Security Analysis":
    st.header("🔐 Security Analysis")
    st.markdown("Analyze security thresholds and QBER bounds.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Known QBER Thresholds")
        st.markdown("""
        - **Ideal (no noise, no Eve)**: QBER ≈ 0% → Secure
        - **With noise**: QBER ≈ 1-2% acceptable
        - **With Eve (full)**: QBER ≈ 25% → Detected
        - **Detection threshold**: QBER > 4% usually indicates attack
        """)
    
    with col2:
        st.subheader("Security Status Codes")
        st.markdown("""
        - 🟢 **SECURE ✓**: QBER below acceptable threshold
        - 🟡 **WARNING ⚠️**: QBER elevated, possible attack
        - 🔴 **ABORT ✗**: QBER too high, abort protocol
        """)
    
    st.markdown("---")
    
    st.subheader("🎯 Interactive Security Check")
    test_qber = st.slider("Test QBER Value", 0.0, 0.50, 0.02, step=0.01)
    
    if test_qber < 0.04:
        st.success(f"✓ QBER={test_qber:.2%} — Channel is SECURE")
    elif test_qber < 0.11:
        st.warning(f"⚠ QBER={test_qber:.2%} — Possible eavesdropping detected!")
    else:
        st.error(f"✗ QBER={test_qber:.2%} — Strong evidence of attack! ABORT protocol.")
