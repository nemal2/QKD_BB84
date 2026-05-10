"""
BB84 QKD Simulator - Web Interface
==================================
Interactive Streamlit UI for running and analyzing BB84 simulations.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Tuple
import plotly.graph_objects as go
import plotly.express as px

from bb84config import SimulationConfig, SimulationResult
from bb84_runner import run_simulation, run_comparison, PRESET_SCENARIOS
from bb84_plots import plot_comparison, plot_qber_vs_intercept_rate
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# CACHING & OPTIMIZATION
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
        'phase_damp_prob': config.phase_damp_prob,
        'amplitude_damp_prob': config.amplitude_damp_prob,
        'sample_fraction': config.sample_fraction,
        'seed': config.seed if config.seed is not None else 42,
        'label': config.label,
        'attack_model': config.attack_model,
    })
    return get_cached_simulation(config_dict, config.seed if config.seed is not None else 42)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BB84 QKD Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("BB84 Quantum Key Distribution Simulator")
st.markdown("Interactive interface for simulating quantum key distribution attacks and noise analysis.")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Single Simulation", "Scenario Comparison", "Eve Analysis", "About"]
)

st.sidebar.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════

if page == "Home":
    st.header("Welcome to BB84 QKD Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What is BB84?")
        st.write("""
        The **BB84 protocol** is a quantum key distribution scheme that allows two parties 
        (Alice and Bob) to establish a secret key using quantum mechanics.
        
        **Key Features:**
        - Detects eavesdropping (Eve)
        - Uses random basis (rectilinear or diagonal)
        - Quantum bit error rate (QBER) reveals attacks
        - Information-theoretically secure
        """)
    
    with col2:
        st.subheader("Quick Start")
        st.write("""
        1. **Single Simulation** - Run one BB84 protocol instance
        2. **Scenario Comparison** - Compare multiple attack scenarios
        3. **Eve Analysis** - Sweep Eve's intercept probability
        4. **Adjust Parameters** - Configure qubits, noise, attacks
        """)
    
    st.markdown("---")
    
    # EXPANDED EXPLANATIONS
    st.subheader("Core Concepts")
    
    # Qubits
    with st.expander("What is a **Qubit**?", expanded=True):
        st.write("""
        A **qubit** (quantum bit) is the fundamental unit of quantum information. Unlike classical bits (0 or 1), 
        qubits can exist in **superposition** — simultaneously 0 AND 1 until measured.
        
        **Key Properties:**
        - **Superposition**: Can be 0, 1, or both at once
        - **Measurement collapses**: When measured, it becomes definitely 0 or 1
        - **BB84 encoding**: Alice uses qubits to send random 0s and 1s using two random bases
        - **No-cloning theorem**: You cannot make a perfect copy of an unknown qubit — this is the security foundation!
        
        **In BB84:**
        - Alice prepares 600-5000 qubits (depending on simulation)
        - Each qubit encodes either a 0 or 1 using a random basis
        - Bob measures them using random bases
        - Approximately 50% of measurements use wrong basis → discarded
        - Remaining ~50% form the "sifted key"
        """)
    
    # QBER
    with st.expander("What is **QBER** (Quantum Bit Error Rate)?", expanded=False):
        st.write("""
        **QBER** measures how many bit errors appear when Alice and Bob compare their keys.
        
        **Formula:**
        ```
        QBER = (Number of Errors) / (Sample Size)
        ```
        
        **Example:**
        - Alice & Bob compare 100 bits after sifting
        - They find 3 errors
        - QBER = 3/100 = **3%**
        
        **Why Does QBER Matter?**
        
        | QBER Range | Status | Meaning |
        |-----------|--------|---------|
        | **0% – 5%** | 🟢 SECURE | Acceptable — likely just quantum noise |
        | **5% – 11%** | 🟡 WARNING | Suspicious — could be eavesdropping or high noise |
        | **11%+** | 🔴 ABORT | Stop! Eve is almost certainly eavesdropping |
        
        **How Eavesdropping Increases QBER:**
        1. Eve intercepts qubits and measures them (she guesses the basis)
        2. Eve gets wrong basis ~50% of the time → changes the qubit state
        3. Bob measures Eve's altered qubits → introduces extra errors
        4. Alice & Bob detect errors above 5% threshold → abort protocol
        
        **Statistical Confidence:**
        - QBER is estimated from a random sample (10% of key by default)
        - Streamlit reports 95% **confidence interval** to account for sampling uncertainty
        - If lower bound > 5%, definitely not secure
        """)
    
    # Eve's Attack
    with st.expander("What is **Eve's Attack**?", expanded=False):
        st.write("""
        **Eve** is an eavesdropper who tries to steal the secret key without being detected.
        
        **Eve's Intercept-Resend Attack (BB84 Phase 1):**
        
        1. **Intercept**: Eve catches qubits traveling from Alice to Bob
        2. **Measure**: Eve guesses a random basis and measures each qubit
           - 50% correct basis → Eve gets the right bit ✓
           - 50% wrong basis → Eve gets wrong bit ✗, AND changes the qubit state ⚠️
        3. **Resend**: Eve sends modified qubits to Bob
        4. **Continue**: Bob measures Eve's altered qubits
        
        **How Alice & Bob Detect Eve:**
        
        ```
        WITHOUT Eve (Ideal):
        Alice bit:  [1, 0, 1, 0, 1]
        Bob bit:    [1, 0, 1, 0, 1]
        Errors:     [0, 0, 0, 0, 0]  →  QBER = 0%  ✓ SECURE
        
        WITH Eve (100% Intercept):
        Alice bit:  [1, 0, 1, 0, 1]
        Eve basis:  [R, D, R, D, R]  (R=rectilinear, D=diagonal)
        Eve's bit:  [1, 1, 1, 1, 1]  (Eve guesses wrong ~50%)
        Bob receives altered qubits
        Bob bit:    [1, 1, 1, 0, 0]  (errors introduced!)
        Errors:     [0, 1, 0, 0, 1]  →  QBER ≈ 25% 🔴 ABORT
        ```
        
        **Eve's Intercept Probability:**
        - **0%**: Eve doesn't attack (no eavesdropping)
        - **50%**: Eve intercepts half the qubits → QBER rises ~12-13%
        - **100%**: Eve intercepts all qubits → QBER ≈ 25%
        
        **Why Can't Eve Cheat?**
        - No-cloning + measurement collapses = Eve must choose a basis
        - Wrong basis choice = Eve alters qubit state
        - Alice & Bob detect via QBER threshold
        - **Quantum mechanics enforces security!**
        
        **Improving Eve's Success (Future Phases):**
        - Direct unclonable attacks (Phase 4)
        - Trojan horse attacks (Phase 5)
        - But they all increase detectable QBER
        """)
    
    # Noise
    with st.expander("What is **Channel Noise**?", expanded=False):
        st.write("""
        **Noise** is unwanted errors caused by the physical quantum channel (not eavesdropping!).
        
        **Common Noise Sources:**
        - **Optical fiber losses** (photons absorbed)
        - **Temperature fluctuations** (phase errors)
        - **Device imperfections** (measurement errors)
        - **Stray electromagnetic fields** (bit flips)
        
        **Noise Models in Simulator:**
        
        1. **Depolarizing Noise**: Random bit-flip errors
           - Probability p: qubit flips to random state
           - Mimics fiber-optic losses
           - Increases QBER by ~2×p
        
        2. **Phase Damping Noise**: Quantum phase information lost
           - Reduces quantum coherence
           - Affects diagonal basis measurements
        
        3. **Amplitude Damping**: Energy decay (T1 relaxation)
           - Qubit loses excitation energy
           - Natural in many quantum systems
        
        **QBER vs Noise vs Eve:**
        - **QBER < 5%** = Likely just noise → Accept key
        - **QBER 5-11%** = Could be noise OR Eve → Warning zone
        - **QBER > 11%** = Likely Eve (noise alone rarely causes this) → Abort
        """)
    
    st.markdown("---")
    
    st.subheader("Protocol Flow")
    st.markdown("""
    ```
    Alice                          Bob
    ├─ Generate random bits        ├─ Generate random bases
    ├─ Choose random bases         ├─ Measure qubits
    ├─ Encode in qubits            └─ Record results
    └─ Send qubits ──────────────→ 
         [Quantum Channel]
         (Eve may intercept here)
         
    ─────────────────────────────────────
    Public Channel (Classical)
    └─ Announce bases & sift key
    └─ Estimate QBER (sample check)
    └─ Accept or abort key
    ```
    """)

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Single Simulation":
    st.header("Single Simulation Run")
    
    # Control Panel
    with st.expander("Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Protocol Parameters")
            n_qubits = st.slider(
                "Number of Qubits", 
                100, 5000, 600, step=100,
                help="Total quantum bits Alice transmits. More qubits = longer key but more simulation time"
            )
            
            # Interactive explanation for qubits
            if n_qubits < 300:
                st.info("**Low qubit count** - May not generate enough key for practical use")
            elif n_qubits > 2000:
                st.info("**High qubit count** - More qubits = more secure key, but simulation takes longer")
            else:
                st.info(f"**{n_qubits} qubits** = ~{int(n_qubits*0.25)} bits in sifted key")
            
            sample_fraction = st.slider(
                "QBER Sample Fraction", 
                0.01, 0.5, 0.1, step=0.01,
                help="What fraction of sifted key to use for error checking (sacrifice for security test)"
            )
            st.caption(f"Sample size: ~{int(n_qubits*0.25*sample_fraction)} bits for error check")
        
        with col2:
            st.subheader("Eve's Attack")
            eve_present = st.checkbox(
                "Eve Intercepts Qubits", 
                value=False,
                help="Enable eavesdropping attack on quantum channel"
            )
            
            if eve_present:
                eve_intercept_prob = st.slider(
                    "Eve's Intercept Probability",
                    0.0, 1.0, 1.0, step=0.1,
                    help="Fraction of qubits Eve tries to intercept (0%=no attack, 100%=full attack)"
                )
                
                # Interactive explanation for Eve
                if eve_intercept_prob == 0.0:
                    st.info("Eve is **disabled** - No eavesdropping")
                elif eve_intercept_prob < 0.3:
                    st.warning(f"**Light eavesdropping** ({eve_intercept_prob:.0%}) - QBER rise ~{eve_intercept_prob*7:.1f}%")
                elif eve_intercept_prob < 0.7:
                    st.warning(f"**Moderate eavesdropping** ({eve_intercept_prob:.0%}) - QBER rise ~{eve_intercept_prob*7:.1f}%")
                else:
                    st.error(f"**Heavy eavesdropping** ({eve_intercept_prob:.0%}) - QBER rise ~{eve_intercept_prob*7:.1f}%")
            else:
                eve_intercept_prob = 0.0
                st.info("Eve is **disabled** - This is the ideal baseline scenario")
        
        with col3:
            st.subheader("Channel Noise")
            noise_enabled = st.checkbox(
                "Enable Noise", 
                value=False,
                help="Add physical quantum noise (depolarizing, phase damping, etc)"
            )
            if noise_enabled:
                noise_model = st.selectbox(
                    "Noise Model",
                    ["depolarizing", "phase_damping", "amplitude_damping"],
                    help="Type of quantum noise to simulate"
                )
                depolar_prob = st.slider(
                    "Noise Probability", 
                    0.0, 0.05, 0.01, step=0.001,
                    help="Probability of noise per quantum gate (typical: 0.5-2%)"
                )
                
                # Interactive explanation for noise
                st.info(f"**{noise_model}** at {depolar_prob:.2%} → QBER rise ~{depolar_prob*2*100:.1f}%")
            else:
                noise_model = "depolarizing"
                depolar_prob = 0.0
                st.info("**Perfect quantum channel** - No noise")
        
        st.markdown("---")
        
        seed = st.number_input("Random Seed (0=random)", 0, 10000, 42)
        if seed == 0:
            seed = None
    
    # Run Simulation
    if st.button("Run Simulation", key="single_run", width="stretch"):
        config = SimulationConfig(
            n_qubits=n_qubits,
            eve_present=eve_present,
            eve_intercept_prob=eve_intercept_prob,
            noise_enabled=noise_enabled,
            noise_model=noise_model,
            depolar_prob=depolar_prob,
            sample_fraction=sample_fraction,
            seed=seed,
            label=f"Custom ({n_qubits} qubits)"
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("⏳ Initializing simulation...")
        progress_bar.progress(10)
        
        with st.spinner("🔄 Running simulation..."):
            progress_bar.progress(20)
            status_text.text("📡 Quantum transmission in progress...")
            result = run_cached_simulation(config)
            progress_bar.progress(100)
            status_text.text("✅ Simulation complete!")
        
        st.success("Simulation Complete!")
        
        # Results Display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Qubits Transmitted", result.n_transmitted)
        with col2:
            st.metric("Sifted Key Length", result.n_sifted)
        with col3:
            st.metric("Sifted Key Rate", f"{result.sifted_key_rate:.1%}")
        with col4:
            status_label = "SECURE" if "SECURE" in result.qber_result.security_status else (
                "WARNING" if "WARNING" in result.qber_result.security_status else "ABORT"
            )
            st.metric("Status", status_label)
        
        st.markdown("---")
        
        # QBER Details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("QBER Analysis")
            qber_percent = result.qber_result.qber * 100
            st.write(f"**QBER:** {qber_percent:.2f}%")
            st.write(f"**Errors Found:** {result.qber_result.errors} / {result.qber_result.sample_size}")
            st.write(f"**Status:** {result.qber_result.security_status}")
            
            # Interactive QBER explanation
            if qber_percent < 1:
                st.success("**Excellent** - Nearly no errors detected")
            elif qber_percent < 5:
                st.success("**Secure** - Below 5% threshold")
            elif qber_percent < 11:
                st.warning("**Warning** - Consider adding error correction or aborting")
            else:
                st.error(f"**Abort** - {qber_percent:.1f}% indicates eavesdropping or severe noise")
        
        with col2:
            st.subheader("Confidence Interval (95%)")
            st.write(f"**Lower:** {result.qber_result.confidence_low:.2%}")
            st.write(f"**Upper:** {result.qber_result.confidence_high:.2%}")
            st.caption("Wilson CI - statistically robust error estimation")
            
            # Confidence interval explanation
            ci_width = result.qber_result.confidence_high - result.qber_result.confidence_low
            if ci_width > 0.1:
                st.info(f"Wide CI ({ci_width:.1%}) - increase sample size for certainty")
            else:
                st.info(f"Tight CI - reliable QBER estimate")
        
        with col3:
            st.subheader("Key Statistics")
            st.write(f"**Secret Key Length:** {result.key_length} bits")
            st.write(f"**Secret Key Rate:** {result.key_generation_rate:.3f} bits/qubit")
            
            # Key rate explanation
            expected_rate = 0.25 * (1 - sample_fraction)  # Rough estimate
            actual_rate = result.key_generation_rate
            if actual_rate > 0.15:
                st.success("Good key generation rate")
            elif actual_rate > 0.05:
                st.warning("Low key rate - need more qubits")
            else:
                st.error("Insufficient key generated")
        
        # Visualization
        st.markdown("---")
        
        fig_qber = go.Figure()
        fig_qber.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=result.qber_result.qber * 100,
            title="QBER Percentage",
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 25]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 5], "color": "#2ecc71"},
                    {"range": [5, 11], "color": "#e67e22"},
                    {"range": [11, 25], "color": "#e74c3c"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 11
                }
            }
        ))
        
        st.plotly_chart(fig_qber, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Scenario Comparison":
    st.header("Scenario Comparison")
    
    st.write("Compare pre-defined scenarios to understand different attack models and noise conditions.")
    
    # Scenario Selection
    scenario_names = [name for name, _ in PRESET_SCENARIOS]
    selected_scenarios = st.multiselect(
        "Select Scenarios to Compare",
        scenario_names,
        default=scenario_names[:3]
    )
    
    if not selected_scenarios:
        st.warning("Please select at least one scenario")
    else:
        # Filter selected scenarios
        selected_configs = [
            (name, cfg) for name, cfg in PRESET_SCENARIOS 
            if name in selected_scenarios
        ]
        
        if st.button("Run Comparison", width="stretch"):
            with st.spinner("🔄 Running comparisons..."):
                results = run_comparison(selected_configs)
            
            st.success(f"Completed {len(results)} scenarios!")
            
            # Create comparison dataframe
            comparison_data = []
            for (scenario_name, _), result in zip(selected_configs, results):
                comparison_data.append({
                    "Scenario": scenario_name,
                    "Qubits": result.n_transmitted,
                    "Sifted Key": result.n_sifted,
                    "Key Rate": f"{result.sifted_key_rate:.1%}",
                    "QBER": f"{result.qber_result.qber:.2%}",
                    "Status": result.qber_result.security_status,
                    "Secret Key": result.key_length,
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, width="stretch")
            
            # Plot comparison
            with st.spinner("📊 Generating plots..."):
                fig = plt.figure(figsize=(14, 5))
                plot_comparison(selected_configs, results)
            
            st.pyplot(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# EVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Eve Analysis":
    st.header("Eve's Intercept Probability Analysis")
    
    st.write("Sweep Eve's intercept probability (0% → 100%) and observe how QBER changes.")
    st.info("""
    **What You'll See:**
    - **Blue line** = QBER vs Eve's intercept %
    - **Green dashed line** = Security threshold (5%)
    - **Red dashed line** = Abort threshold (11%)
    - As Eve intercepts more qubits, QBER rises due to state collapses
    """)
    
    with st.expander("Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            n_qubits = st.slider("Number of Qubits", 100, 5000, 1000, step=100)
            st.caption(f"{n_qubits} qubits ≈ {int(n_qubits*0.25)} sifted bits")
            
            n_points = st.slider("Number of Points to Sample", 5, 21, 11, step=1)
            st.caption(f"Will run {n_points} separate simulations")
        
        with col2:
            noise_enabled = st.checkbox("Enable Noise (Eve analysis)", value=False)
            if noise_enabled:
                depolar_prob = st.slider("Noise Probability (Eve)", 0.0, 0.05, 0.01, step=0.001)
                st.warning(f"Noise adds ~{depolar_prob*2:.1f}% to QBER baseline")
            else:
                depolar_prob = 0.0
                st.success("Clean channel for pure Eve attack analysis")
    
    if st.button("Run Eve Analysis", width="stretch"):
        with st.spinner("🔄 Sweeping Eve's intercept probability..."):
            intercept_probs = np.linspace(0, 1, n_points)
            qbers = []
            
            for prob in intercept_probs:
                config = SimulationConfig(
                    n_qubits=n_qubits,
                    eve_present=True,
                    eve_intercept_prob=prob,
                    noise_enabled=noise_enabled,
                    depolar_prob=depolar_prob,
                    seed=42
                )
                result = run_simulation(config)
                qbers.append(result.qber_result.qber * 100)
        
        st.success(f"Completed {n_points} simulations!")
        
        # Plot Eve Analysis
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=intercept_probs * 100,
            y=qbers,
            mode='lines+markers',
            name='QBER',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        
        # Add security thresholds
        fig.add_hline(y=5, line_dash="dash", line_color="#2ecc71", annotation_text="Secure (5%)")
        fig.add_hline(y=11, line_dash="dash", line_color="#e74c3c", annotation_text="Abort (11%)")
        
        fig.update_layout(
            title="QBER vs Eve's Intercept Probability",
            xaxis_title="Eve's Intercept Probability (%)",
            yaxis_title="QBER (%)",
            hovermode='x unified',
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Summary with interactive explanations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            detected_at = next((p for p, q in zip(intercept_probs, qbers) if q > 5), None)
            if detected_at:
                st.metric("Eve Detected At", f"{detected_at:.0%}")
                if detected_at < 0.3:
                    st.success("Eve easily detected with low intercept rate")
                elif detected_at < 0.7:
                    st.info("Eve must intercept significantly to hide")
                else:
                    st.warning("Eve can hide with low intercept rates")
            else:
                st.metric("Eve Detected At", "N/A")
                st.error("Eve never detected - insufficient simulation points")
        
        with col2:
            abort_at = next((p for p, q in zip(intercept_probs, qbers) if q > 11), None)
            if abort_at:
                st.metric("Protocol Aborts At", f"{abort_at:.0%}")
                st.info(f"Channel automatically rejects key above {abort_at:.0%} intercept")
            else:
                st.metric("Protocol Aborts At", "N/A")
                st.warning("No abort threshold reached - check noise settings")
        
        with col3:
            max_qber = max(qbers)
            min_qber = min(qbers)
            st.metric("Maximum QBER", f"{max_qber:.2f}%")
            st.caption(f"Range: {min_qber:.2f}% → {max_qber:.2f}%")
            
        # Interpretation
        st.markdown("---")
        st.subheader("Interpretation")
        
        if detected_at and detected_at < 0.5:
            st.success("""
            **Eve is easily detected** - Even light eavesdropping raises QBER significantly.
            This shows BB84's strength: quantum mechanics makes eavesdropping detectable!
            """)
        elif detected_at and detected_at >= 0.5:
            st.warning("""
            **Eve can partially hide** - She can intercept many qubits before detection.
            In practice, high noise might hide moderate eavesdropping.
            """)
        else:
            st.error("""
            **Simulation issue** - Check qubit count and sample size.
            Eve should always be detected (QBER ≈ 25% with full intercept).
            """)

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "About":
    st.header("About BB84 Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Protocol Overview")
        st.write("""
        **BB84** (Bennett & Brassard, 1984) is the first quantum key distribution protocol.
        
        **Security Principle:**
        - Quantum mechanics prevents perfect copying (no-cloning theorem)
        - Any measurement changes the quantum state
        - Eavesdropping introduces detectable errors
        - QBER threshold determines if Eve is present
        """)
    
    with col2:
        st.subheader("Project Phases")
        st.write("""
        - **Phase 1:** baseline BB84 + depolarizing noise
        - **Phase 3:** phase damping, amplitude damping
        - **Phase 4:** configurable attack models
        - **Phase 5:** error correction & privacy amplification
        """)
    
    st.markdown("---")
    
    st.subheader("Security Thresholds")
    
    threshold_data = {
        "Status": ["SECURE", "WARNING", "ABORT"],
        "QBER Range": ["0% – 5%", "5% – 11%", "11% +"],
        "Interpretation": [
            "No eavesdropping detected",
            "Possible attack or high noise",
            "Likely eavesdropping"
        ]
    }
    
    st.dataframe(pd.DataFrame(threshold_data), width="stretch")
    
    st.markdown("---")
    
    st.subheader("Technical Details")
    st.info("""
    **Quantum Channel Noise Models:**
    - **Depolarizing:** Probabilistic bit-flip errors
    - **Phase Damping:** Quantum phase information loss
    - **Amplitude Damping:** Energy relaxation (T1)
    
    **QBER Estimation:**
    - Uses Wilson confidence interval (95%)
    - Samples 10% of sifted key by default
    - Statistically robust error detection
    """)
    
    st.markdown("---")
    st.markdown("**Final Year Project** - Quantum Key Distribution Security Analysis")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "BB84 QKD Simulator v1.0</div>",
    unsafe_allow_html=True
)
