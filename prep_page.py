"""
prep_page.py
============
Streamlit educational module — QKD Preparation Phase.

Run standalone:
    streamlit run prep_page.py

Or call render_prep_page() from qkd_app.py to embed it as a tab.

Tabs
────
  1. Hilbert Space     — Qubit state space, superposition, inner products
  2. Density Operator  — ρ definition, properties, visualisation
  3. Types of Uncertainty — Classical ignorance vs quantum uncertainty
  4. BB84 States       — Alice's 4 encoding states with full analysis
  5. Solved Examples   — Step-by-step worked problems
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from prep_hilbert     import QubitState, CARDINAL_STATES
from prep_density     import DensityMatrix
from prep_uncertainty import (
    classical_ignorance_half,
    quantum_uncertainty_plus,
    compare_uncertainty_types,
    bb84_basis_analysis,
    analyse_ensemble,
    trace_distance,
    fidelity,
)
from prep_examples import get_all_examples
from prep_viz import (
    plot_bloch_sphere,
    plot_density_matrix,
    plot_measurement_probs,
    plot_purity_gauge,
    plot_eigenvalue_bar,
    plot_bloch_sphere_comparison,
    plot_measurement_histogram,
    plot_purity_spectrum,
)

# ── Shared CSS (small addons on top of qkd_app.py base styles) ───────────────
_CSS = """
<style>
.prep-card {
    background: #F9FAFB; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 18px 22px; margin-bottom: 14px;
}
.prep-formula {
    background: #EFF6FF; border-left: 3px solid #2563EB;
    border-radius: 0 8px 8px 0; padding: 12px 18px; margin: 10px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 14px;
}
.prep-insight {
    background: #F0FDF4; border-left: 3px solid #16A34A;
    border-radius: 0 8px 8px 0; padding: 12px 18px; margin: 10px 0;
}
.prep-warning {
    background: #FFF7ED; border-left: 3px solid #EA580C;
    border-radius: 0 8px 8px 0; padding: 12px 18px; margin: 10px 0;
}
.step-badge {
    display: inline-block; background: #2563EB; color: white;
    border-radius: 50%; width: 28px; height: 28px; line-height: 28px;
    text-align: center; font-weight: 700; font-size: 13px; margin-right: 8px;
}
.prop-row { padding: 8px 0; border-bottom: 1px solid #F3F4F6; }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HILBERT SPACE
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_hilbert_space():
    st.subheader("Qubit as a 2-Dimensional Hilbert Space")

    st.markdown("""
A **Hilbert space** is a complex vector space with an inner product — the mathematical
arena where quantum states live. For a single qubit, this space is **ℂ²**: all
two-component complex vectors.
""")

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("### State vector")
        st.latex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle")
        st.markdown("""
- **α, β** are complex amplitudes
- **|α|² + |β|²  = 1**  (normalisation)
- **|α|²** = probability of measuring 0
- **|β|²** = probability of measuring 1
""")
        st.markdown("### Bloch sphere parameterisation")
        st.latex(
            r"|\psi\rangle = \cos\!\tfrac{\theta}{2}|0\rangle"
            r" + e^{i\varphi}\sin\!\tfrac{\theta}{2}|1\rangle"
        )
        st.markdown("""
- θ ∈ [0, π] — polar angle
- φ ∈ [0, 2π) — azimuthal angle
- The **global phase** is unobservable; only relative phase matters
""")

    with col2:
        st.markdown("### Inner product")
        st.latex(r"\langle\phi|\psi\rangle = \alpha^*_\phi\,\alpha_\psi + \beta^*_\phi\,\beta_\psi \in \mathbb{C}")
        st.markdown("""
- **|⟨φ|ψ⟩|²** = probability that |ψ⟩ is found in state |φ⟩
- Orthogonal states: ⟨φ|ψ⟩ = 0 → perfectly distinguishable
""")
        st.markdown("### Basis states")
        st.latex(r"|0\rangle = \begin{pmatrix}1\\0\end{pmatrix},\quad |1\rangle = \begin{pmatrix}0\\1\end{pmatrix}")
        st.markdown("""
Any qubit state can be written as a linear combination of these two.
They are **orthonormal**: ⟨0|0⟩ = ⟨1|1⟩ = 1,  ⟨0|1⟩ = 0
""")

    st.divider()
    st.markdown("### Interactive — Prepare a qubit on the Bloch sphere")

    c1, c2, c3 = st.columns([1, 1, 1.6])
    with c1:
        theta_deg = st.slider(
            "Polar angle θ (degrees)", 0, 180, 90,
            help="θ = 0 → |0⟩ (north pole),  θ = 90° → equator,  θ = 180° → |1⟩"
        )
    with c2:
        phi_deg = st.slider(
            "Azimuthal angle φ (degrees)", 0, 360, 0,
            help="φ = 0° → |+⟩ direction,  φ = 90° → |+i⟩ direction"
        )

    theta = np.radians(theta_deg)
    phi   = np.radians(phi_deg)
    qs    = QubitState.from_bloch(theta, phi)
    dm    = DensityMatrix.from_pure(qs)
    rx, ry, rz = qs.bloch_vector()
    z0, z1 = qs.measure_prob("z")
    x0, x1 = qs.measure_prob("x")
    y0, y1 = qs.measure_prob("y")

    with c3:
        st.markdown("**Current state:**")
        st.latex(
            rf"|\psi\rangle = {qs.alpha.real:.4f}|0\rangle "
            rf"+ ({qs.beta.real:.4f} + {qs.beta.imag:.4f}i)|1\rangle"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("P(0)  Z-basis", f"{z0:.3f}")
        m2.metric("P(+)  X-basis", f"{x0:.3f}")
        m3.metric("P(+i) Y-basis", f"{y0:.3f}")

    # Bloch sphere plot
    named_states = list(CARDINAL_STATES.values())
    named_labels = list(CARDINAL_STATES.keys())
    fig = plot_bloch_sphere(
        [qs] + named_states,
        ["Custom |ψ⟩"] + named_labels,
        title=f"Custom qubit  θ={theta_deg}°, φ={phi_deg}°",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Orthogonality of BB84 basis pairs")

    pairs = [
        (QubitState.ket0(), QubitState.ket1(), "|0⟩  and  |1⟩"),
        (QubitState.plus(), QubitState.minus(), "|+⟩  and  |−⟩"),
    ]
    cols = st.columns(len(pairs))
    for col, (s1, s2, name) in zip(cols, pairs):
        ip = s1.inner_product(s2)
        col.markdown(f"**{name}**")
        col.latex(rf"\langle\psi_1|\psi_2\rangle = {ip.real:.4f} + {ip.imag:.4f}i")
        col.success("Orthogonal ✓" if abs(ip) < 1e-9 else f"|⟨·|·⟩|² = {abs(ip)**2:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DENSITY OPERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_density_operator():
    st.subheader("Density Operator  ρ")

    st.markdown("""
The density operator is the most general description of a quantum state.
It handles both **pure states** (full quantum knowledge) and **mixed states**
(quantum + classical uncertainty).
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Pure state")
        st.latex(r"\rho = |\psi\rangle\langle\psi|")
        st.markdown("- Tr(ρ²) = 1\n- Bloch vector on sphere surface\n- Non-zero off-diagonal elements (coherences)")
    with c2:
        st.markdown("#### Mixed state (ensemble)")
        st.latex(r"\rho = \sum_i p_i\,|\psi_i\rangle\langle\psi_i|")
        st.markdown("- Tr(ρ²) < 1\n- Bloch vector inside sphere\n- Coherences partially or fully washed out")

    st.divider()
    st.markdown("### Four defining properties")

    props_md = [
        ("Hermitian",         r"\rho = \rho^\dagger",
         "Self-adjoint. All eigenvalues are real."),
        ("Positive semi-def", r"\langle\psi|\rho|\psi\rangle \geq 0",
         "All eigenvalues ≥ 0. Probabilities are non-negative."),
        ("Unit trace",        r"\text{Tr}(\rho) = 1",
         "Total probability is 1. Normalisation."),
        ("Purity bound",      r"\tfrac{1}{d} \leq \text{Tr}(\rho^2) \leq 1",
         "1 = pure, 1/2 = maximally mixed (qubit). d = dimension."),
    ]
    for name, formula, meaning in props_md:
        with st.expander(f"**{name}**", expanded=False):
            st.latex(formula)
            st.markdown(meaning)

    st.divider()
    st.markdown("### Interactive — Explore with custom state")

    mode = st.radio(
        "State type", ["Pure state (θ, φ sliders)", "Classical ensemble (mix |0⟩ and |1⟩)"],
        horizontal=True,
    )

    if mode.startswith("Pure"):
        c1, c2 = st.columns(2)
        theta_deg = c1.slider("θ (degrees)", 0, 180, 45, key="dm_theta")
        phi_deg   = c2.slider("φ (degrees)", 0, 360, 0,  key="dm_phi")
        theta = np.radians(theta_deg)
        phi   = np.radians(phi_deg)
        qs = QubitState.from_bloch(theta, phi)
        dm = DensityMatrix.from_pure(qs)
        description = f"|ψ⟩ with θ={theta_deg}°, φ={phi_deg}°"
    else:
        p0 = st.slider("Weight of |0⟩ in ensemble", 0.0, 1.0, 0.5, 0.01, key="dm_p0")
        p1 = 1.0 - p0
        st.caption(f"Weight of |1⟩ = {p1:.2f}")
        dm = DensityMatrix.from_ensemble(
            [QubitState.ket0(), QubitState.ket1()], [p0, p1]
        )
        description = f"Ensemble: {p0:.2f}|0⟩⟨0| + {p1:.2f}|1⟩⟨1|"

    props = dm.verify_properties()
    rx, ry, rz = dm.bloch_vector()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Purity  Tr(ρ²)", f"{dm.purity():.4f}")
    c2.metric("Trace  Tr(ρ)",   f"{dm.trace():.4f}")
    c3.metric("Bloch radius |r|", f"{dm.bloch_radius():.4f}")
    c4.metric("Coherence |ρ₀₁|", f"{dm.coherence():.4f}")

    col_sphere, col_dm = st.columns([1.1, 0.9])
    with col_sphere:
        fig_bs = plot_bloch_sphere([dm], [description], title="Bloch Sphere")
        st.plotly_chart(fig_bs, use_container_width=True)
    with col_dm:
        fig_dm = plot_density_matrix(dm, title=f"Density Matrix — {description}")
        st.plotly_chart(fig_dm, use_container_width=True)
        fig_eig = plot_eigenvalue_bar(dm, title="Eigenvalues")
        st.plotly_chart(fig_eig, use_container_width=True)

    col_g, col_mp = st.columns(2)
    with col_g:
        st.plotly_chart(plot_purity_gauge(dm), use_container_width=True)
    with col_mp:
        st.plotly_chart(
            plot_measurement_probs(
                [{"label": description, "z_basis": dict(zip(["P(0)","P(1)"], dm.measure_prob("z"))),
                  "x_basis": dict(zip(["P(+)","P(-)"], dm.measure_prob("x")))}],
                title="Measurement Probabilities",
            ),
            use_container_width=True,
        )

    # Properties check table
    st.markdown("#### Properties verification")
    cols = st.columns(4)
    check_items = [
        ("Hermitian",   props["hermitian"]["passed"],     "ρ = ρ†"),
        ("Unit trace",  props["unit_trace"]["passed"],    f"Tr = {props['unit_trace']['value']:.6f}"),
        ("PSD",         props["positive_semidefinite"]["passed"],
                        f"λ = {[round(e,4) for e in props['positive_semidefinite']['eigenvalues']]}"),
        ("Pure state",  props["purity"]["is_pure"],       f"Tr(ρ²) = {props['purity']['value']:.4f}"),
    ]
    for col, (name, ok, detail) in zip(cols, check_items):
        icon = "✅" if ok else "ℹ️"
        col.markdown(f"{icon} **{name}**")
        col.caption(detail)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TYPES OF UNCERTAINTY
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_uncertainty():
    st.subheader("Two Kinds of Not Knowing")

    st.markdown("""
Quantum physics introduces a distinction that does not exist in classical probability:

| | **Classical Ignorance** | **Quantum Uncertainty** |
|---|---|---|
| Cause | Missing information | No fact of the matter |
| State type | Mixed (ρ² ≠ ρ) | Pure (ρ² = ρ) |
| Purity | < 1 | = 1 |
| Coherence | Zero (in some basis) | Non-zero |
| Can you distinguish? | Only via complementary basis | Yes |
""")

    comparison = compare_uncertainty_types()
    classical  = comparison["classical"]
    quantum    = comparison["quantum"]

    st.divider()
    st.markdown("### Side-by-side: same Z-basis, different X-basis")

    fig_cmp = plot_bloch_sphere_comparison(
        [(classical["density_matrix"], classical["name"]),
         (quantum["density_matrix"],   quantum["name"])],
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    col1, col2 = st.columns(2)
    for col, data in zip([col1, col2], [classical, quantum]):
        with col:
            st.markdown(f"#### {data['name']}")
            st.caption(data["subtitle"])
            st.markdown(f"*{data['story']}*")

            dm = data["density_matrix"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Purity",       f"{data['purity']:.4f}")
            m2.metric("Bloch |r|",    f"{data['bloch_radius']:.4f}")
            m3.metric("Coherence",    f"{dm.coherence():.4f}")

            st.plotly_chart(
                plot_density_matrix(dm, title=f"ρ — {data['name']}"),
                use_container_width=True
            )
            st.markdown(f"**Z-basis:** P(0) = {data['z_basis']['P(0)']:.3f},  P(1) = {data['z_basis']['P(1)']:.3f}")
            st.markdown(f"**X-basis:** P(+) = {data['x_basis']['P(+)']:.3f},  P(−) = {data['x_basis']['P(-)']:.3f}")
            st.info(data["matrix_note"])

    z_same = comparison["z_basis_indistinguishable"]
    x_diff = comparison["x_basis_distinguishable"]

    st.divider()
    if z_same:
        st.success("**Z-basis:** Both states give P(0) = P(1) = 0.5 — indistinguishable here.")
    if x_diff:
        st.error(
            "**X-basis:** |+⟩ gives P(+) = 1.0 while mixed state gives P(+) = 0.5. "
            "The difference is measurable!"
        )
    st.info(comparison["key_insight"])

    st.divider()
    st.markdown("### Quantitative distinguishability")

    cl_dm = classical["density_matrix"]
    qu_dm = quantum["density_matrix"]
    td    = trace_distance(cl_dm, qu_dm)
    fid   = fidelity(cl_dm, qu_dm)

    c1, c2 = st.columns(2)
    c1.metric(
        "Trace distance  D(ρ, σ)",
        f"{td:.4f}",
        help="Max probability advantage in distinguishing the two states. 0 = identical, 1 = perfect.",
    )
    c2.metric(
        "Fidelity  F(ρ, σ)",
        f"{fid:.4f}",
        help="Closeness measure. 1 = identical, 0 = orthogonal.",
    )

    st.divider()
    st.markdown("### Interactive measurement simulation")

    c1, c2, c3 = st.columns(3)
    n_shots = c1.selectbox("Number of shots", [100, 500, 1000, 5000], index=1)
    basis   = c2.selectbox("Measurement basis", ["z", "x", "y"], index=0)
    seed_v  = c3.number_input("Random seed", 0, 9999, 42)

    if st.button("▶ Simulate measurements on both states"):
        left, right = st.columns(2)
        for col, dm, name in [
            (left,  cl_dm, "Classical Ignorance"),
            (right, qu_dm, "Quantum Uncertainty  |+⟩"),
        ]:
            shots = dm.simulate_measurements(n_shots, basis, seed=seed_v)
            fig   = plot_measurement_histogram(
                shots, basis, title=f"{name}  ({basis.upper()}-basis)"
            )
            col.plotly_chart(fig, use_container_width=True)
            p0, p1 = dm.measure_prob(basis)
            col.caption(
                f"Theoretical: P(outcome 0) = {p0:.3f},  P(outcome 1) = {p1:.3f}"
            )

    st.divider()
    st.markdown("### Build your own ensemble")
    st.caption("Mix any two BB84 states with adjustable weights")

    state_options = {"|0⟩": QubitState.ket0(), "|1⟩": QubitState.ket1(),
                     "|+⟩": QubitState.plus(), "|−⟩": QubitState.minus()}
    c1, c2, c3 = st.columns(3)
    s1_name = c1.selectbox("State 1", list(state_options.keys()), index=0)
    s2_name = c2.selectbox("State 2", list(state_options.keys()), index=1)
    p1_val  = c3.slider(f"Weight of {s1_name}", 0.0, 1.0, 0.5, 0.01)

    custom = analyse_ensemble(
        [state_options[s1_name], state_options[s2_name]],
        [p1_val, 1 - p1_val],
        name=f"{p1_val:.2f}×{s1_name} + {1-p1_val:.2f}×{s2_name}",
    )
    cdm = custom["density_matrix"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Purity",       f"{custom['purity']:.4f}")
    m2.metric("Bloch |r|",    f"{custom['bloch_radius']:.4f}")
    m3.metric("Coherence",    f"{custom['coherence']:.4f}")

    col_bs, col_dm2 = st.columns(2)
    with col_bs:
        st.plotly_chart(
            plot_bloch_sphere([cdm], [custom["name"]], title="Custom ensemble"),
            use_container_width=True,
        )
    with col_dm2:
        st.plotly_chart(
            plot_density_matrix(cdm, title="Custom ρ"),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BB84 PREPARATION STATES
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_bb84_states():
    st.subheader("Alice's Four BB84 Preparation States")

    st.markdown("""
Alice randomly picks **bit ∈ {0, 1}** and **basis ∈ {Rectilinear (+), Diagonal (×)}**,
then prepares the corresponding qubit and transmits it to Bob.
""")

    encoding_table = {
        "Bit": [0, 1, 0, 1],
        "Basis": ["Rectilinear (+)", "Rectilinear (+)", "Diagonal (×)", "Diagonal (×)"],
        "State": ["|0⟩", "|1⟩", "|+⟩", "|−⟩"],
        "Bloch position": ["North pole (+Z)", "South pole (−Z)", "Equator (+X)", "Equator (−X)"],
    }
    import pandas as pd
    st.table(pd.DataFrame(encoding_table))

    # Build state data
    configs = [
        (0, 0, "|0⟩"), (1, 0, "|1⟩"), (0, 1, "|+⟩"), (1, 1, "|−⟩"),
    ]
    states_data = []
    for bit, basis, label in configs:
        qs  = QubitState.from_bb84(bit, basis)
        dm  = DensityMatrix.from_pure(qs)
        z0, z1 = dm.measure_prob("z")
        x0, x1 = dm.measure_prob("x")
        states_data.append({
            "label": label, "state": qs, "density_matrix": dm,
            "bloch_vector": qs.bloch_vector(),
            "z_basis": {"P(0)": z0, "P(1)": z1},
            "x_basis": {"P(+)": x0, "P(-)": x1},
        })

    # Bloch sphere with all 4 states
    fig_all = plot_bloch_sphere(
        [d["state"] for d in states_data],
        [d["label"] for d in states_data],
        title="All Four BB84 States on the Bloch Sphere",
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # Measurement probability comparison
    fig_probs = plot_measurement_probs(
        states_data,
        title="Measurement Probabilities for Each BB84 State",
    )
    st.plotly_chart(fig_probs, use_container_width=True)

    st.divider()
    st.markdown("### Individual state deep-dive")

    selected = st.selectbox("Select a state", [d["label"] for d in states_data])
    d = next(x for x in states_data if x["label"] == selected)
    dm = d["density_matrix"]
    qs = d["state"]

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Purity", f"{dm.purity():.4f}")
    mc2.metric("Bloch radius", f"{dm.bloch_radius():.4f}")
    mc3.metric("P(0) Z-basis", f"{d['z_basis']['P(0)']:.3f}")
    mc4.metric("P(+) X-basis", f"{d['x_basis']['P(+)']:.3f}")

    col_bs2, col_dm2 = st.columns([1.1, 0.9])
    with col_bs2:
        fig_s = plot_bloch_sphere([qs], [selected], title=f"Bloch sphere — {selected}")
        st.plotly_chart(fig_s, use_container_width=True)
    with col_dm2:
        fig_dm2 = plot_density_matrix(dm, title=f"Density matrix — {selected}")
        st.plotly_chart(fig_dm2, use_container_width=True)

    st.divider()
    st.markdown("### Basis mismatch — what happens when Bob measures in the wrong basis?")

    analysis = bb84_basis_analysis()
    for info in analysis:
        if info["state"] != selected:
            continue
        c1, c2 = st.columns(2)
        c1.markdown(f"**Correct basis ({info['correct_basis']}):**")
        c1.markdown(f"P(0) = {info['correct_probs']['P(0)']:.3f},  P(1) = {info['correct_probs']['P(1)']:.3f}")
        c1.success("Deterministic outcome — no ambiguity")

        c2.markdown(f"**Wrong basis ({info['wrong_basis']}):**")
        c2.markdown(f"P(0) = {info['wrong_probs']['P(0)']:.3f},  P(1) = {info['wrong_probs']['P(1)']:.3f}")
        if info["wrong_is_random"]:
            c2.error("Completely random outcome — basis mismatch destroys information")
        else:
            c2.warning("Partially informative (non-orthogonal basis)")

    st.markdown('<div class="prep-insight">💡 <b>Key insight:</b> Measuring in the wrong basis gives a random result and destroys the original state. Eve cannot avoid this — she must guess the basis, and any wrong guess introduces detectable errors into the sifted key.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Mutually unbiased bases (MUB)")

    st.markdown("""
The two BB84 bases are **mutually unbiased**:

> Knowing the state perfectly in one basis gives *zero* information about the outcome in the other.
""")
    st.latex(
        r"|\langle 0|+\rangle|^2 = |\langle 0|-\rangle|^2 = |\langle 1|+\rangle|^2 = |\langle 1|-\rangle|^2 = \tfrac{1}{2}"
    )
    st.markdown("""
This means any state from the rectilinear basis is completely undetermined in the diagonal basis and vice versa.
It is exactly what makes BB84 secure — an eavesdropper guessing the wrong basis gains no advantage.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SOLVED EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_solved_examples():
    st.subheader("Solved Worked Examples")

    all_examples = get_all_examples()
    titles = [ex["title"] for ex in all_examples]
    choice = st.selectbox("Choose an example", titles)
    ex = next(e for e in all_examples if e["title"] == choice)

    st.markdown(f"**{ex.get('symbol', '')}**")
    st.divider()

    # ── Example 1, 2, 4: step-based ──────────────────────────────────────────
    if "steps" in ex:
        for step in ex["steps"]:
            with st.container():
                st.markdown(
                    f'<div class="prep-card">'
                    f'<span class="step-badge">{step["step"]}</span>'
                    f'<b>{step["heading"]}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns([1, 1])
                with c1:
                    try:
                        st.latex(step["math"])
                    except Exception:
                        st.code(step["math"])
                    st.markdown(f'<div class="prep-formula">{step["result"]}</div>',
                                unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="prep-insight">{step["explanation"]}</div>',
                                unsafe_allow_html=True)

        st.divider()
        st.markdown(f'<div class="prep-insight"><b>Key takeaway:</b> {ex["key_takeaway"]}</div>',
                    unsafe_allow_html=True)

        # Visualisations for examples 1 and 2
        if "state" in ex:
            qs = ex["state"]
            dm = ex["density_matrix"]
            st.markdown("#### Visualisations")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(
                    plot_bloch_sphere([qs], [ex.get("symbol","")], title="Bloch sphere"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    plot_density_matrix(dm, title="Density matrix  ρ"),
                    use_container_width=True,
                )
            with c3:
                st.plotly_chart(
                    plot_purity_gauge(dm),
                    use_container_width=True,
                )

        elif "density_matrix" in ex:
            dm = ex["density_matrix"]
            st.markdown("#### Visualisations")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(
                    plot_bloch_sphere([dm], [ex.get("symbol","")], title="Bloch sphere"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    plot_density_matrix(dm, title="Density matrix  ρ"),
                    use_container_width=True,
                )
            with c3:
                st.plotly_chart(
                    plot_purity_gauge(dm),
                    use_container_width=True,
                )

    # ── Example 3: BB84 states ────────────────────────────────────────────────
    elif "states" in ex:
        st.markdown("#### All four BB84 preparation states at a glance")
        states_data = ex["states"]

        # Bloch sphere
        fig_all = plot_bloch_sphere(
            [d["state"] for d in states_data],
            [d["label"] for d in states_data],
            title="All Four BB84 States",
        )
        st.plotly_chart(fig_all, use_container_width=True)

        # Density matrices in a row
        cols = st.columns(4)
        for col, d in zip(cols, states_data):
            with col:
                st.markdown(f"**{d['label']}**")
                st.caption(d["basis_name"])
                fig_dm = plot_density_matrix(d["density_matrix"], title=d["label"])
                st.plotly_chart(fig_dm, use_container_width=True)
                col.metric("Purity", f"{d['purity']:.1f}")
                col.metric("Coherence", f"{d['coherence']:.3f}")

        # Purity spectrum
        fig_pur = plot_purity_spectrum(
            [d["density_matrix"] for d in states_data],
            [d["label"] for d in states_data],
            title="Purity of BB84 States",
        )
        st.plotly_chart(fig_pur, use_container_width=True)

        st.divider()
        st.markdown("#### Key insights")
        for insight in ex["key_insights"]:
            st.markdown(f"✅ {insight}")

        st.divider()
        st.markdown(
            f'<div class="prep-insight"><b>Key takeaway:</b> {ex["key_takeaway"]}</div>',
            unsafe_allow_html=True,
        )

    # ── Example 4: properties verification ──────────────────────────────────
    elif "properties" in ex:
        import pandas as pd

        st.markdown("#### States under comparison")
        c1, c2 = st.columns(2)
        for col, dm, lbl in [
            (c1, ex["pure_dm"],  ex["pure_label"]),
            (c2, ex["mixed_dm"], ex["mixed_label"]),
        ]:
            with col:
                st.markdown(f"**{lbl}**")
                st.plotly_chart(
                    plot_density_matrix(dm, title=f"ρ — {lbl}"),
                    use_container_width=True,
                )
                st.plotly_chart(plot_purity_gauge(dm, label=lbl), use_container_width=True)

        st.markdown("#### Property table")
        rows = []
        for prop in ex["properties"]:
            rows.append({
                "Property":     prop["name"],
                "Formula":      prop["symbol"],
                "Pure state":   prop["pure"],
                "Mixed state":  prop["mixed"],
                "Meaning":      prop["meaning"],
            })
        st.dataframe(pd.DataFrame(rows).drop(columns=["Formula"]), use_container_width=True)

        st.divider()
        for prop in ex["properties"]:
            with st.expander(f"📐 {prop['name']}", expanded=False):
                st.latex(prop["symbol"])
                st.markdown(prop["meaning"])
                mc1, mc2 = st.columns(2)
                mc1.info(f"**Pure state:** {prop['pure']}")
                mc2.info(f"**Mixed state:** {prop['mixed']}")

        st.divider()
        st.markdown(
            f'<div class="prep-insight"><b>Key takeaway:</b> {ex["key_takeaway"]}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def render_prep_page():
    """
    Main render function.
    Call this from qkd_app.py to embed the preparation module as a page/tab,
    or run prep_page.py directly for standalone mode.
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    st.title("QKD Preparation Phase — Educational Module")
    st.markdown(
        "Explore the quantum mechanics behind qubit state preparation in BB84. "
        "All theory, visualisations, and simulations are for the **preparation stage only**."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Hilbert Space",
        "🔢 Density Operator",
        "❓ Types of Uncertainty",
        "📡 BB84 States",
        "📖 Solved Examples",
    ])

    with tab1:
        _tab_hilbert_space()
    with tab2:
        _tab_density_operator()
    with tab3:
        _tab_uncertainty()
    with tab4:
        _tab_bb84_states()
    with tab5:
        _tab_solved_examples()


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="QKD Preparation Phase",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_prep_page()
