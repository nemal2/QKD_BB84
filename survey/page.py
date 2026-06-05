"""
survey.page
══════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Streamlit UI for the survey flow. These functions are *pure rendering*:
they draw a screen and RETURN what the participant submitted (or None).
All state transitions and database writes live in survey/flow.py, so the
UI stays dumb and the flow logic stays in one place.

Screens
───────
  render_entry()            consent + anonymous demographics  → dict | None
  render_survey_form(...)   a pre/post/feedback questionnaire  → dict | None
  render_activity_banner()  slim bar shown while using the app → bool (proceed?)
  render_done_banner()      slim "study complete" bar
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from .models import Question, QType, Stage

# ── palette (mirrors qkd_app.py) ──────────────────────────────────────────────
BLUE, GREEN, AMBER, GRAY = "#2563EB", "#059669", "#D97706", "#6B7280"

# Ordered steps shown in the progress indicator.
_STEPS = [
    (Stage.PRE, "Pre-Survey"),
    (Stage.ACTIVITY, "Explore"),
    (Stage.POST, "Post-Survey"),
    (Stage.FEEDBACK, "Feedback"),
]


# ══════════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    st.markdown(
        """
<style>
.sv-wrap { max-width: 760px; margin: 0 auto; }
.sv-hero h1 {
    font-family: 'Outfit', sans-serif; font-size: 30px; font-weight: 700;
    color: #111827; letter-spacing: -.02em; margin: 0 0 8px;
}
.sv-hero p { font-size: 15px; color: #6B7280; line-height: 1.7; margin: 0 0 8px; }
.sv-card {
    background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 22px 26px; margin: 18px 0;
}
.sv-code {
    font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 15px;
    color: #2563EB; background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 6px; padding: 3px 10px;
}
/* progress steps */
.sv-steps { display: flex; gap: 8px; margin: 4px 0 26px; }
.sv-step {
    flex: 1; text-align: center; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .05em; padding: 9px 6px;
    border-radius: 7px; border: 1px solid #E5E7EB; color: #9CA3AF; background: #fff;
}
.sv-step.done    { color: #059669; border-color: #BBF7D0; background: #F0FDF4; }
.sv-step.active  { color: #fff; border-color: #2563EB; background: #2563EB; }
.sv-req { color: #DC2626; }
.sv-banner {
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    border-radius: 10px; padding: 11px 18px; margin: 0 0 14px;
    font-size: 13px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_progress(current: Stage) -> None:
    """Horizontal step indicator: Pre → Explore → Post → Feedback."""
    html = ['<div class="sv-wrap"><div class="sv-steps">']
    for stage, label in _STEPS:
        if current == Stage.DONE or stage.order < current.order:
            cls = "done"
        elif stage == current:
            cls = "active"
        else:
            cls = ""
        html.append(f'<div class="sv-step {cls}">{label}</div>')
    html.append("</div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# QUESTION WIDGETS
# ══════════════════════════════════════════════════════════════════════

def _input_widget(q: Question, key_prefix: str) -> Any:
    """Render one question and return its current value."""
    key = f"{key_prefix}_{q.id}"
    label = q.prompt + (" <span class='sv-req'>*</span>" if q.required else "")
    # st.radio/text labels don't render HTML, so show the prompt ourselves.
    st.markdown(
        f"<div style='font-size:14px;font-weight:500;color:#374151;margin:2px 0 2px;'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )
    if q.help:
        st.caption(q.help)

    if q.type in (QType.SINGLE, QType.LIKERT):
        labels = {o.key: o.label for o in q.options}
        return st.radio(
            q.prompt, options=[o.key for o in q.options],
            format_func=lambda k: labels[k], index=None,
            key=key, horizontal=(q.type == QType.LIKERT),
            label_visibility="collapsed",
        )
    if q.type == QType.MULTI:
        labels = {o.key: o.label for o in q.options}
        return st.multiselect(
            q.prompt, options=[o.key for o in q.options],
            format_func=lambda k: labels[k], key=key,
            label_visibility="collapsed",
        )
    if q.type == QType.OPEN:
        return st.text_area(q.prompt, key=key, height=110, label_visibility="collapsed")
    return st.text_input(q.prompt, key=key, label_visibility="collapsed")


def _is_blank(q: Question, value: Any) -> bool:
    if q.type == QType.MULTI:
        return not value
    if q.type in (QType.OPEN, QType.SHORT):
        return not (value or "").strip()
    return value is None


# ══════════════════════════════════════════════════════════════════════
# SCREEN: ENTRY  (consent + demographics)
# ══════════════════════════════════════════════════════════════════════

def render_entry(demographics: List[Question]) -> Optional[Dict[str, Any]]:
    """
    Welcome / consent / demographics. Returns
        {"consent": True, "demographics": {...}}
    when the participant agrees and submits, else None.
    """
    inject_css()
    st.markdown(
        """
<div class="sv-wrap sv-hero">
  <h1>Educational Evaluation of the BB84 QKD Simulator</h1>
  <p>This survey is part of a research study evaluating the BB84 Quantum Key
     Distribution (QKD) Simulator as an educational tool in computing and
     engineering courses. Findings will be submitted for peer review to the ACM
     SIGCSE 2027 Technical Symposium.</p>
  <p><strong>Anonymous &amp; voluntary.</strong> You are not asked for your name,
     student/staff ID, or any identifying information. You may withdraw at any time
     without consequence. Responses are stored securely, reported only in
     de-identified, aggregated form, and retained for at least five years in line with
     institutional research-data policy.</p>
  <p style='font-size:12px;'>Questions? Contact the research team:
     nemalperera2@gmail.com · engjegant@gmail.com · chaveendias@gmail.com ·
     prabhanijayaweera@gmail.com</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sv-wrap">', unsafe_allow_html=True)
    st.markdown(
        "<div class='sv-card'><div style='font-size:13px;font-weight:600;color:#374151;"
        "margin-bottom:8px;'>By completing and submitting this survey, you confirm that:</div>"
        "<ul style='margin:0;padding-left:18px;font-size:13px;color:#374151;line-height:1.75;'>"
        "<li>I have read and understood the participant information provided above.</li>"
        "<li>I understand that my participation is voluntary and that I may withdraw at any time.</li>"
        "<li>I understand that my responses are anonymous and will be reported only in "
        "de-identified, aggregated form.</li>"
        "<li>I consent to my responses being used for research purposes and potential publication.</li>"
        "<li>I am 18 years of age or older.</li></ul></div>",
        unsafe_allow_html=True,
    )

    st.caption("Questions marked * are required.")
    with st.form("survey_entry_form"):
        st.markdown(
            "<div style='font-size:13px;font-weight:600;color:#374151;"
            "margin-bottom:6px;'>Participant background</div>",
            unsafe_allow_html=True,
        )
        demo_values: Dict[str, Any] = {}
        for q in demographics:
            demo_values[q.id] = _input_widget(q, "demo")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.divider()
        consent = st.checkbox(
            "I confirm all of the statements above and consent to participate."
        )
        submitted = st.form_submit_button("Begin the study  →", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return None
    if not consent:
        st.error("Please confirm the consent statements to begin.")
        return None
    missing = [q for q in demographics if q.required and _is_blank(q, demo_values[q.id])]
    if missing:
        st.error(
            f"Please answer the {len(missing)} required background question(s) marked *."
        )
        return None

    demographics_clean = {
        qid: val for qid, val in demo_values.items()
        if val not in (None, [], "")
    }
    return {"consent": True, "demographics": demographics_clean}


# ══════════════════════════════════════════════════════════════════════
# SCREEN: SURVEY FORM  (pre / post / feedback)
# ══════════════════════════════════════════════════════════════════════

def render_survey_form(
    *,
    pid: str,
    current: Stage,
    questions: List[Question],
    title: str,
    subtitle: str,
    submit_label: str,
    key_prefix: str,
) -> Optional[Dict[str, Any]]:
    """
    Render a questionnaire inside an st.form. Returns {question_id: answer}
    once submitted with all required questions answered, else None.
    """
    inject_css()
    render_progress(current)

    st.markdown(
        f"<div class='sv-wrap sv-hero'><h1>{title}</h1><p>{subtitle}</p>"
        f"<p style='font-size:12px;'>Participant&nbsp;code: "
        f"<span class='sv-code'>{pid}</span></p></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sv-wrap">', unsafe_allow_html=True)
    st.caption("Questions marked * are required.")
    with st.form(f"survey_form_{key_prefix}"):
        values: Dict[str, Any] = {}
        for i, q in enumerate(questions):
            values[q.id] = _input_widget(q, key_prefix)
            if i < len(questions) - 1:
                st.markdown(
                    "<div style='height:10px;border-bottom:1px solid #F3F4F6;"
                    "margin-bottom:12px;'></div>",
                    unsafe_allow_html=True,
                )
        submitted = st.form_submit_button(submit_label, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return None

    missing = [q for q in questions if q.required and _is_blank(q, values[q.id])]
    if missing:
        st.error(
            f"Please answer the {len(missing)} required question(s) before continuing."
        )
        return None
    return values


# ══════════════════════════════════════════════════════════════════════
# IN-APP BANNERS  (shown while the participant is allowed to use the app)
# ══════════════════════════════════════════════════════════════════════

def render_activity_banner(pid: str, n_activities: int) -> bool:
    """
    Slim bar shown above the simulator during the ACTIVITY stage.
    Returns True if the participant clicked 'Continue to Post-Survey'.
    """
    inject_css()
    c_msg, c_btn = st.columns([4, 1.3])
    with c_msg:
        runs = (
            f"{n_activities} simulation run(s) recorded · "
            if n_activities else ""
        )
        st.markdown(
            f"<div class='sv-banner' style='background:#EFF6FF;border:1px solid #BFDBFE;'>"
            f"<span style='font-weight:600;color:#1D4ED8;'>Explore the simulator</span>"
            f"<span style='color:#6B7280;'>{runs}When you're ready, take the "
            f"post-survey.</span>"
            f"<span style='margin-left:auto;color:#9CA3AF;'>Code "
            f"<span class='sv-code'>{pid}</span></span></div>",
            unsafe_allow_html=True,
        )
    with c_btn:
        proceed = st.button(
            "Continue to Post-Survey  →", type="primary", use_container_width=True
        )
    return proceed


def render_done_banner(pid: str) -> None:
    """Slim 'study complete' bar; the participant may keep exploring freely."""
    inject_css()
    st.markdown(
        f"<div class='sv-banner' style='background:#F0FDF4;border:1px solid #BBF7D0;'>"
        f"<span style='font-weight:600;color:#059669;'>✓ Study complete — thank you!</span>"
        f"<span style='color:#6B7280;'>All four steps are done. Feel free to keep "
        f"exploring the simulator.</span>"
        f"<span style='margin-left:auto;color:#9CA3AF;'>Code "
        f"<span class='sv-code'>{pid}</span></span></div>",
        unsafe_allow_html=True,
    )
