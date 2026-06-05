"""
survey.flow
══════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

The flow controller — the brain of the survey system. It owns every state
transition and decides, on each Streamlit rerun, whether the participant
should see a survey screen or be allowed into the simulator.

qkd_app.py uses exactly two things from here:

    from survey.flow import survey_gate, log_sim_activity

    gate = survey_gate()
    if gate.stop:
        st.stop()                 # a survey/admin screen was rendered
    ...
    log_sim_activity(result)      # after a successful simulation run

Forced flow:  entry → pre → activity → post → feedback → done
Identity:     an anonymous code (QKD-XXXX-XXXX) kept in the URL + session,
              so a page refresh resumes where the participant left off.
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import secrets as _secrets
from dataclasses import dataclass
from datetime import datetime, timezone

import streamlit as st

from . import db, page, sheets
from .models import Stage
from .questions import DEMOGRAPHICS, FEEDBACK, POST_SURVEY, PRE_SURVEY
from .scoring import grade, score_survey

# Code alphabet — no ambiguous characters (I, L, O, 0, 1).
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_PID_PARAM = "pid"
_SESSION_KEY = "survey_pid"


# ══════════════════════════════════════════════════════════════════════
# CONFIG / ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════

def _bootstrap_env() -> None:
    """
    Bridge Streamlit secrets → environment variables.

    On Heroku / DigitalOcean you set DATABASE_URL etc. as real env vars and
    this is a no-op. On Streamlit Community Cloud (no env vars) you put the
    same keys in .streamlit/secrets.toml and they get copied here so db.py —
    which only reads os.environ — keeps working unchanged.
    """
    keys = ("DATABASE_URL", "SURVEY_ADMIN_PASSWORD", "SURVEY_ENABLED")
    try:
        for k in keys:
            if k not in os.environ and k in st.secrets:
                os.environ[k] = str(st.secrets[k])
    except Exception:
        # No secrets file configured — perfectly fine.
        pass


def is_enabled() -> bool:
    val = os.environ.get("SURVEY_ENABLED", "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ══════════════════════════════════════════════════════════════════════
# PARTICIPANT IDENTITY
# ══════════════════════════════════════════════════════════════════════

def _new_code() -> str:
    a = "".join(_secrets.choice(_ALPHABET) for _ in range(4))
    b = "".join(_secrets.choice(_ALPHABET) for _ in range(4))
    return f"QKD-{a}-{b}"


def _resolve_pid() -> str | None:
    """Read the participant code from the URL first, then the session."""
    qp = st.query_params.get(_PID_PARAM)
    if qp:
        return qp
    return st.session_state.get(_SESSION_KEY)


def _remember_pid(pid: str) -> None:
    st.session_state[_SESSION_KEY] = pid
    if st.query_params.get(_PID_PARAM) != pid:
        st.query_params[_PID_PARAM] = pid


def _is_admin_request() -> bool:
    return str(st.query_params.get("admin", "")).strip().lower() in ("1", "true", "yes")


# ══════════════════════════════════════════════════════════════════════
# PERSISTENCE HELPERS
# ══════════════════════════════════════════════════════════════════════

def _serialize(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(v) for v in value)
    return str(value)


def _save_survey(pid: str, stage_key: str, questions, answers) -> None:
    rows = []
    for q in questions:
        val = answers.get(q.id)
        is_correct, points = grade(q, val)
        rows.append({
            "question_id": q.id,
            "question_type": q.type.value,
            "answer": _serialize(val),
            "is_correct": is_correct,
            "score": points if q.scored else None,
        })
    db.save_responses(pid, stage_key, rows)
    sheets.append_responses(pid, stage_key, rows)        # secondary mirror


def _mirror_participant(pid: str) -> None:
    """Best-effort: push the latest participant snapshot to Google Sheets."""
    p = db.get_participant(pid)
    if p:
        sheets.upsert_participant(p)


# ══════════════════════════════════════════════════════════════════════
# STAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════

def _render_entry() -> None:
    result = page.render_entry(DEMOGRAPHICS)
    if result is None:
        return
    pid = _new_code()
    while db.get_participant(pid) is not None:        # avoid the rare collision
        pid = _new_code()
    db.create_participant(
        pid, consent=result["consent"], demographics=result["demographics"]
    )
    _remember_pid(pid)
    _mirror_participant(pid)
    st.rerun()


def _render_pre(pid: str) -> None:
    answers = page.render_survey_form(
        pid=pid, current=Stage.PRE, questions=PRE_SURVEY,
        title="Pre-Survey",
        subtitle="Before you start, a few quick questions. Don't worry about getting "
                 "them right — answer as best you can.",
        submit_label="Submit & start exploring  →",
        key_prefix="pre",
    )
    if answers is None:
        return
    _save_survey(pid, "pre", PRE_SURVEY, answers)
    result = score_survey(PRE_SURVEY, answers)
    db.set_scores(pid, pre_score=result["percent"], pre_completed_at=_utcnow())
    db.update_stage(pid, Stage.ACTIVITY.value)
    _mirror_participant(pid)
    st.rerun()


def _render_post(pid: str) -> None:
    answers = page.render_survey_form(
        pid=pid, current=Stage.POST, questions=POST_SURVEY,
        title="Post-Survey",
        subtitle="Now that you've explored the simulator, please answer the same kind "
                 "of questions again.",
        submit_label="Submit & continue to feedback  →",
        key_prefix="post",
    )
    if answers is None:
        return
    _save_survey(pid, "post", POST_SURVEY, answers)
    result = score_survey(POST_SURVEY, answers)
    db.set_scores(pid, post_score=result["percent"], post_completed_at=_utcnow())
    db.update_stage(pid, Stage.FEEDBACK.value)
    _mirror_participant(pid)
    st.rerun()


def _render_feedback(pid: str) -> None:
    answers = page.render_survey_form(
        pid=pid, current=Stage.FEEDBACK, questions=FEEDBACK,
        title="Feedback",
        subtitle="Last step! Tell us about your experience using the simulator.",
        submit_label="Finish  ✓",
        key_prefix="feedback",
    )
    if answers is None:
        return
    _save_survey(pid, "feedback", FEEDBACK, answers)
    db.set_scores(pid, feedback_completed_at=_utcnow())
    db.update_stage(pid, Stage.DONE.value)
    _mirror_participant(pid)
    st.session_state["survey_just_finished"] = True
    st.rerun()


def _render_activity_banner(pid: str) -> None:
    n = db.count_activities(pid, "sim_run")
    if page.render_activity_banner(pid, n):
        db.update_stage(pid, Stage.POST.value)
        _mirror_participant(pid)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Gate:
    """Result of survey_gate(). `stop=True` → the caller should st.stop()."""
    stop: bool


def survey_gate() -> Gate:
    """
    Main entry point, called once near the top of qkd_app.py.

    Renders whatever screen the participant should see and reports back
    whether the host app should halt (a survey/admin screen was drawn) or
    proceed to render the simulator (activity / done stages).
    """
    _bootstrap_env()

    if not is_enabled():
        return Gate(stop=False)

    if _is_admin_request():
        from .admin import render_admin
        render_admin()
        return Gate(stop=True)

    pid = _resolve_pid()
    participant = db.get_participant(pid) if pid else None

    if participant is None:
        _render_entry()
        return Gate(stop=True)

    pid = participant["id"]
    _remember_pid(pid)
    stage = Stage(participant["current_stage"])

    if stage in (Stage.ENTRY, Stage.PRE):
        _render_pre(pid)
        return Gate(stop=True)

    if stage == Stage.ACTIVITY:
        _render_activity_banner(pid)
        return Gate(stop=False)

    if stage == Stage.POST:
        _render_post(pid)
        return Gate(stop=True)

    if stage == Stage.FEEDBACK:
        _render_feedback(pid)
        return Gate(stop=True)

    # Stage.DONE — celebrate once, then let them explore freely.
    if st.session_state.pop("survey_just_finished", False):
        st.balloons()
    page.render_done_banner(pid)
    return Gate(stop=False)


def log_event(event_type: str, payload: dict | None = None) -> None:
    """Resilient activity logger — never raises into the host app."""
    try:
        if not is_enabled():
            return
        pid = _resolve_pid()
        if pid and db.get_participant(pid) is not None:
            db.log_activity(pid, event_type, payload or {})
            sheets.append_activity(pid, event_type, payload or {})
    except Exception:
        pass


def log_sim_activity(result) -> None:
    """
    Log one simulation run against the current participant. Called from
    qkd_app.py right after a run succeeds. Silently does nothing if the
    survey is disabled or there is no active participant.
    """
    try:
        cfg = result.config
        qr = result.qber_result
        log_event("sim_run", {
            "label": cfg.label,
            "n_qubits": cfg.n_qubits,
            "eve_present": bool(cfg.eve_present),
            "eve_intercept_prob": float(cfg.eve_intercept_prob),
            "noise_enabled": bool(cfg.noise_enabled),
            "noise_model": cfg.noise_model if cfg.noise_enabled else None,
            "qber": float(qr.qber),
            "security_status": qr.security_status.strip(),
            "key_length": int(result.key_length),
        })
    except Exception:
        pass
