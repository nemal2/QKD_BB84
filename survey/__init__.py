"""
survey
══════
Integrated pre/post survey + feedback system for the BB84 QKD simulator.
University of Ruhuna — Dept. of Computer Engineering

Flow:  entry → pre-survey → use the app → post-survey → feedback → done
       (forced, gated, with progress tracked per anonymous participant)

Submodules
──────────
  models     — Stage lifecycle + Question/Option/QType schema (pure data)
  questions  — ✏️ EDIT ME: your pre/post/feedback content
  db         — SQLAlchemy storage (SQLite locally, Postgres via $DATABASE_URL)
  scoring    — grades the pre/post knowledge test, computes learning gain
  page       — Streamlit UI (entry, survey forms, progress, banners)
  flow       — the controller: gating, transitions, activity logging
  admin      — password-gated dashboard + CSV export  (/?admin=1)

Public API used by qkd_app.py:
  from survey.flow import survey_gate, log_sim_activity
"""

from .flow import Gate, log_event, log_sim_activity, survey_gate

__all__ = ["survey_gate", "log_sim_activity", "log_event", "Gate"]
