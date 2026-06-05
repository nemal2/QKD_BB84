"""
survey.db
══════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Storage layer for the survey system, built on SQLAlchemy Core so the
*same code* runs on two backends:

  • Local / development  →  SQLite file  (survey_data/qkd_survey.db)
  • Production (Heroku /
    DigitalOcean / etc.) →  Postgres, auto-detected from $DATABASE_URL

WHY THIS MATTERS
────────────────
Heroku dynos and DigitalOcean's App Platform have an *ephemeral*
filesystem — anything written to disk is wiped on every restart/redeploy.
A bare SQLite file would silently lose all your survey data. So on those
platforms you provision a **managed Postgres** add-on; it exposes a
`DATABASE_URL` env var, which this module picks up automatically. No code
change needed — develop on SQLite, deploy on Postgres.

The module exposes a small repository API (create_participant, get_participant,
update_stage, save_responses, set_scores, log_activity, …) plus bulk readers
for the admin dashboard / CSV export. No Streamlit imports here — keep it
testable.
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import functools
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData,
    String, Table, Text, UniqueConstraint, create_engine, delete, func,
    insert, select, update,
)
from sqlalchemy.engine import Engine


# ══════════════════════════════════════════════════════════════════════
# ENGINE  —  SQLite locally, Postgres in production
# ══════════════════════════════════════════════════════════════════════

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _database_url() -> str:
    """
    Resolve the SQLAlchemy URL.

    Priority:
      1. $DATABASE_URL  (managed Postgres on Heroku / DigitalOcean / etc.)
      2. SQLite file at $SURVEY_DB_PATH  (default: survey_data/qkd_survey.db)
    """
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        # Heroku & some providers hand out the legacy "postgres://" scheme,
        # which SQLAlchemy 1.4+ rejects. Normalise to the psycopg2 driver.
        if url.startswith("postgres://"):
            url = "postgresql+psycopg2://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://"):]
        return url

    path = os.environ.get("SURVEY_DB_PATH", os.path.join("survey_data", "qkd_survey.db"))
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return f"sqlite:///{path}"


@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide singleton engine. Creates tables on first call."""
    url = _database_url()
    kwargs: Dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # Streamlit serves reruns on worker threads; allow cross-thread use.
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    metadata.create_all(engine)
    return engine


def backend_name() -> str:
    """'sqlite' or 'postgresql' — handy for the admin diagnostics panel."""
    return get_engine().dialect.name


# ══════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════

metadata = MetaData()

participants = Table(
    "participants", metadata,
    Column("id", String(32), primary_key=True),          # the anonymous auto-code
    Column("created_at", DateTime, default=_utcnow),
    Column("updated_at", DateTime, default=_utcnow, onupdate=_utcnow),
    Column("current_stage", String(32), nullable=False, default="entry"),
    Column("consent", Boolean, default=False),
    Column("demographics", JSON, default=dict),
    Column("pre_score", Float),                           # percent 0–100
    Column("post_score", Float),                          # percent 0–100
    Column("pre_completed_at", DateTime),
    Column("post_completed_at", DateTime),
    Column("feedback_completed_at", DateTime),
)

responses = Table(
    "responses", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("participant_id", String(32), ForeignKey("participants.id"), index=True),
    Column("stage", String(16)),                         # 'pre' | 'post' | 'feedback'
    Column("question_id", String(64)),
    Column("question_type", String(32)),
    Column("answer", Text),                              # raw answer (option key, text, "a|c", …)
    Column("is_correct", Boolean),                       # None for unscored questions
    Column("score", Float),                              # points earned
    Column("created_at", DateTime, default=_utcnow),
    UniqueConstraint("participant_id", "stage", "question_id", name="uq_response"),
)

activity_events = Table(
    "activity_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("participant_id", String(32), ForeignKey("participants.id"), index=True),
    Column("event_type", String(32)),                    # 'sim_run' | 'sweep_run' | 'page_view' | …
    Column("payload", JSON),
    Column("created_at", DateTime, default=_utcnow),
)


# ══════════════════════════════════════════════════════════════════════
# PARTICIPANTS
# ══════════════════════════════════════════════════════════════════════

def _row_to_dict(row) -> Optional[Dict[str, Any]]:
    return dict(row._mapping) if row is not None else None


def create_participant(pid: str, consent: bool,
                       demographics: Optional[Dict[str, Any]] = None) -> None:
    """Insert a new participant, already advanced to the PRE stage."""
    now = _utcnow()
    with get_engine().begin() as conn:
        conn.execute(insert(participants).values(
            id=pid,
            created_at=now,
            updated_at=now,
            current_stage="pre_survey",
            consent=consent,
            demographics=demographics or {},
        ))


def get_participant(pid: str) -> Optional[Dict[str, Any]]:
    if not pid:
        return None
    with get_engine().connect() as conn:
        row = conn.execute(
            select(participants).where(participants.c.id == pid)
        ).fetchone()
    return _row_to_dict(row)


def update_stage(pid: str, stage: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            update(participants)
            .where(participants.c.id == pid)
            .values(current_stage=stage, updated_at=_utcnow())
        )


def set_scores(pid: str, **fields: Any) -> None:
    """
    Update score / completion-timestamp columns, e.g.
        set_scores(pid, pre_score=80.0, pre_completed_at=datetime.utcnow())
    """
    if not fields:
        return
    fields["updated_at"] = _utcnow()
    with get_engine().begin() as conn:
        conn.execute(
            update(participants).where(participants.c.id == pid).values(**fields)
        )


# ══════════════════════════════════════════════════════════════════════
# RESPONSES
# ══════════════════════════════════════════════════════════════════════

def save_responses(pid: str, stage_key: str, rows: List[Dict[str, Any]]) -> None:
    """
    Replace all responses for (participant, stage) with `rows`.

    A whole survey is submitted at once, so we delete any previous answers for
    this stage first (makes re-submission idempotent) then insert fresh.

    Each row: {question_id, question_type, answer, is_correct, score}
    """
    now = _utcnow()
    with get_engine().begin() as conn:
        conn.execute(
            delete(responses).where(
                (responses.c.participant_id == pid) & (responses.c.stage == stage_key)
            )
        )
        if rows:
            conn.execute(insert(responses), [
                {
                    "participant_id": pid,
                    "stage": stage_key,
                    "question_id": r["question_id"],
                    "question_type": r["question_type"],
                    "answer": r["answer"],
                    "is_correct": r.get("is_correct"),
                    "score": r.get("score"),
                    "created_at": now,
                }
                for r in rows
            ])


# ══════════════════════════════════════════════════════════════════════
# ACTIVITY
# ══════════════════════════════════════════════════════════════════════

def log_activity(pid: str, event_type: str,
                 payload: Optional[Dict[str, Any]] = None) -> None:
    with get_engine().begin() as conn:
        conn.execute(insert(activity_events).values(
            participant_id=pid,
            event_type=event_type,
            payload=payload or {},
            created_at=_utcnow(),
        ))


def count_activities(pid: str, event_type: Optional[str] = None) -> int:
    stmt = select(func.count()).select_from(activity_events).where(
        activity_events.c.participant_id == pid
    )
    if event_type:
        stmt = stmt.where(activity_events.c.event_type == event_type)
    with get_engine().connect() as conn:
        return int(conn.execute(stmt).scalar() or 0)


# ══════════════════════════════════════════════════════════════════════
# BULK READERS  (admin dashboard / CSV export)
# ══════════════════════════════════════════════════════════════════════

def all_participants() -> List[Dict[str, Any]]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(participants).order_by(participants.c.created_at)
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def all_responses() -> List[Dict[str, Any]]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(responses).order_by(
                responses.c.participant_id, responses.c.stage, responses.c.question_id
            )
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def all_activities() -> List[Dict[str, Any]]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(activity_events).order_by(activity_events.c.created_at)
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def stage_counts() -> Dict[str, int]:
    """How many participants are currently sitting in each stage."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(participants.c.current_stage, func.count())
            .group_by(participants.c.current_stage)
        ).fetchall()
    return {stage: int(n) for stage, n in rows}


def activity_counts_by_participant() -> Dict[str, int]:
    """{participant_id: number of logged activity events}."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(activity_events.c.participant_id, func.count())
            .group_by(activity_events.c.participant_id)
        ).fetchall()
    return {pid: int(n) for pid, n in rows}
