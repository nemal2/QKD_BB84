"""
survey.sheets
══════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

OPTIONAL secondary store: mirror survey data into a Google Sheet.

The SQL database (survey/db.py) is always the source of truth. This module
is a best-effort *mirror* on top of it — handy as a live, shareable backup
your supervisor can open in a browser, and as a second copy independent of
the app's database.

  • Live mirror   — survey responses and activity events are appended to the
                    sheet as they happen (append-only, never blocks the app).
  • Full sync     — the admin dashboard's "Sync all to Google Sheets" button
                    rebuilds every tab from the database (authoritative copy).

If it isn't configured, every function here is a silent no-op, so the app
runs exactly as before.

SETUP (see DEPLOYMENT.md)
─────────────────────────
  1. Create a Google Cloud service account, download its JSON key.
  2. Create a Google Sheet and share it with the service account's email
     (Editor).
  3. Provide credentials + the sheet id via Streamlit secrets:

        gsheets_id = "….spreadsheet id or full URL…"
        [gcp_service_account]
        type = "service_account"
        project_id = "…"
        private_key = "-----BEGIN PRIVATE KEY-----\n…\n-----END PRIVATE KEY-----\n"
        client_email = "…@….iam.gserviceaccount.com"
        # … the rest of the JSON key …

     (or env vars GCP_SERVICE_ACCOUNT_JSON + GSHEETS_ID for Heroku/DO.)
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import db

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_HEADERS: Dict[str, List[str]] = {
    "participants": [
        "participant_id", "created_at", "current_stage", "consent",
        "pre_score", "post_score", "delta", "n_sim_runs", "demographics",
        "pre_completed_at", "post_completed_at", "feedback_completed_at",
    ],
    "responses": [
        "participant_id", "stage", "question_id", "question_type",
        "answer", "is_correct", "score", "created_at",
    ],
    "activity": [
        "participant_id", "event_type", "payload", "created_at",
    ],
}


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

def _service_account_info() -> Optional[Dict[str, Any]]:
    """Service-account dict from Streamlit secrets or an env var."""
    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass
    raw = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


def _sheet_id() -> str:
    sid = ""
    try:
        import streamlit as st
        sid = str(st.secrets.get("gsheets_id", "") or "")
    except Exception:
        sid = ""
    sid = sid or os.environ.get("GSHEETS_ID", "")
    if "/d/" in sid:                       # accept a full sheet URL
        sid = sid.split("/d/", 1)[1].split("/", 1)[0]
    return sid.strip()


def is_configured() -> bool:
    return _service_account_info() is not None and bool(_sheet_id())


def sheet_url() -> Optional[str]:
    sid = _sheet_id()
    return f"https://docs.google.com/spreadsheets/d/{sid}" if sid else None


# ══════════════════════════════════════════════════════════════════════
# CLIENT (cached)
# ══════════════════════════════════════════════════════════════════════

def _spreadsheet():
    """
    Authorised gspread Spreadsheet. Cached for the process via
    st.cache_resource when Streamlit is available, otherwise memoised locally.
    """
    try:
        import streamlit as st
        return _spreadsheet_cached_st()
    except Exception:
        return _spreadsheet_build()


_SS_CACHE: Dict[str, Any] = {}


def _spreadsheet_build():
    if "ss" not in _SS_CACHE:
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(
            _service_account_info(), scopes=_SCOPES
        )
        _SS_CACHE["ss"] = gspread.authorize(creds).open_by_key(_sheet_id())
    return _SS_CACHE["ss"]


try:
    import streamlit as _st

    @_st.cache_resource(show_spinner=False)
    def _spreadsheet_cached_st():
        return _spreadsheet_build()
except Exception:  # pragma: no cover - streamlit always present in app
    def _spreadsheet_cached_st():
        return _spreadsheet_build()


def _worksheet(name: str):
    import gspread
    ss = _spreadsheet()
    try:
        return ss.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=name, rows=2000, cols=len(_HEADERS[name]))
        ws.append_row(_HEADERS[name], value_input_option="RAW")
        return ws


# ══════════════════════════════════════════════════════════════════════
# ROW SERIALISERS
# ══════════════════════════════════════════════════════════════════════

def _s(v: Any) -> Any:
    if v is None:
        return ""
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _participant_row(p: Dict[str, Any]) -> List[Any]:
    pre, post = p.get("pre_score"), p.get("post_score")
    delta = (post - pre) if (pre is not None and post is not None) else ""
    return [
        p["id"], _s(p.get("created_at")), p.get("current_stage"),
        bool(p.get("consent")), _s(pre), _s(post), _s(delta),
        db.count_activities(p["id"]), json.dumps(p.get("demographics") or {}),
        _s(p.get("pre_completed_at")), _s(p.get("post_completed_at")),
        _s(p.get("feedback_completed_at")),
    ]


def _response_row(r: Dict[str, Any]) -> List[Any]:
    return [
        r.get("participant_id"), r.get("stage"), r.get("question_id"),
        r.get("question_type"), _s(r.get("answer")), _s(r.get("is_correct")),
        _s(r.get("score")), _s(r.get("created_at")),
    ]


def _activity_row(a: Dict[str, Any]) -> List[Any]:
    return [
        a.get("participant_id"), a.get("event_type"),
        json.dumps(a.get("payload") or {}), _s(a.get("created_at")),
    ]


# ══════════════════════════════════════════════════════════════════════
# LIVE MIRROR  (best-effort — never raises into the app)
# ══════════════════════════════════════════════════════════════════════

def upsert_participant(participant: Dict[str, Any]) -> None:
    if not is_configured():
        return
    try:
        ws = _worksheet("participants")
        row = _participant_row(participant)
        cell = ws.find(participant["id"], in_column=1)
        if cell:
            ws.update([row], f"A{cell.row}")
        else:
            ws.append_row(row, value_input_option="RAW")
    except Exception:
        pass


def append_responses(pid: str, stage_key: str, rows: List[Dict[str, Any]]) -> None:
    if not is_configured() or not rows:
        return
    try:
        ws = _worksheet("responses")
        now = _now()
        data = [
            [pid, stage_key, r["question_id"], r["question_type"],
             _s(r.get("answer")), _s(r.get("is_correct")), _s(r.get("score")), now]
            for r in rows
        ]
        ws.append_rows(data, value_input_option="RAW")
    except Exception:
        pass


def append_activity(pid: str, event_type: str, payload: Dict[str, Any]) -> None:
    if not is_configured():
        return
    try:
        ws = _worksheet("activity")
        ws.append_row(
            [pid, event_type, json.dumps(payload or {}), _now()],
            value_input_option="RAW",
        )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# FULL SYNC  (admin-triggered — authoritative rebuild; may raise)
# ══════════════════════════════════════════════════════════════════════

def sync_all() -> Dict[str, int]:
    """Rebuild all three tabs from the database. Returns row counts."""
    if not is_configured():
        raise RuntimeError("Google Sheets is not configured.")

    parts = db.all_participants()
    resp = db.all_responses()
    acts = db.all_activities()

    payloads = {
        "participants": [_HEADERS["participants"]] + [_participant_row(p) for p in parts],
        "responses":    [_HEADERS["responses"]] + [_response_row(r) for r in resp],
        "activity":     [_HEADERS["activity"]] + [_activity_row(a) for a in acts],
    }
    for name, rows in payloads.items():
        ws = _worksheet(name)
        ws.clear()
        ws.update(rows, "A1", value_input_option="RAW")

    return {
        "participants": len(parts),
        "responses": len(resp),
        "activity": len(acts),
    }
