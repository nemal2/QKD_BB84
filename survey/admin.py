"""
survey.admin
══════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Password-gated researcher dashboard. Reach it at:

    https://<your-app>/?survey=1     (or /?admin=1, or /survey via nginx)

Set the password via the SURVEY_ADMIN_PASSWORD environment variable
(or the same key in .streamlit/secrets.toml). Without it the dashboard
stays locked.

What it shows
─────────────
  • completion funnel across the four stages
  • pre vs post knowledge scores and the learning gain (delta)
  • a per-participant table linking scores to recorded app activity
  • Excel (.xlsx) workbook export + CSV: participants, responses
    (long + wide), activity
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
import json
import os
from collections import Counter
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import db
from .models import QType, Stage
from .questions import DEMOGRAPHICS, FEEDBACK, POST_SURVEY, PRE_SURVEY

BLUE, GREEN, AMBER, GRAY, RED = "#2563EB", "#059669", "#D97706", "#6B7280", "#DC2626"
_PL = dict(
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    font=dict(color="#6B7280", family="Outfit, sans-serif", size=11),
    margin=dict(t=40, b=36, l=10, r=10),
)


# ══════════════════════════════════════════════════════════════════════
# DATA FRAMES
# ══════════════════════════════════════════════════════════════════════

def _participants_wide() -> pd.DataFrame:
    parts = db.all_participants()
    acts = db.activity_counts_by_participant()
    rows = []
    for p in parts:
        demo = p.get("demographics") or {}
        pre, post = p.get("pre_score"), p.get("post_score")
        delta = (post - pre) if (pre is not None and post is not None) else None
        row = {
            "participant_id": p["id"],
            "created_at": p["created_at"],
            "current_stage": p["current_stage"],
            "consent": p["consent"],
            "pre_score": pre,
            "post_score": post,
            "delta": delta,
            "n_sim_runs": acts.get(p["id"], 0),
            "pre_completed_at": p.get("pre_completed_at"),
            "post_completed_at": p.get("post_completed_at"),
            "feedback_completed_at": p.get("feedback_completed_at"),
        }
        row.update(demo)  # demographic keys (demo_year, demo_prior, …)
        rows.append(row)
    return pd.DataFrame(rows)


def _responses_long() -> pd.DataFrame:
    return pd.DataFrame(db.all_responses())


def _activities_long() -> pd.DataFrame:
    acts = db.all_activities()
    for a in acts:
        a["payload"] = json.dumps(a.get("payload") or {})
    return pd.DataFrame(acts)


def _responses_wide(resp: pd.DataFrame) -> pd.DataFrame:
    """Analysis-ready pivot: one row per participant, one column per question."""
    if resp.empty:
        return pd.DataFrame()
    wide = resp.pivot_table(
        index="participant_id", columns="question_id",
        values="answer", aggfunc="first",
    )
    return wide.reset_index()


def _excel_bytes(sheets_map: dict) -> bytes:
    """Build a single .xlsx workbook with one tab per DataFrame."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, frame in sheets_map.items():
            out = frame if (frame is not None and not frame.empty) \
                else pd.DataFrame({"info": ["no data"]})
            out.to_excel(xl, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ── Human-readable response helpers ────────────────────────────────────

# stage key → (title, question list)
_STAGE_DEFS = [
    ("pre", "Pre-Survey", PRE_SURVEY),
    ("post", "Post-Survey", POST_SURVEY),
    ("feedback", "Feedback", FEEDBACK),
]


def _qindex() -> dict:
    """{question_id: Question} across every stage incl. demographics."""
    return {q.id: q for q in (DEMOGRAPHICS + PRE_SURVEY + POST_SURVEY + FEEDBACK)}


def _answer_label(q, raw) -> str:
    """Turn a stored answer (option key / text) into human-readable form."""
    if raw is None or str(raw) == "":
        return "—"
    if q is None:
        return str(raw)
    if q.type in (QType.SINGLE, QType.LIKERT):
        lab = {o.key: o.label for o in q.options}.get(str(raw))
        if lab is None:
            return str(raw)
        return f"{raw} — {lab}" if q.type == QType.LIKERT else lab
    if q.type == QType.MULTI:
        labs = {o.key: o.label for o in q.options}
        return ", ".join(labs.get(k, k) for k in str(raw).split("|") if k)
    return str(raw)


def _stage_summary_df(questions, resp_stage) -> pd.DataFrame:
    """One row per choice/likert question: prompt, N, mean (likert), distribution."""
    rows = []
    for q in questions:
        if q.type not in (QType.SINGLE, QType.LIKERT, QType.MULTI):
            continue
        ans = [str(a) for a in resp_stage[resp_stage.question_id == q.id]["answer"].tolist()]
        labs = {o.key: o.label for o in q.options}
        counter: Counter = Counter()
        for a in ans:
            for k in (a.split("|") if q.type == QType.MULTI else [a]):
                if k:
                    counter[labs.get(k, k)] += 1
        mean = ""  # keep this column all-strings so st.dataframe (pyarrow) is happy
        if q.type == QType.LIKERT:
            nums = [int(a) for a in ans if a.isdigit()]
            mean = f"{sum(nums) / len(nums):.2f}" if nums else ""
        rows.append({
            "Question": q.prompt,
            "N": len(ans),
            "Mean (1–5)": mean,
            "Responses": " · ".join(f"{lbl}: {cnt}" for lbl, cnt in counter.most_common()),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════

def _check_password() -> bool:
    configured: Optional[str] = os.environ.get("SURVEY_ADMIN_PASSWORD")
    if not configured:
        st.error(
            "Admin dashboard is locked. Set the **SURVEY_ADMIN_PASSWORD** "
            "environment variable (or add it to `.streamlit/secrets.toml`) to enable it."
        )
        return False

    if st.session_state.get("survey_admin_ok"):
        return True

    st.markdown("### 🔒 Researcher login")
    with st.form("admin_login"):
        pw = st.text_input("Admin password", type="password")
        ok = st.form_submit_button("Enter", type="primary")
    if ok:
        if pw == configured:
            st.session_state["survey_admin_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ══════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════

def render_admin() -> None:
    st.markdown(
        "<h2 style='font-size:22px;font-weight:700;color:#111827;margin:0 0 4px;'>"
        "Survey Admin · Researcher Dashboard</h2>",
        unsafe_allow_html=True,
    )

    if not _check_password():
        return

    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.caption(f"Storage backend: **{db.backend_name()}**")
    with top_r:
        if st.button("← Exit dashboard", use_container_width=True):
            for _k in ("admin", "survey"):
                if _k in st.query_params:
                    del st.query_params[_k]
            st.rerun()

    df = _participants_wide()
    total = len(df)
    if total == 0:
        st.info("No participants yet. Share the app link to start collecting data.")
        return

    # ── Funnel / completion ────────────────────────────────────────────
    counts = db.stage_counts()

    def _reached(stage: Stage) -> int:
        return sum(n for s, n in counts.items() if Stage(s).order >= stage.order)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Participants", total)
    m2.metric("Completed pre", _reached(Stage.ACTIVITY))
    m3.metric("Completed post", _reached(Stage.FEEDBACK))
    m4.metric("Finished study", counts.get("done", 0))
    completion = counts.get("done", 0) / total * 100 if total else 0
    m5.metric("Completion rate", f"{completion:.0f}%")

    st.divider()

    # ── Pre vs Post ────────────────────────────────────────────────────
    from .questions import POST_SURVEY, PRE_SURVEY
    _has_scored = (any(q.scored for q in PRE_SURVEY)
                   and any(q.scored for q in POST_SURVEY))
    paired = df.dropna(subset=["pre_score", "post_score"])
    st.markdown("#### Knowledge: pre vs post")
    if not _has_scored:
        st.info(
            "This study uses self-report (Likert) items only — there is no scored "
            "knowledge test, so there is no pre/post score to compare. Analyse the "
            "rated and written responses via the CSV / Google Sheets export below."
        )
    elif paired.empty:
        st.info("No participant has completed both the pre- and post-survey yet.")
    else:
        mean_pre = paired["pre_score"].mean()
        mean_post = paired["post_score"].mean()
        mean_delta = paired["delta"].mean()

        a, b, c, d = st.columns(4)
        a.metric("Mean pre-score", f"{mean_pre:.1f}%")
        b.metric("Mean post-score", f"{mean_post:.1f}%", f"{mean_delta:+.1f} pts")
        c.metric("Mean learning gain", f"{mean_delta:+.1f} pts")
        improved = int((paired["delta"] > 0).sum())
        d.metric("Improved", f"{improved}/{len(paired)}")

        cl, cr = st.columns(2)
        with cl:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Pre", "Post"], y=[mean_pre, mean_post],
                marker_color=[GRAY, GREEN],
                text=[f"{mean_pre:.1f}%", f"{mean_post:.1f}%"], textposition="outside",
            ))
            fig.update_layout(**{**_PL, "height": 300, "title": "Mean score"})
            fig.update_yaxes(range=[0, 105], title="Score (%)")
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            fig2 = go.Figure()
            for _, row in paired.iterrows():
                fig2.add_trace(go.Scatter(
                    x=["Pre", "Post"], y=[row["pre_score"], row["post_score"]],
                    mode="lines+markers", line=dict(color="rgba(37,99,235,.35)", width=1),
                    marker=dict(size=6, color=BLUE), showlegend=False,
                    hovertext=row["participant_id"],
                ))
            fig2.update_layout(**{**_PL, "height": 300, "title": "Per-participant change"})
            fig2.update_yaxes(range=[0, 105], title="Score (%)")
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Per-participant table ──────────────────────────────────────────
    st.markdown("#### Participants")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Responses (readable) ───────────────────────────────────────────
    st.markdown("#### Responses")
    resp_view = _responses_long()
    if resp_view.empty:
        st.caption("No survey responses submitted yet.")
    else:
        qidx = _qindex()
        mode = st.radio(
            "View responses", ["By question (summary)", "By participant"],
            horizontal=True, label_visibility="collapsed",
        )
        if mode.startswith("By question"):
            for key, title, qs in _STAGE_DEFS:
                rs = resp_view[resp_view.stage == key]
                if rs.empty:
                    continue
                st.markdown(f"**{title}**  ·  {rs['participant_id'].nunique()} respondents")
                summary = _stage_summary_df(qs, rs)
                if not summary.empty:
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                for q in qs:
                    if q.type in (QType.OPEN, QType.SHORT):
                        texts = [str(a) for a in rs[rs.question_id == q.id]["answer"].tolist()
                                 if str(a).strip()]
                        if texts:
                            with st.expander(f"✍  {q.prompt}   ({len(texts)})"):
                                for t in texts:
                                    st.markdown(f"- {t}")
        else:
            pids = sorted(resp_view["participant_id"].unique())
            sel = st.selectbox("Participant", pids)
            prow = next((p for p in db.all_participants() if p["id"] == sel), None)
            demo = (prow or {}).get("demographics") or {}
            if demo:
                st.markdown("**Background**")
                for qid, val in demo.items():
                    q = qidx.get(qid)
                    st.markdown(f"- {q.prompt if q else qid} → **{_answer_label(q, val)}**")
            for key, title, qs in _STAGE_DEFS:
                rs = resp_view[(resp_view.stage == key)
                               & (resp_view.participant_id == sel)]
                if rs.empty:
                    continue
                st.markdown(f"**{title}**")
                amap = dict(zip(rs["question_id"], rs["answer"]))
                for q in qs:
                    if q.id in amap:
                        st.markdown(f"- {q.prompt} → **{_answer_label(q, amap[q.id])}**")

    # ── Exports ────────────────────────────────────────────────────────
    st.markdown("#### Export")
    resp = _responses_long()
    act = _activities_long()
    resp_wide = _responses_wide(resp)

    x1, x2 = st.columns([1.5, 3])
    with x1:
        st.download_button(
            "⬇  Excel workbook (.xlsx)",
            _excel_bytes({
                "participants": df,
                "responses": resp,
                "responses_wide": resp_wide,
                "activity": act,
            }),
            "qkd_survey_data.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary", use_container_width=True,
        )
    with x2:
        st.caption(
            "One workbook, four tabs — **participants**, **responses** (long), "
            "**responses_wide** (one row per participant, ready for analysis), and "
            "**activity**. Prefer plain CSVs? Use the buttons below."
        )

    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "participants.csv", df.to_csv(index=False),
        "participants_wide.csv", "text/csv", use_container_width=True,
    )
    e2.download_button(
        "responses.csv",
        resp.to_csv(index=False) if not resp.empty else "no data",
        "responses_long.csv", "text/csv", use_container_width=True,
        disabled=resp.empty,
    )
    e3.download_button(
        "activity.csv",
        act.to_csv(index=False) if not act.empty else "no data",
        "activity_long.csv", "text/csv", use_container_width=True,
        disabled=act.empty,
    )

    # ── Secondary store · Google Sheets ────────────────────────────────
    st.divider()
    st.markdown("#### Secondary store · Google Sheets")
    from . import sheets
    if not sheets.is_configured():
        st.caption(
            "Not configured — data is stored only in the primary database. Add a "
            "Google service account + sheet id (see DEPLOYMENT.md) to enable a live "
            "secondary mirror you can open in a browser."
        )
    else:
        sc1, sc2 = st.columns([1.3, 3])
        with sc1:
            if st.button("⟳ Sync all to Google Sheets", type="primary",
                         use_container_width=True):
                try:
                    with st.spinner("Pushing to Google Sheets…"):
                        counts = sheets.sync_all()
                    st.success(
                        f"Synced — participants: {counts['participants']}, "
                        f"responses: {counts['responses']}, activity: {counts['activity']}."
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Sync failed: {e}")
        with sc2:
            st.caption(
                "Live mirror is **on**: responses & activity append automatically and "
                "participant rows update on each stage change. Use Sync for a full, "
                "authoritative rebuild from the database."
            )
            url = sheets.sheet_url()
            if url:
                st.markdown(f"[Open the Google Sheet ↗]({url})")

    with st.expander("Raw responses (long format)"):
        st.dataframe(resp, use_container_width=True, hide_index=True)
    with st.expander("Raw activity log"):
        st.dataframe(act, use_container_width=True, hide_index=True)
