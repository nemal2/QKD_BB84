"""
survey.models
══════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Pure data models for the integrated survey system. No Streamlit, no
database imports — so this module can be imported anywhere and unit
tested in isolation.

Two things live here:

  1. Stage      — the forced participant lifecycle
                  (entry → pre → activity → post → feedback → done)
  2. Question   — the schema you fill in inside `survey/questions.py`
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════
# PARTICIPANT LIFECYCLE
# ══════════════════════════════════════════════════════════════════════

class Stage(str, Enum):
    """
    The single forced flow every participant moves through.

    The string values are what gets stored in the database
    (`participants.current_stage`), so do not rename them casually.
    """
    ENTRY    = "entry"        # consent + auto-code + (optional) demographics
    PRE      = "pre_survey"   # pre-knowledge survey
    ACTIVITY = "activity"     # free use of the BB84 simulator
    POST     = "post_survey"  # post-knowledge survey (same scored items as pre)
    FEEDBACK = "feedback"     # experience / feedback form
    DONE     = "done"         # finished everything

    @property
    def order(self) -> int:
        return _ORDER.index(self)

    def advance(self) -> "Stage":
        """Return the next stage (clamped at DONE)."""
        i = _ORDER.index(self)
        return _ORDER[min(i + 1, len(_ORDER) - 1)]


_ORDER: List[Stage] = [
    Stage.ENTRY, Stage.PRE, Stage.ACTIVITY, Stage.POST, Stage.FEEDBACK, Stage.DONE
]

# Stage keys used for the `responses.stage` column (only stages that collect answers).
SURVEY_STAGE_KEYS = {"pre", "post", "feedback"}


# ══════════════════════════════════════════════════════════════════════
# QUESTION SCHEMA
# ══════════════════════════════════════════════════════════════════════

class QType(str, Enum):
    """Question input types supported by the rendering engine."""
    SINGLE = "single_choice"   # radio — exactly one answer; may be scored
    MULTI  = "multi_choice"    # checkboxes — zero or more answers; may be scored
    LIKERT = "likert"          # 1–5 agreement scale (a styled single choice)
    OPEN   = "open_text"       # multi-line free text
    SHORT  = "short_text"      # single-line free text


@dataclass(frozen=True)
class Option:
    """One selectable answer for SINGLE / MULTI / LIKERT questions."""
    key: str     # stable identifier stored in the DB (e.g. "a", "b", "1")
    label: str   # text shown to the participant


@dataclass
class Question:
    """
    One survey question.

    Scoring
    ───────
    Set `scored=True` and provide `correct` to make a question count toward
    the pre/post knowledge score:
      • SINGLE → `correct` is a single option key, e.g. "b"
      • MULTI  → `correct` is a "|"-joined set of keys, e.g. "a|c"
    Open / short / likert questions are never scored (qualitative only).
    """
    id:       str
    type:     QType
    prompt:   str
    options:  List[Option]    = field(default_factory=list)
    correct:  Optional[str]   = None
    points:   float           = 1.0
    scored:   bool            = False
    required: bool            = True
    help:     Optional[str]   = None

    def __post_init__(self) -> None:
        if self.scored:
            if self.type not in (QType.SINGLE, QType.MULTI):
                raise ValueError(
                    f"Question {self.id!r}: only SINGLE/MULTI questions can be scored."
                )
            if not self.correct:
                raise ValueError(
                    f"Question {self.id!r}: scored questions need a `correct` answer."
                )
        if self.type in (QType.SINGLE, QType.MULTI, QType.LIKERT) and not self.options:
            raise ValueError(f"Question {self.id!r}: choice questions need `options`.")


# ══════════════════════════════════════════════════════════════════════
# LIKERT HELPER
# ══════════════════════════════════════════════════════════════════════

LIKERT_5: List[Option] = [
    Option("1", "Strongly disagree"),
    Option("2", "Disagree"),
    Option("3", "Neutral"),
    Option("4", "Agree"),
    Option("5", "Strongly agree"),
]


def likert(qid: str, prompt: str, *, required: bool = True,
           help: Optional[str] = None) -> Question:
    """Convenience builder for a standard 1–5 agreement-scale question."""
    return Question(
        id=qid, type=QType.LIKERT, prompt=prompt,
        options=list(LIKERT_5), required=required, help=help,
    )
