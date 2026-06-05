"""
survey.scoring
══════════════════════════════════════════════════════════════
University of Ruhuna — Dept. of Computer Engineering

Grades the scored multiple-choice questions and turns a set of raw answers
into a percentage score. Because the *same* KNOWLEDGE questions are asked in
both the pre- and post-survey (see survey/questions.py), the pre/post
percentages are directly comparable — their difference is the learning gain.

Pure functions, no Streamlit / DB. Answers come in as:
    { question_id: <answer> }
where <answer> is an option key (SINGLE/LIKERT), a list of keys (MULTI),
or a string (OPEN/SHORT).
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .models import Question, QType


def grade(question: Question, answer: Any) -> Tuple[Optional[bool], float]:
    """
    Grade a single answer.

    Returns (is_correct, points_earned). For unscored questions returns
    (None, 0.0) — they carry no marks.
    """
    if not question.scored:
        return None, 0.0

    if question.type == QType.SINGLE:
        is_correct = (answer is not None and answer == question.correct)
        return is_correct, (question.points if is_correct else 0.0)

    if question.type == QType.MULTI:
        correct_set = set((question.correct or "").split("|"))
        answer_set = set(answer or [])
        is_correct = (answer_set == correct_set)
        return is_correct, (question.points if is_correct else 0.0)

    return None, 0.0


def score_survey(questions: List[Question],
                 answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grade an entire survey.

    Returns:
        {
          "earned":   float,         # marks earned
          "possible": float,         # total marks available
          "percent":  float | None,  # 0–100, or None when nothing is scored
          "n_scored": int,           # how many questions counted
          "graded":   [ {question_id, is_correct, points} , … ],
        }
    """
    earned = 0.0
    possible = 0.0
    n_scored = 0
    graded: List[Dict[str, Any]] = []

    for q in questions:
        if not q.scored:
            continue
        n_scored += 1
        possible += q.points
        is_correct, points = grade(q, answers.get(q.id))
        earned += points
        graded.append(
            {"question_id": q.id, "is_correct": is_correct, "points": points}
        )

    # None (not 0) when there are no scored questions, so downstream code can
    # tell "scored 0%" apart from "this survey has no knowledge test".
    percent = round(earned / possible * 100.0, 2) if possible > 0 else None
    return {
        "earned": earned,
        "possible": possible,
        "percent": percent,
        "n_scored": n_scored,
        "graded": graded,
    }
