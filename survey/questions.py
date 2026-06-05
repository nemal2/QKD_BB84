"""
survey.questions
══════════════════════════════════════════════════════════════
Survey content for the study:

   "Educational Evaluation of the BB84 Quantum Key Distribution
    Network Simulator"  ·  for submission to ACM SIGCSE 2027.

This is a self-report evaluation instrument — there are no scored
right/wrong knowledge items, so the pre/post *score* comparison is
inactive (perceived learning is captured by the Likert items instead).

Mapping onto the system's forced flow
─────────────────────────────────────
  entry     consent + participant background (role, discipline)
  pre       prior knowledge (asked BEFORE using the simulator)
  activity  free use of the BB84 simulator
  post      usage, usability, learning, engagement, self-efficacy, overall
  feedback  open-ended written responses

To edit: change text / options here. Keep question `id`s stable once you
have collected real data (they are database keys and CSV columns).
══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Option, Question, QType, likert


def _single(qid: str, prompt: str, options: List[Tuple[str, str]], *,
            required: bool = False, help: str | None = None) -> Question:
    return Question(
        id=qid, type=QType.SINGLE, prompt=prompt,
        options=[Option(k, lbl) for k, lbl in options],
        required=required, help=help,
    )


# Shared option set — prior-knowledge level.
_KNOW_LEVELS: List[Tuple[str, str]] = [
    ("none", "None (no prior exposure)"),
    ("basic", "Basic (aware of the concept but limited understanding)"),
    ("intermediate", "Intermediate (studied in at least one course or workshop)"),
    ("advanced", "Advanced (substantial academic or research experience)"),
]


# ══════════════════════════════════════════════════════════════════════
# ENTRY — participant background  (shown with the consent screen)
# ══════════════════════════════════════════════════════════════════════

DEMOGRAPHICS: List[Question] = [
    _single("bg_role", "What is your current role?", [
        ("undergrad", "Undergraduate Student"),
        ("postgrad", "Postgraduate Student"),
        ("lecturer", "Lecturer"),
        ("researcher", "Researcher"),
        ("other", "Other"),
    ], required=False),
    _single("bg_discipline", "What is your primary academic discipline? *", [
        ("cs", "Computer Science / Engineering"),
        ("eee", "Electrical and Electronic Engineering"),
        ("physics", "Physics"),
        ("other", "Other"),
    ], required=True),
]


# ══════════════════════════════════════════════════════════════════════
# PRE-SURVEY — prior knowledge (before using the simulator)
# ══════════════════════════════════════════════════════════════════════

PRE_SURVEY: List[Question] = [
    _single(
        "pre_know_qc",
        "What is your prior level of knowledge of Quantum Computing "
        "(before using the simulator)?",
        _KNOW_LEVELS, required=False,
    ),
    _single(
        "pre_know_qkd",
        "What is your prior level of knowledge of Quantum Key Distribution (QKD) "
        "specifically (before using the simulator)?",
        _KNOW_LEVELS, required=False,
    ),
]


# ══════════════════════════════════════════════════════════════════════
# POST-SURVEY — usage + rated evaluation (1–5 Likert) + overall
# ══════════════════════════════════════════════════════════════════════

# Likert matrices from the form — each row becomes its own 1–5 question.
_USABILITY: List[Tuple[str, str]] = [
    ("post_use_nav", "The simulator interface was easy to navigate without prior guidance."),
    ("post_use_speed", "The simulator responded without noticeable delay during simulations."),
    ("post_use_layout", "The layout and organisation of the interface were logical and intuitive."),
    ("post_use_ux", "The overall user experience of the simulator was satisfactory."),
    ("post_use_reliable", "The simulator functioned reliably without crashes or unexpected behaviour."),
]
_LEARNING: List[Tuple[str, str]] = [
    ("post_learn_bb84", "The simulator improved my understanding of the BB84 protocol and its operational steps."),
    ("post_learn_eve", "The simulator helped me understand how eavesdropping attacks are detected in QKD."),
    ("post_learn_qber", "The simulator improved my understanding of the Quantum Bit Error Rate (QBER) and its significance."),
    ("post_learn_abstract", "The simulator made abstract quantum concepts more concrete and easier to understand."),
    ("post_learn_sifting", "The simulator helped me understand the basis reconciliation and sifting process in BB84."),
]
_ENGAGEMENT: List[Tuple[str, str]] = [
    ("post_eng_interest", "The simulator increased my interest in quantum computing and quantum communication."),
    ("post_eng_vs_lecture", "Using the simulator was more engaging than traditional lecture-based explanations of the same content."),
    ("post_eng_sustained", "I remained engaged throughout my use of the simulator without losing interest."),
    ("post_eng_experiment", "The simulator made me want to experiment further by adjusting parameters and observing outcomes."),
]
_EFFICACY: List[Tuple[str, str]] = [
    ("post_eff_explain", "After using the simulator, I feel more confident explaining the BB84 protocol to others."),
    ("post_eff_qber", "After using the simulator, I feel more confident interpreting quantum bit error rate values."),
    ("post_eff_eve", "After using the simulator, I feel more confident identifying the impact of eavesdropping on key security."),
    ("post_eff_topic", "The simulator helped me feel that quantum cryptography is a topic I can understand and engage with."),
]

POST_SURVEY: List[Question] = (
    [
        _single("post_duration", "For how long did you use the simulator in total?", [
            ("lt15", "Less than 15 minutes"),
            ("15_30", "15 to 30 minutes"),
            ("31_60", "31 to 60 minutes"),
            ("gt60", "More than 60 minutes"),
        ], required=False),
        _single("post_runs", "How many separate simulation runs did you complete?", [
            ("1_2", "1 to 2 runs"),
            ("3_5", "3 to 5 runs"),
            ("6_10", "6 to 10 runs"),
            ("gt10", "More than 10 runs"),
        ], required=False),
    ]
    + [likert(qid, text, required=True) for qid, text in _USABILITY]
    + [likert(qid, text, required=True) for qid, text in _LEARNING]
    + [likert(qid, text, required=True) for qid, text in _ENGAGEMENT]
    + [likert(qid, text, required=True) for qid, text in _EFFICACY]
    + [
        _single("post_overall", "Overall, how would you rate the BB84 QKD Simulator as an educational tool?", [
            ("poor", "Poor"), ("fair", "Fair"), ("good", "Good"),
            ("very_good", "Very Good"), ("excellent", "Excellent"),
        ], required=False),
        _single("post_reuse", "How likely are you to use this simulator again in a study or teaching context?", [
            ("very_unlikely", "Very Unlikely"), ("unlikely", "Unlikely"),
            ("neutral", "Neutral"), ("likely", "Likely"), ("very_likely", "Very Likely"),
        ], required=False),
        _single("post_difficulty", "Overall, how would you rate the difficulty level of the simulator relative to your prior knowledge?", [
            ("too_easy", "Too easy for my level"),
            ("slightly_easy", "Slightly easy for my level"),
            ("appropriate", "Appropriately challenging"),
            ("slightly_hard", "Slightly difficult for my level"),
            ("too_hard", "Too difficult for my level"),
        ], required=False),
        _single("post_vs_other", "Compared to other learning tools or resources you have used, how effective was this simulator?", [
            ("much_less", "Much less effective than other tools"),
            ("slightly_less", "Slightly less effective than other tools"),
            ("about_same", "About as effective as other tools"),
            ("more", "More effective than most tools"),
            ("among_most", "Among the most effective learning tools I have used"),
        ], required=False),
    ]
)


# ══════════════════════════════════════════════════════════════════════
# FEEDBACK — open-ended written responses
# ══════════════════════════════════════════════════════════════════════

FEEDBACK: List[Question] = [
    Question(
        id="fb_useful", type=QType.OPEN,
        prompt="What aspects of the simulator did you find most useful for supporting "
               "your learning? Please describe specific features or experiences.",
        required=False,
    ),
    Question(
        id="fb_improve", type=QType.OPEN,
        prompt="Which features or areas of the simulator need improvement? "
               "Please be as specific as possible.",
        required=False,
    ),
    Question(
        id="fb_other", type=QType.OPEN,
        prompt="Is there anything else you would like to share about your experience "
               "with the simulator, including any general comments or suggestions "
               "for the research team?",
        required=False,
    ),
]


# ══════════════════════════════════════════════════════════════════════
# REGISTRY — read by the flow controller. Don't rename the keys.
# ══════════════════════════════════════════════════════════════════════

SURVEYS: Dict[str, List[Question]] = {
    "pre":      PRE_SURVEY,
    "post":     POST_SURVEY,
    "feedback": FEEDBACK,
}
