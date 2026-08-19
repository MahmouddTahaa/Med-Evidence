"""Prompt templates for classify / rewrite / expand / grounded answer.

Always pass `citation_blocks(hits)` into the grounded-answer path.
"""

from __future__ import annotations

from clinical_rag.generate.models import ChatMessage, DISCLAIMER, InputClass
from clinical_rag.query.context import citation_blocks
from clinical_rag.schemas import SmokeHit

TAGGED_ANSWER_INSTRUCTIONS = """\
You are Med-Evidence, an educational pharmacology assistant for clinicians and students.

Rules:
- Use ONLY the numbered SOURCE blocks below. Do not invent facts, doses, or citations.
- Every clinical claim must be supportable by one or more source numbers.
- If the SOURCE blocks clearly answer the question, you MUST give a grounded recommendation \
(do not claim evidence is insufficient when sources are on-topic).
- If sources are truly off-topic or empty, say so clearly and do not invent a recommendation.
- Match recommendation tone to CONFIDENCE: high → "The source supports…"; \
medium → "The source suggests…"; low → "Limited evidence in retrieved passages…"; \
insufficient → state that evidence is insufficient (do not guess).
- Never imply this replaces clinical judgment. Do not soften a refusal for helpfulness.
- Output EXACTLY these tagged sections in order (no markdown fences):

<<<RECOMMENDATION>>>
<concise educational recommendation or insufficient-evidence statement>
<<<EVIDENCE>>>
- "<short quote>" [n]
<<<CITATIONS>>>
[n] chunk_id=<exact chunk_id from SOURCE header>
<<<CONFIDENCE>>>
high|medium|low|insufficient
"""


def _history_blob(history: list[ChatMessage], *, max_turns: int) -> str:
    if max_turns <= 0 or not history:
        return "(none)"
    trimmed = history[-(max_turns * 2) :]
    lines = [f"{m.role}: {m.content}" for m in trimmed]
    return "\n".join(lines)


def classify_messages(message: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Classify the user message for Med-Evidence, a clinician/student "
                "pharmacology copilot (not a patient app). "
                'Return JSON only: {"class":"Allowed"|"NeedsCaution"|"Refuse"}. '
                "Refuse: (1) the user sounds like a patient seeking personal care "
                "(I was prescribed, should I take, my symptoms, treat my condition); "
                "(2) emergency self-harm, weapons; (3) clearly unrelated topics "
                "(weather, sports, general coding). "
                "NeedsCaution: clinician/student questions about pregnancy, pediatrics, "
                "or organ failure that are educational (not first-person patient care). "
                "Allowed: educational pharmacology — mechanism, contraindications, "
                "interactions, monitoring, dosing facts for learning."
            ),
        },
        {"role": "user", "content": message},
    ]


def rewrite_messages(message: str, history: list[ChatMessage]) -> list[dict]:
    hist = _history_blob(history, max_turns=6)
    return [
        {
            "role": "system",
            "content": (
                "Rewrite the latest user message as a standalone pharmacology search query. "
                'Return JSON only: {"query":"..."}. '
                "Resolve pronouns from history. Do not answer the question."
            ),
        },
        {
            "role": "user",
            "content": f"History:\n{hist}\n\nLatest message:\n{message}",
        },
    ]


def expand_messages(message: str, *, n: int = 3) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                f"Generate {n} diverse short pharmacology search query variants for the same intent. "
                'Return JSON only: {"queries":["...","..."]}. No hypothetical documents.'
            ),
        },
        {"role": "user", "content": message},
    ]


def grounded_answer_messages(
    message: str,
    hits: list[SmokeHit],
    *,
    history: list[ChatMessage] | None = None,
    input_class: InputClass = InputClass.allowed,
    memory_turns: int = 6,
    prompt_k: int = 5,
) -> list[dict]:
    window = hits[:prompt_k]
    context = citation_blocks(window)
    caution = ""
    if input_class is InputClass.needs_caution:
        caution = (
            "\nEXTRA CAUTION: This topic may involve high-risk populations or personal dosing. "
            "Prefer conservative educational language; emphasize specialist confirmation.\n"
        )
    hist = _history_blob(history or [], max_turns=memory_turns)
    user = (
        f"{caution}"
        f"Dialogue context (not a substitute for sources):\n{hist}\n\n"
        f"User question:\n{message}\n\n"
        f"SOURCES:\n{context}\n\n"
        f"Fixed disclaimer to respect: {DISCLAIMER}"
    )
    return [
        {"role": "system", "content": TAGGED_ANSWER_INSTRUCTIONS},
        {"role": "user", "content": user},
    ]
