"""Input classifier: deterministic keyword gate, then optional cheap LLM class."""

from __future__ import annotations

import re

from clinical_rag.adapters.llms import LlmClient, parse_json_object
from clinical_rag.generate.models import InputClass
from clinical_rag.generate.prompts import classify_messages

# Hard refuse — emergency / self-harm / clearly unsafe for this product.
_UNSAFE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in (
        r"\b(kill\s+myself|suicide|suicidal|end\s+my\s+life|self[-\s]?harm)\b",
        r"\b(call\s+911\s+now|dial\s+911\s+now)\b",
        r"\b(overdose\s+on\s+purpose|how\s+to\s+poison)\b",
        r"\b(make\s+(a\s+)?bomb|build\s+(a\s+)?weapon)\b",
    )
]

# Clearly unrelated to pharmacology / clinical learning.
_UNRELATED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in (
        r"\b(weather|forecast|stock\s+price|crypto|bitcoin)\b",
        r"\b(write\s+(me\s+)?(python|javascript|java)\s+code|leetcode)\b",
        r"\b(who\s+won\s+the\s+(game|match)|sports\s+score|nba|nfl)\b",
        r"\b(recipe\s+for\s+pasta|movie\s+recommendation)\b",
    )
]

# Patient / personal-care voice — product is a clinician/student copilot only.
_PATIENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in (
        r"\b(i\s*('?m|am)\s+a\s+patient|as\s+a\s+patient)\b",
        r"\b(i\s+was\s+prescribed|my\s+doctor\s+(said|told|prescribed|gave))\b",
        r"\b(should\s+i\s+(take|stop|start|dose)|can\s+i\s+take|do\s+i\s+need\s+to\s+take)\b",
        r"\b(what\s+should\s+i\s+(take|dose)|prescribe\s+me|for\s+my\s+(symptoms?|condition|illness))\b",
        r"\b(i\s+have\s+(been\s+)?(having\s+)?(symptoms?|pain|a\s+fever)|my\s+symptoms?)\b",
        r"\b(treat\s+my|help\s+me\s+(decide|choose)\s+(a\s+)?(drug|medication|medicine))\b",
        r"\b(i\s*('?m|am)\s+(pregnant|breastfeeding).{0,40}\b(take|taking|dose)\b)",
        r"\b(i\s+take|i\s*'?m\s+taking|i\s+have\s+been\s+taking)\b.{0,40}\b(should\s+i|is\s+it\s+safe\s+for\s+me)\b",
    )
]

# Clinician educational topics that need stronger caution language (still served).
_CAUTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in (
        r"\b(pregnan(t|cy)|breastfeed|lactat)\b",
        r"\b(pediatric|neonat|infant|child\s+dose)\b",
        r"\b(renal\s+fail|hepatic\s+fail|dialysis)\b",
    )
]

# Guideline / screening / risk-score topics this StatPearls pharma index is not for.
# Still retrieve (show off-topic hits) but force Case C — do not generate a recommendation.
_OUT_OF_CORPUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in (
        r"\b(uspstf|u\.s\.p\.s\.t\.f)\b",
        r"\b(acc\s*/\s*aha|aha\s*/\s*acc|acc\/aha|aha\/acc)\b",
        r"\b(nccn|nice\s+guideline|who\s+guideline)\b",
        r"\bper\s+(the\s+)?(cdc|uspstf|acc|aha|nice)\b",
        r"\baccording\s+to\s+(the\s+)?(cdc|uspstf|acc|aha|nice|ada)\b",
        r"\b(cancer\s+screening|screening\s+interval|colorectal\s+cancer\s+screening|"
        r"breast\s+cancer\s+screening)\b",
        r"\b(chads2|cha2ds2|cha₂ds₂|chads₂)\b",
        r"\b(a1c|hba1c|hgba1c)\b.{0,60}\b(target|goal)\b",
        r"\b(target|goal)\b.{0,60}\b(a1c|hba1c|hgba1c)\b",
        r"\b(influenza|flu)\s+vaccination\b",
        r"\bvaccination\b.{0,40}\b(cdc|schedule|interval)\b",
        r"\bfirst[-\s]?line\b.{0,80}\b(hypertension|essential\s+hypertension)\b",
        r"\b(guideline|guidelines)\b.{0,40}\b(hypertension|diabetes|screening|anticoag)",
    )
]


def is_out_of_corpus(message: str) -> bool:
    """True when the question targets guidelines/screening/scores outside this corpus."""
    text = (message or "").strip()
    if not text:
        return False
    return any(pat.search(text) for pat in _OUT_OF_CORPUS_PATTERNS)


def keyword_classify(message: str) -> InputClass | None:
    """Return a definitive class from keywords, or None if unclear."""
    text = (message or "").strip()
    if not text:
        return InputClass.refuse
    for pat in _UNSAFE_PATTERNS:
        if pat.search(text):
            return InputClass.refuse
    for pat in _UNRELATED_PATTERNS:
        if pat.search(text):
            return InputClass.refuse
    for pat in _PATIENT_PATTERNS:
        if pat.search(text):
            return InputClass.refuse
    for pat in _CAUTION_PATTERNS:
        if pat.search(text):
            return InputClass.needs_caution
    return None


def classify_input(
    message: str,
    llm: LlmClient | None = None,
) -> InputClass:
    """Deterministic first; optional LLM if unclear. LLM failure → NeedsCaution."""
    fixed = keyword_classify(message)
    if fixed is not None:
        return fixed
    if llm is None:
        return InputClass.allowed
    try:
        raw = llm.complete(classify_messages(message), json_mode=True)
        data = parse_json_object(raw)
        label = str(data.get("class") or data.get("label") or "").strip()
        normalized = label.replace(" ", "").lower()
        if normalized in ("refuse", "refused"):
            return InputClass.refuse
        if normalized in ("needscaution", "caution", "needs_caution"):
            return InputClass.needs_caution
        if normalized in ("allowed", "allow", "ok"):
            return InputClass.allowed
    except Exception:
        return InputClass.needs_caution
    return InputClass.needs_caution
