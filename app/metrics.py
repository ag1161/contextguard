"""
Heuristic metrics for evaluating LLM response quality.

These are lightweight proxy signals — not ground-truth measures.
Designed to be fast, deterministic, and interpretable as a first pass
before integrating LLM-as-judge or human evaluation pipelines.

Metrics
-------
hallucination_score     : Regex-based hedging/uncertainty language detection
instruction_drift       : Deviation between stated constraints and actual output
stability_score         : Word-count variance across repeated runs of same prompt
length_penalty          : Normalised verbosity relative to a reference length
confidence_score        : Presence of assertive/declarative language
"""

import re


# ── Constants ──────────────────────────────────────────────────────────────────

UNCERTAINTY_MARKERS = [
    "might",
    "possibly",
    "could",
    "uncertain",
    "I think",
    "likely",
    "may be",
    "suggests",
    "I believe",
    "not sure",
    "it seems",
    "perhaps",
]

CONFIDENCE_SIGNALS = [
    "is",
    "are",
    "was",
    "were",
    "will",
    "the answer is",
    "clearly",
]

# Max word count used to normalise length_penalty to [0, 1]
LENGTH_NORM = 200


# ── Core metrics ───────────────────────────────────────────────────────────────

def hallucination_score(response: str) -> float:
    """
    Estimate hallucination likelihood via hedging-language frequency.

    Uses word-boundary-aware regex to avoid false positives (e.g., "might"
    inside "mightily"). Returns a float in [0, 1].

    0.0 = no hedging signals detected
    1.0 = saturated with speculative language
    """
    lower = response.lower()
    hits = sum(
        bool(re.search(rf"\b{re.escape(m)}\b", lower))
        for m in UNCERTAINTY_MARKERS
    )
    return round(min(hits / len(UNCERTAINTY_MARKERS), 1.0), 4)


def instruction_drift(prompt: str, response: str) -> float:
    """
    Measure deviation between stated instruction constraints and actual output.

    Acts as a proxy for instruction-following robustness. Examples:
      - "1 sentence" constraint violated by multi-sentence response → 1.0
      - No detectable constraint in prompt → 0.5 (neutral / unknown)
      - Constraint appears satisfied → 0.0

    Returns a float in {0.0, 0.5, 1.0}.
    """
    prompt_lower = prompt.lower()

    if "1 sentence" in prompt_lower or "one sentence" in prompt_lower:
        sentences = [s.strip() for s in re.split(r"[.!?]", response) if s.strip()]
        return 1.0 if len(sentences) > 1 else 0.0

    if "bullet" in prompt_lower or "list" in prompt_lower:
        has_bullets = bool(re.search(r"^\s*[-*•]", response, re.MULTILINE))
        return 0.0 if has_bullets else 0.8

    return 0.5  # no detectable constraint → neutral


def stability_score(responses: list[str]) -> float:
    """
    Measure response length variance across repeated runs of the same prompt.

    Higher values indicate less stable (more variable) outputs.
    Returns the range of word counts (max − min).

    Args:
        responses: List of response strings from repeated identical queries.

    Returns:
        Float representing word-count spread. 0.0 = perfectly stable.
    """
    if not responses:
        return 0.0
    lengths = [len(r.split()) for r in responses]
    return float(max(lengths) - min(lengths))


def length_penalty(response: str) -> float:
    """
    Penalise responses that are excessively long relative to LENGTH_NORM.

    Returns a float in [0, 1]:
      0.0 = very short response
      1.0 = at or above LENGTH_NORM words
    """
    word_count = len(response.split())
    return round(min(word_count / LENGTH_NORM, 1.0), 4)


def confidence_score(response: str) -> float:
    """
    Rough measure of assertive language in the response.

    Higher = more declarative/confident phrasing.
    Returns a float in [0, 1].
    """
    lower = response.lower()
    hits = sum(1 for s in CONFIDENCE_SIGNALS if s in lower)
    return round(min(hits / len(CONFIDENCE_SIGNALS), 1.0), 4)
