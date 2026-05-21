"""
Heuristic metrics for evaluating LLM response quality.

These are lightweight proxy signals — not ground-truth measures.
Designed to be fast, deterministic, and interpretable as a first pass
before integrating LLM-as-judge or human evaluation pipelines.

Metrics
-------
hallucination_score   : Regex-based hedging/uncertainty language detection
instruction_drift     : Deviation between stated constraints and actual output
stability_score       : Word-count variance across N repeated runs of same prompt
length_penalty        : Normalised verbosity relative to a reference length
confidence_score      : Presence of genuinely assertive/declarative phrasing

Note on stability_score
-----------------------
This metric requires N_RUNS > 1 in config.py so the same prompt is run
multiple times. With N_RUNS = 1 (default) it always returns 0.0 and is
excluded from result records.
"""

import re
import statistics


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

# Signals chosen for discriminative power — common copulas (is/are/was)
# were removed because they appear in virtually every sentence and make
# the score effectively constant across all variants.
CONFIDENCE_SIGNALS = [
    "the answer is",
    "clearly",
    "definitively",
    "certainly",
    "without a doubt",
    "the fact is",
    "it is known",
    "research shows",
    "evidence shows",
    "proven",
]

# Max word count used to normalise length_penalty to [0, 1]
LENGTH_NORM = 200


# ── Core metrics ───────────────────────────────────────────────────────────────

def hallucination_score(response: str) -> float:
    """
    Estimate hallucination likelihood via hedging-language frequency.

    Uses word-boundary-aware regex to avoid false positives (e.g. "might"
    inside "mightily"). Returns a float in [0, 1].

    0.0 = no hedging signals detected
    1.0 = saturated with speculative language
    """
    lower = response.lower()
    hits = sum(
        bool(re.search(rf"\b{re.escape(m.lower())}\b", lower))
        for m in UNCERTAINTY_MARKERS
    )
    return round(min(hits / len(UNCERTAINTY_MARKERS), 1.0), 4)


def instruction_drift(prompt: str, response: str) -> float:
    """
    Measure deviation between stated instruction constraints and actual output.

    Returns a float in {0.0, 0.5, 1.0}:
      0.0 = constraint detected and satisfied
      0.5 = no detectable constraint (neutral / unknown)
      1.0 = constraint detected but violated
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
    Measure response consistency across N repeated runs of the same prompt.

    Uses the coefficient of variation (std / mean) of word counts, normalised
    to [0, 1]. A score of 0.0 means perfectly stable output; higher scores
    indicate greater variability.

    Requires N >= 2 responses. Returns 0.0 for a single response.

    Args:
        responses: List of N response strings from repeated identical queries.

    Returns:
        Float in [0, 1] representing normalised output instability.
    """
    if len(responses) < 2:
        return 0.0
    lengths = [len(r.split()) for r in responses]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0
    cv = statistics.stdev(lengths) / mean  # coefficient of variation
    return round(min(cv, 1.0), 4)


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
    Measure the presence of genuinely assertive or declarative language.

    Signals are chosen to be discriminating — common copulas (is/are/was/were)
    were deliberately excluded as they appear in virtually every sentence.

    Returns a float in [0, 1].
    """
    lower = response.lower()
    hits = sum(
        bool(re.search(rf"\b{re.escape(s)}\b", lower))
        for s in CONFIDENCE_SIGNALS
    )
    return round(min(hits / len(CONFIDENCE_SIGNALS), 1.0), 4)
