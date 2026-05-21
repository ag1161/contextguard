"""
Heuristic metrics for evaluating LLM response quality.

These are lightweight proxies — not ground-truth measures.
They are designed to be fast, deterministic, and interpretable
as a first pass before more sophisticated evaluation (e.g., LLM-as-judge).
"""


# Phrases associated with speculative or uncertain language
HALLUCINATION_SIGNALS = [
    "might",
    "possibly",
    "I think",
    "could be",
    "uncertain",
    "I believe",
    "not sure",
    "it seems",
    "perhaps",
    "may be",
]

# Max word count used to normalise the length penalty to [0, 1]
LENGTH_NORM = 200


def hallucination_score(response: str) -> float:
    """
    Estimate hallucination likelihood via hedging-language frequency.

    Returns a float in [0, 1]:
      0.0 = no hedging signals detected
      1.0 = saturated with speculative language
    """
    lower = response.lower()
    hits = sum(1 for s in HALLUCINATION_SIGNALS if s.lower() in lower)
    return round(min(hits / len(HALLUCINATION_SIGNALS), 1.0), 4)


def length_penalty(response: str) -> float:
    """
    Penalise responses that are excessively long relative to the norm.

    Returns a float in [0, 1]:
      0.0 = very short response
      1.0 = response at or above LENGTH_NORM words
    """
    word_count = len(response.split())
    return round(min(word_count / LENGTH_NORM, 1.0), 4)


def confidence_score(response: str) -> float:
    """
    Rough measure of assertive language in the response.

    Returns a float in [0, 1]:
      Higher = more confident / declarative phrasing
    """
    confidence_signals = ["is", "are", "was", "were", "will", "the answer is", "clearly"]
    lower = response.lower()
    hits = sum(1 for s in confidence_signals if s in lower)
    return round(min(hits / len(confidence_signals), 1.0), 4)
