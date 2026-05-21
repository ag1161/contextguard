"""
Evaluation pipeline for ContextGuard experiments.

Each response is scored across multiple metrics and returned as a
structured result dict that can be serialised to JSON.
"""

from app.metrics import (
    hallucination_score,
    instruction_drift,
    length_penalty,
    confidence_score,
)
from app.utils import timestamp


def evaluate_response(
    prompt: str,
    response: str,
    variant: str,
    question: str = "",
) -> dict:
    """
    Score a single LLM response and return a result record.

    Args:
        prompt:    The full prompt sent to the model.
        response:  The model's raw text response.
        variant:   The prompt variant label (e.g., "baseline", "adversarial").
        question:  The original question (for traceability).

    Returns:
        A dict with all scores and metadata.
    """
    return {
        "timestamp": timestamp(),
        "question": question,
        "variant": variant,
        "prompt": prompt,
        "response": response,
        "hallucination_score": hallucination_score(response),
        "instruction_drift": instruction_drift(prompt, response),
        "length_penalty": length_penalty(response),
        "confidence_score": confidence_score(response),
        "word_count": len(response.split()),
    }


def aggregate_results(results: list[dict]) -> dict:
    """
    Compute mean metrics across all result records.

    Args:
        results: List of dicts returned by evaluate_response.

    Returns:
        A dict of averaged metrics.
    """
    n = len(results)
    if n == 0:
        return {}

    return {
        "total_evaluations": n,
        "avg_hallucination_score": round(
            sum(r["hallucination_score"] for r in results) / n, 4
        ),
        "avg_instruction_drift": round(
            sum(r["instruction_drift"] for r in results) / n, 4
        ),
        "avg_length_penalty": round(
            sum(r["length_penalty"] for r in results) / n, 4
        ),
        "avg_confidence_score": round(
            sum(r["confidence_score"] for r in results) / n, 4
        ),
        "avg_word_count": round(
            sum(r["word_count"] for r in results) / n, 1
        ),
    }
