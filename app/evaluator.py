"""
Evaluation pipeline for ContextGuard experiments.

Each response is scored across multiple metrics and returned as a
structured result dict that can be serialised to JSON.
"""

import logging

from app.metrics import (
    hallucination_score,
    instruction_drift,
    stability_score,
    length_penalty,
    confidence_score,
)
from app.utils import timestamp

logger = logging.getLogger(__name__)


def evaluate_response(
    prompt: str,
    response: str,
    variant: str,
    question: str = "",
    all_responses: list[str] | None = None,
) -> dict:
    """
    Score a single LLM response and return a result record.

    Args:
        prompt:         The full prompt sent to the model.
        response:       The model's raw text response.
        variant:        The prompt variant label (e.g. "baseline", "adversarial").
        question:       The original question (for traceability).
        all_responses:  All N responses for this prompt (enables stability_score).
                        Pass None or a single-item list for single-run mode.

    Returns:
        A dict with all scores and metadata.
    """
    record = {
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

    # Only include stability_score when multiple runs were performed
    if all_responses and len(all_responses) > 1:
        record["stability_score"] = stability_score(all_responses)

    return record


def aggregate_results(results: list[dict]) -> dict:
    """
    Compute mean metrics across all valid (non-error) result records.

    Error records (where response starts with "[ERROR:") are excluded
    from metric averages but counted separately.

    Args:
        results: List of dicts returned by evaluate_response.

    Returns:
        A dict of averaged metrics plus error count.
    """
    valid = [r for r in results if not r["response"].startswith("[ERROR:")]
    errors = len(results) - len(valid)
    n = len(valid)

    if errors:
        logger.warning("%d result(s) were errors and excluded from aggregation.", errors)

    if n == 0:
        return {"total_evaluations": 0, "error_count": errors}

    summary: dict = {
        "total_evaluations": len(results),
        "valid_evaluations": n,
        "error_count": errors,
        "avg_hallucination_score": round(
            sum(r["hallucination_score"] for r in valid) / n, 4
        ),
        "avg_instruction_drift": round(
            sum(r["instruction_drift"] for r in valid) / n, 4
        ),
        "avg_length_penalty": round(
            sum(r["length_penalty"] for r in valid) / n, 4
        ),
        "avg_confidence_score": round(
            sum(r["confidence_score"] for r in valid) / n, 4
        ),
        "avg_word_count": round(
            sum(r["word_count"] for r in valid) / n, 1
        ),
    }

    # Include stability summary only if multi-run data is present
    if "stability_score" in valid[0]:
        summary["avg_stability_score"] = round(
            sum(r.get("stability_score", 0.0) for r in valid) / n, 4
        )

    return summary
