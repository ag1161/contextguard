"""
Main experiment runner for ContextGuard.

Iterates over all test cases × prompt variants, queries the LLM,
evaluates each response, and writes timestamped results to results/.

Usage:
    python experiments/run_experiment.py

Environment variables (via .env):
    OPENAI_API_KEY   — required
    MODEL_NAME       — default: gpt-4o-mini
    TEMPERATURE      — default: 0.7  (set to 0 for deterministic/reproducible runs)
    N_RUNS           — default: 1    (>1 enables stability_score)
"""

import json
import sys
import os
import logging
from tqdm import tqdm
import openai

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm_client import query_llm
from app.prompts import PROMPT_VARIANTS
from app.evaluator import evaluate_response, aggregate_results
from app.utils import save_json, timestamped_filename
from app.config import N_RUNS

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
)

TEST_CASES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_cases.json"
)


def run_experiment(test_cases_path: str = TEST_CASES_PATH) -> list[dict]:
    """
    Run the full experiment grid and return all result records.

    Each run produces a NEW timestamped file in results/ so prior runs
    are never overwritten. The latest run is always also saved as
    results/sample_results.json for dashboard compatibility.

    Args:
        test_cases_path: Path to the JSON file with test questions.

    Returns:
        List of evaluated result dicts (errors are included but flagged).
    """
    with open(test_cases_path, "r") as f:
        test_cases = json.load(f)

    results = []
    error_count = 0
    total = len(test_cases) * len(PROMPT_VARIANTS)

    logger.info(
        "Starting experiment: %d questions × %d variants × %d run(s) = %d LLM calls",
        len(test_cases), len(PROMPT_VARIANTS), N_RUNS, total * N_RUNS,
    )

    with tqdm(total=total, desc="Evaluating", unit="prompt") as pbar:
        for case in test_cases:
            question = case["question"]

            for variant, template in PROMPT_VARIANTS.items():
                prompt = template.format(question=question)
                responses: list[str] = []

                for run_idx in range(N_RUNS):
                    try:
                        resp = query_llm(prompt)
                        responses.append(resp)
                    except KeyboardInterrupt:
                        logger.warning("Interrupted by user — saving partial results.")
                        _save_results(results)
                        raise
                    except openai.RateLimitError:
                        logger.error("Rate limit hit on run %d/%d — retrying in 60s", run_idx + 1, N_RUNS)
                        import time; time.sleep(60)
                        try:
                            resp = query_llm(prompt)
                            responses.append(resp)
                        except Exception as retry_err:
                            logger.error("Retry failed: %s", retry_err)
                            responses.append(f"[ERROR: {retry_err}]")
                            error_count += 1
                    except openai.OpenAIError as api_err:
                        logger.error("API error (variant=%s, q=%s): %s", variant, question[:40], api_err)
                        responses.append(f"[ERROR: {api_err}]")
                        error_count += 1

                # Use first response as the canonical one; pass all for stability
                primary = responses[0] if responses else "[ERROR: no response]"

                result = evaluate_response(
                    prompt=prompt,
                    response=primary,
                    variant=variant,
                    question=question,
                    all_responses=responses if N_RUNS > 1 else None,
                )
                results.append(result)
                pbar.update(1)
                pbar.set_postfix({"variant": variant, "q": question[:25]})

    _save_results(results)

    summary = aggregate_results(results)
    logger.info("── Aggregate Results ──────────────────────────────────")
    for k, v in summary.items():
        logger.info("  %-32s %s", k, v)
    logger.info("────────────────────────────────────────────────────")

    if error_count:
        logger.warning("%d API error(s) occurred. Check logs above.", error_count)

    return results


def _save_results(results: list[dict]) -> None:
    """Save results to a timestamped file AND overwrite sample_results.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Timestamped archive — never overwritten
    ts_name = timestamped_filename("results", "json")
    ts_path = os.path.join(RESULTS_DIR, ts_name)
    save_json(results, ts_path)
    logger.info("Saved %d records → %s", len(results), ts_path)

    # sample_results.json — latest run for dashboard
    latest_path = os.path.join(RESULTS_DIR, "sample_results.json")
    save_json(results, latest_path)
    logger.info("Updated dashboard data → %s", latest_path)


if __name__ == "__main__":
    run_experiment()
