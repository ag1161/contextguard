"""
Main experiment runner for ContextGuard.

Iterates over all test cases × prompt variants, queries the LLM,
evaluates each response, writes results to results/sample_results.json,
and prints aggregate metrics.

Usage:
    python experiments/run_experiment.py
"""

import json
import sys
import os
from tqdm import tqdm

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm_client import query_llm
from app.prompts import PROMPT_VARIANTS
from app.evaluator import evaluate_response, aggregate_results
from app.utils import save_json

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "sample_results.json",
)

TEST_CASES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_cases.json"
)


def run_experiment(test_cases_path: str = TEST_CASES_PATH) -> list[dict]:
    """
    Run the full experiment grid and return all result records.

    Args:
        test_cases_path: Path to the JSON file with test questions.

    Returns:
        List of evaluated result dicts.
    """
    with open(test_cases_path, "r") as f:
        test_cases = json.load(f)

    results = []
    total = len(test_cases) * len(PROMPT_VARIANTS)

    print(f"\nRunning {len(test_cases)} questions × {len(PROMPT_VARIANTS)} variants = {total} evaluations\n")

    with tqdm(total=total, desc="Evaluating") as pbar:
        for case in test_cases:
            question = case["question"]

            for variant, template in PROMPT_VARIANTS.items():
                prompt = template.format(question=question)

                try:
                    response = query_llm(prompt)
                except Exception as e:
                    response = f"[ERROR: {e}]"

                result = evaluate_response(
                    prompt=prompt,
                    response=response,
                    variant=variant,
                    question=question,
                )
                results.append(result)
                pbar.update(1)
                pbar.set_postfix({"variant": variant, "q": question[:30]})

    save_json(results, RESULTS_PATH)

    summary = aggregate_results(results)
    print("\n── Aggregate Results ──────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print("────────────────────────────────────────────────────\n")

    return results


if __name__ == "__main__":
    run_experiment()
