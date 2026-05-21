"""
ContextGuard — entry point for running experiments from the command line.

Usage:
    python -m app.main
"""

import logging
from experiments.run_experiment import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    print("=" * 60)
    print("  ContextGuard: LLM Reliability Evaluation Framework")
    print("=" * 60)
    run_experiment()
