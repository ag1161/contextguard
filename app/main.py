"""
ContextGuard — entry point for running experiments from the command line.

Usage:
    python -m app.main
"""

from experiments.run_experiment import run_experiment

if __name__ == "__main__":
    print("=" * 60)
    print("  ContextGuard: LLM Reliability Evaluation Framework")
    print("=" * 60)
    run_experiment()
