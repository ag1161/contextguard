import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
N_RUNS = int(os.getenv("N_RUNS", "1"))  # runs per prompt (>1 enables stability_score)

# ── Guard: fail fast with a clear message if the API key is missing ────────────
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "\n\nOPENAI_API_KEY is not set.\n"
        "  1. Copy .env.example to .env\n"
        "  2. Add your OpenAI API key\n"
        "  3. Re-run the experiment\n"
    )
