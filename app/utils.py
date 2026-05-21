import json
import os
from datetime import datetime, timezone


def save_json(data: list | dict, path: str) -> None:
    """Save data to a JSON file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> list | dict:
    """Load data from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def timestamp() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    # datetime.utcnow() is deprecated since Python 3.12 — use timezone-aware form
    return datetime.now(timezone.utc).isoformat()


def timestamped_filename(base: str, ext: str) -> str:
    """
    Build a filename with a UTC timestamp suffix to avoid overwriting prior runs.

    Example:
        timestamped_filename("results", "json")
        → "results_20260521_120000.json"
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}.{ext}"
