import json
import os
from datetime import datetime


def save_json(data: list | dict, path: str) -> None:
    """Save data to a JSON file, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data) if isinstance(data, list) else 1} records to {path}")


def load_json(path: str) -> list | dict:
    """Load data from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def timestamp() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.utcnow().isoformat()


def flatten_results(results: list[dict], key: str) -> list:
    """Extract a single field from a list of result dicts."""
    return [r[key] for r in results if key in r]
