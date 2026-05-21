"""
Unit tests for app/utils.py
"""

import pytest
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils import timestamp, timestamped_filename, save_json, load_json


class TestTimestamp:
    def test_returns_string(self):
        assert isinstance(timestamp(), str)

    def test_iso_format(self):
        ts = timestamp()
        # Should be parseable as ISO 8601 and contain timezone info
        assert "T" in ts
        assert "+" in ts or ts.endswith("Z")

    def test_two_calls_differ(self):
        import time
        t1 = timestamp()
        time.sleep(0.01)
        t2 = timestamp()
        assert t1 != t2


class TestTimestampedFilename:
    def test_returns_string(self):
        assert isinstance(timestamped_filename("results", "json"), str)

    def test_contains_base(self):
        fn = timestamped_filename("results", "json")
        assert fn.startswith("results_")

    def test_contains_extension(self):
        fn = timestamped_filename("results", "json")
        assert fn.endswith(".json")

    def test_unique_on_repeated_calls(self):
        import time
        f1 = timestamped_filename("x", "txt")
        time.sleep(1.01)
        f2 = timestamped_filename("x", "txt")
        assert f1 != f2


class TestSaveAndLoadJson:
    def test_round_trip(self):
        data = [{"key": "value", "num": 42}]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_json(data, path)
            loaded = load_json(path)
            assert loaded == data
        finally:
            os.unlink(path)

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.json")
            save_json({"a": 1}, path)
            assert os.path.exists(path)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_json("/tmp/definitely_does_not_exist_12345.json")
