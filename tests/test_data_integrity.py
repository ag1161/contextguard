"""
Data integrity tests for results/sample_results.json

Validates schema, value ranges, and expected record count.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "sample_results.json"
)

REQUIRED_FIELDS = {
    "timestamp", "question", "variant", "prompt",
    "response", "hallucination_score", "instruction_drift",
    "length_penalty", "confidence_score", "word_count",
}

VALID_VARIANTS = {"baseline", "adversarial", "ambiguous", "overconstrained"}

SCORE_FIELDS = ["hallucination_score", "instruction_drift", "length_penalty", "confidence_score"]


@pytest.fixture(scope="module")
def results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


class TestDataIntegrity:
    def test_file_exists(self):
        assert os.path.exists(RESULTS_PATH), f"Missing: {RESULTS_PATH}"

    def test_is_list(self, results):
        assert isinstance(results, list)

    def test_has_records(self, results):
        assert len(results) >= 4, "Expected at least 4 records"

    def test_all_required_fields_present(self, results):
        for i, record in enumerate(results):
            missing = REQUIRED_FIELDS - set(record.keys())
            assert not missing, f"Record {i} missing fields: {missing}"

    def test_variants_are_valid(self, results):
        for i, record in enumerate(results):
            assert record["variant"] in VALID_VARIANTS, \
                f"Record {i} has invalid variant: {record['variant']}"

    def test_scores_in_range(self, results):
        for i, record in enumerate(results):
            for field in SCORE_FIELDS:
                val = record[field]
                assert isinstance(val, (int, float)), f"Record {i}.{field} not numeric"
                assert 0.0 <= val <= 1.0, f"Record {i}.{field} = {val} out of [0,1]"

    def test_word_count_positive(self, results):
        for i, record in enumerate(results):
            assert record["word_count"] >= 0, f"Record {i} negative word_count"

    def test_responses_not_error(self, results):
        errors = [r for r in results if str(r["response"]).startswith("[ERROR:")]
        assert len(errors) == 0, f"{len(errors)} error records found in sample_results.json"

    def test_timestamps_are_strings(self, results):
        for i, record in enumerate(results):
            assert isinstance(record["timestamp"], str), f"Record {i} timestamp not a string"
