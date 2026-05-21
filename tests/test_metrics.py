"""
Unit tests for app/metrics.py

Covers every public function with edge cases, boundary values,
and the case-sensitivity regression that was fixed (m.lower()).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.metrics import (
    hallucination_score,
    instruction_drift,
    stability_score,
    length_penalty,
    confidence_score,
    UNCERTAINTY_MARKERS,
    CONFIDENCE_SIGNALS,
    LENGTH_NORM,
)


# ── hallucination_score ────────────────────────────────────────────────────────

class TestHallucinationScore:
    def test_empty_response(self):
        assert hallucination_score("") == 0.0

    def test_no_markers(self):
        assert hallucination_score("Paris is the capital of France.") == 0.0

    def test_single_lowercase_marker(self):
        score = hallucination_score("this might be correct")
        assert score > 0.0

    def test_uppercase_i_markers_case_fix(self):
        """Regression: 'I think' and 'I believe' must match despite lowercasing."""
        score = hallucination_score("I think this is correct. I believe it too.")
        assert score > 0.0, "Case-insensitive markers 'I think'/'I believe' must match"

    def test_saturated_response_returns_1(self):
        """All UNCERTAINTY_MARKERS present → score == 1.0"""
        text = " ".join(UNCERTAINTY_MARKERS)
        assert hallucination_score(text) == 1.0

    def test_score_bounded_0_to_1(self):
        extreme = " ".join(UNCERTAINTY_MARKERS * 5)
        s = hallucination_score(extreme)
        assert 0.0 <= s <= 1.0

    def test_word_boundary_no_false_positive(self):
        # "mightily" should NOT trigger "might"
        assert hallucination_score("He acted mightily.") == 0.0

    def test_result_is_float(self):
        assert isinstance(hallucination_score("might"), float)


# ── instruction_drift ──────────────────────────────────────────────────────────

class TestInstructionDrift:
    def test_no_constraint(self):
        assert instruction_drift("Tell me about AI", "AI is fascinating.") == 0.5

    def test_one_sentence_satisfied(self):
        prompt = "Answer in 1 sentence."
        response = "Paris is the capital of France."
        assert instruction_drift(prompt, response) == 0.0

    def test_one_sentence_violated(self):
        prompt = "Answer in one sentence."
        response = "Paris is great. It is beautiful."
        assert instruction_drift(prompt, response) == 1.0

    def test_bullet_satisfied(self):
        prompt = "Give me a bullet list."
        response = "- Item one\n- Item two"
        assert instruction_drift(prompt, response) == 0.0

    def test_bullet_violated(self):
        prompt = "Give a list."
        response = "Item one. Item two. Item three."
        assert instruction_drift(prompt, response) == 0.8

    def test_empty_response_one_sentence(self):
        # No sentences after split → treated as 0 sentences → satisfied
        result = instruction_drift("Answer in 1 sentence.", "")
        assert result == 0.0

    def test_returns_float(self):
        assert isinstance(instruction_drift("hello", "world"), float)


# ── stability_score ────────────────────────────────────────────────────────────

class TestStabilityScore:
    def test_single_response_returns_zero(self):
        assert stability_score(["Only one response here."]) == 0.0

    def test_empty_list_returns_zero(self):
        assert stability_score([]) == 0.0

    def test_identical_responses_returns_zero(self):
        r = "The capital of France is Paris."
        assert stability_score([r, r, r]) == 0.0

    def test_very_different_responses(self):
        short = "Yes."
        long = " ".join(["word"] * 100)
        score = stability_score([short, long])
        assert score > 0.0

    def test_score_bounded_0_to_1(self):
        responses = ["a", "b " * 500, "c " * 200]
        s = stability_score(responses)
        assert 0.0 <= s <= 1.0

    def test_all_empty_responses(self):
        assert stability_score(["", ""]) == 0.0

    def test_returns_float(self):
        assert isinstance(stability_score(["a b", "c d e"]), float)


# ── length_penalty ─────────────────────────────────────────────────────────────

class TestLengthPenalty:
    def test_empty_string(self):
        assert length_penalty("") == 0.0

    def test_short_response(self):
        assert length_penalty("Paris.") < 1.0

    def test_at_norm_boundary(self):
        text = " ".join(["word"] * LENGTH_NORM)
        assert length_penalty(text) == 1.0

    def test_over_norm_capped_at_1(self):
        text = " ".join(["word"] * (LENGTH_NORM * 2))
        assert length_penalty(text) == 1.0

    def test_proportionality(self):
        half = " ".join(["word"] * (LENGTH_NORM // 2))
        assert abs(length_penalty(half) - 0.5) < 0.01

    def test_returns_float(self):
        assert isinstance(length_penalty("hello world"), float)


# ── confidence_score ───────────────────────────────────────────────────────────

class TestConfidenceScore:
    def test_empty_response(self):
        assert confidence_score("") == 0.0

    def test_no_signals(self):
        assert confidence_score("The weather is nice today.") == 0.0

    def test_single_signal(self):
        assert confidence_score("Clearly, Paris is the capital of France.") > 0.0

    def test_saturated_response(self):
        text = " ".join(CONFIDENCE_SIGNALS)
        assert confidence_score(text) == 1.0

    def test_score_bounded_0_to_1(self):
        text = " ".join(CONFIDENCE_SIGNALS * 3)
        s = confidence_score(text)
        assert 0.0 <= s <= 1.0

    def test_case_insensitive(self):
        assert confidence_score("CLEARLY this is right.") > 0.0

    def test_returns_float(self):
        assert isinstance(confidence_score("certainly"), float)
