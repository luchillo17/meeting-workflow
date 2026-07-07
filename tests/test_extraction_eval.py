"""Tests for extraction quality spot-checks."""

from __future__ import annotations

from workflow.extraction_eval import evaluate_extraction


def test_evaluate_extraction_passes_when_requirements_met() -> None:
    extraction = {
        "topic": "Convenio en Vital Link",
        "key_decisions": ["Incluir campo convenio"],
        "action_items": [],
    }
    checks = {
        "min_transcript_chars": 1_000,
        "min_decisions": 1,
        "must_contain": ("convenio",),
    }

    assert evaluate_extraction(extraction, checks=checks, transcript_chars=5_000) == []


def test_evaluate_extraction_reports_missing_terms() -> None:
    extraction = {
        "topic": "Reunión general",
        "key_decisions": ["Seguimos mañana"],
        "action_items": [],
    }
    checks = {
        "min_transcript_chars": 1_000,
        "min_decisions": 1,
        "must_contain": ("convenio",),
    }

    failures = evaluate_extraction(extraction, checks=checks, transcript_chars=5_000)

    assert any("convenio" in failure for failure in failures)
