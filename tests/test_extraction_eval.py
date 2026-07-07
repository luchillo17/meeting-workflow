"""Tests for extraction quality spot-checks."""

from __future__ import annotations

from pathlib import Path

from workflow.extraction_eval import (
    evaluate_extraction,
    evaluate_output_dir,
    load_chapter_count,
    load_transcript_chars,
    load_transcript_text,
    load_visual_frame_count,
)
from workflow.utils import write_json


def test_load_transcript_text_falls_back_to_transcript_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    write_json(
        output_dir / "transcript.json",
        {"segments": [{"text": "Hablamos del portal y el POC."}]},
    )
    assert "portal" in load_transcript_text(output_dir)
    assert load_transcript_chars(output_dir) > 0


def test_evaluate_extraction_passes_when_requirements_met() -> None:
    extraction = {
        "topic": "Convenio en Vital Link",
        "key_decisions": ["Incluir campo convenio", "Validar IPS"],
        "action_items": [
            {"owner": "Ana", "task": "Revisar convenio", "deadline": ""},
            {"owner": "Luis", "task": "Probar IPS", "deadline": ""},
            {"owner": "", "task": "Documentar", "deadline": ""},
        ],
    }
    checks = {
        "min_transcript_chars": 1_000,
        "min_decisions": 2,
        "min_action_items": 3,
        "min_chapters": 2,
        "must_contain": ("convenio",),
        "transcript_terms": ("convenio",),
    }

    assert (
        evaluate_extraction(
            extraction,
            checks=checks,
            transcript_chars=20_000,
            transcript_text="El convenio con las IPS es clave.",
            chapter_count=4,
            visual_frame_count=3,
        )
        == []
    )


def test_evaluate_extraction_reports_missing_transcript_terms() -> None:
    extraction = {
        "topic": "Reunión general",
        "key_decisions": ["Seguimos mañana", "Otro punto"],
        "action_items": [{"owner": "", "task": "Tarea", "deadline": ""}],
    }
    checks = {
        "min_transcript_chars": 1_000,
        "min_decisions": 1,
        "transcript_terms": ("portal",),
    }

    failures = evaluate_extraction(
        extraction,
        checks=checks,
        transcript_chars=5_000,
        transcript_text="Ajustar el portal responsive.",
    )

    assert any("portal" in failure for failure in failures)


def test_evaluate_extraction_requires_chapters_for_long_meetings() -> None:
    extraction = {"topic": "Long meeting", "key_decisions": ["a", "b"], "action_items": []}
    checks = {"min_transcript_chars": 1_000, "min_chapters": 4}

    failures = evaluate_extraction(
        extraction,
        checks=checks,
        transcript_chars=15_000,
        chapter_count=1,
    )

    assert any("chapters" in failure for failure in failures)


def test_evaluate_output_dir_reads_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "revision-avances-mvp-demo"
    output_dir.mkdir()
    write_json(
        output_dir / "extraction.json",
        {
            "topic": "Avances del MVP",
            "key_decisions": ["Uno", "Dos"],
            "action_items": [
                {"owner": "", "task": "A", "deadline": ""},
                {"owner": "", "task": "B", "deadline": ""},
            ],
        },
    )
    (output_dir / "transcript.txt").write_text("x" * 45_000 + " mvp", encoding="utf-8")
    write_json(output_dir / "chapters.json", [{"index": i} for i in range(5)])
    write_json(output_dir / "visual_content.json", [{"description": "slide"}] * 4)

    failures = evaluate_output_dir(
        output_dir,
        checks={
            "min_transcript_chars": 40_000,
            "min_decisions": 2,
            "min_action_items": 2,
            "min_chapters": 4,
            "min_visual_frames": 3,
            "transcript_terms": ("mvp",),
        },
    )
    assert failures == []


def test_load_chapter_and_visual_counts(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    write_json(output_dir / "chapters.json", [{"index": 1}, {"index": 2}])
    write_json(output_dir / "visual_content.json", [{"x": 1}])
    assert load_chapter_count(output_dir) == 2
    assert load_visual_frame_count(output_dir) == 1


def test_evaluate_extraction_grounding_ratio_failure() -> None:
    extraction = {
        "topic": "Portal paciente",
        "key_decisions": ["Priorizar portal responsivo"],
        "action_items": [{"owner": "", "task": "Inventar blockchain", "deadline": ""}],
        "blockers_risks": [],
        "status_updates": [],
        "technical_details": [],
        "open_questions": [],
        "next_steps": [],
    }
    transcript = "Priorizamos el portal responsivo y el menú hamburguesa." * 30

    failures = evaluate_extraction(
        extraction,
        checks={"min_transcript_chars": 1_000, "min_grounding_ratio": 0.75},
        transcript_chars=len(transcript),
        transcript_text=transcript,
    )

    assert any("grounding ratio" in message for message in failures)
