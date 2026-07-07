"""Tests for Structured Extraction via Ollama."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow.extraction import (
    OllamaStructuredExtractor,
    build_extraction_prompt,
    extraction_has_language_drift,
    extraction_is_structurally_empty,
    extraction_json_schema,
    extraction_schema_keys,
    is_low_value_visual_description,
    merge_chapter_extractions_deterministic,
    normalize_extraction,
    sample_transcript_text,
    sanitize_visual_description,
    select_visual_for_extraction,
)
from workflow.runner import WorkflowRunner
from workflow.settings import Settings
from workflow.summary import render_summary
from workflow.utils import slugify, write_json


def test_build_extraction_prompt_includes_fidelity_rules() -> None:
    prompt = build_extraction_prompt("texto", [], output_language="es")

    assert "Fidelity rules" in prompt
    assert "Do not infer unstated" in prompt
    assert "Write all JSON string values in Spanish" in prompt


def test_build_extraction_prompt_uses_configured_output_language() -> None:
    prompt = build_extraction_prompt("hello", [], output_language="en")

    assert "Write all JSON string values in English" in prompt


def test_build_extraction_prompt_includes_transcript_and_visual() -> None:
    prompt = build_extraction_prompt(
        "Hablamos del tablero.",
        [{"timestamp": "00:01:00", "type": "whiteboard", "description": "Diagrama"}],
        output_language="es",
    )

    assert "Hablamos del tablero." in prompt
    assert "Diagrama" in prompt
    assert "00:01:00" in prompt
    assert "Source priority" in prompt


def test_sample_transcript_text_head_tail_preserves_ending() -> None:
    text = ("A" * 5_000) + ("B" * 5_000)
    sampled = sample_transcript_text(text, 6_000, sampling="head_tail", head_ratio=0.7)

    assert sampled.startswith("A")
    assert sampled.endswith("B")
    assert "middle omitted" in sampled


def test_select_visual_for_extraction_filters_tiles_and_caps() -> None:
    visual = [
        {
            "timestamp": "00:00:02",
            "type": "other",
            "description": "La imagen muestra un fondo negro con una burbuja circular read.ai",
        },
        {
            "timestamp": "00:08:37",
            "type": "whiteboard",
            "description": "Whiteboard con parámetros del servicio y convenio " + ("x" * 500),
        },
        {
            "timestamp": "00:20:51",
            "type": "whiteboard",
            "description": "Diagrama de convenio con tarifas",
        },
    ]

    selected = select_visual_for_extraction(visual, max_frames=1, max_description_chars=50)

    assert len(selected) == 1
    assert "convenio" in selected[0]["description"].lower()
    assert len(selected[0]["description"]) <= 53


def test_is_low_value_visual_description() -> None:
    assert is_low_value_visual_description("read.ai meeting notes con círculos CJ")
    assert not is_low_value_visual_description("Pantalla del navegador con parámetros del convenio")


def test_sanitize_visual_description_unwraps_json() -> None:
    raw = '{"type":"whiteboard","description":"Diagrama de convenio con tarifas"}'

    assert sanitize_visual_description(raw) == "Diagrama de convenio con tarifas"


def test_merge_chapter_extractions_deterministic_dedupes() -> None:
    from workflow.transcript_chapters import TranscriptChapter

    chapters = [
        (
            TranscriptChapter(1, 0.0, 10.0, "a", "chapters/chapter_001.txt"),
            {"topic": "A", "key_decisions": ["Acordamos X"], "action_items": []},
        ),
        (
            TranscriptChapter(2, 10.0, 20.0, "b", "chapters/chapter_002.txt"),
            {"topic": "B", "key_decisions": ["Acordamos X"], "action_items": []},
        ),
    ]

    merged = merge_chapter_extractions_deterministic(chapters)

    assert merged["topic"] == "A"
    assert merged["key_decisions"] == ["Acordamos X"]


def test_map_reduce_writes_chapter_artifacts(tmp_path: Path) -> None:
    segments = [
        {"start": float(i), "end": float(i + 1), "text": "palabra " * 500} for i in range(12)
    ]
    transcript = type(
        "T",
        (),
        {"text": " ".join(s["text"] for s in segments), "segments": segments},
    )()
    calls = {"n": 0}

    def fake_chat(_payload: dict) -> dict:
        calls["n"] += 1
        return {
            "message": {
                "content": json.dumps(
                    {
                        "topic": "t",
                        "key_decisions": [f"d{calls['n']}"],
                        "action_items": [],
                        "blockers_risks": [],
                        "status_updates": [],
                        "technical_details": [],
                        "open_questions": [],
                        "next_steps": [],
                    }
                )
            }
        }

    extractor = OllamaStructuredExtractor(
        {
            "ollama": {"unload_between_stages": False},
            "extraction": {
                "mode": "map_reduce",
                "chapter_trigger_chars": 1_000,
                "chapter_target_chars": 2_000,
                "grounding_check": False,
            },
        },
        chat_fn=fake_chat,
    )

    extractor.build_extraction(transcript, [], "meeting-20260101.mp4", tmp_path)

    assert (tmp_path / "chapters.json").is_file()
    assert (tmp_path / "extraction" / "chapters" / "chapter_001.json").is_file()
    assert calls["n"] >= 3


def test_extraction_has_language_drift_detects_english_in_spanish_mode() -> None:
    raw = {
        "topic": "Patient Portal",
        "key_decisions": ["The portal includes Home and Personal Information sections"],
        "action_items": [],
        "blockers_risks": [],
        "status_updates": [],
        "technical_details": [],
        "open_questions": [],
        "next_steps": [],
    }

    assert extraction_has_language_drift(raw, output_language="es")


def test_extraction_is_structurally_empty() -> None:
    assert extraction_is_structurally_empty(
        {
            "topic": "Tema",
            "key_decisions": [],
            "action_items": [],
            "blockers_risks": [],
            "status_updates": [],
            "technical_details": [],
            "open_questions": [],
            "next_steps": [],
        }
    )
    assert not extraction_is_structurally_empty(
        {"topic": "Tema", "key_decisions": ["Acordamos el piloto"]}
    )


def test_normalize_extraction_fills_schema_and_meeting_date() -> None:
    extraction = normalize_extraction(
        {
            "topic": "Plan piloto",
            "key_decisions": ["Aprobar MVP"],
            "action_items": [{"owner": "Ana", "task": "Probar workflow", "deadline": ""}],
        },
        meeting_date="2026-06-19",
        visual_content=[{"timestamp": "00:00:01", "type": "slide", "description": "Portada"}],
    )

    assert set(extraction) == set(extraction_schema_keys())
    assert extraction["meeting_date"] == "2026-06-19"
    assert extraction["topic"] == "Plan piloto"
    assert extraction["action_items"][0]["task"] == "Probar workflow"
    assert extraction["visual_content"][0]["description"] == "Portada"


def test_normalize_extraction_coerces_scalar_llm_fields() -> None:
    extraction = normalize_extraction(
        {
            "topic": None,
            "key_decisions": "Aprobar el piloto",
            "action_items": {"owner": "Ana", "task": "Enviar resumen", "deadline": ""},
            "next_steps": "Revisar en la próxima reunión",
        },
        meeting_date="2026-06-19",
        visual_content=[],
    )

    assert extraction["topic"] == "Reunión"
    assert extraction["key_decisions"] == ["Aprobar el piloto"]
    assert extraction["action_items"] == [
        {"owner": "Ana", "task": "Enviar resumen", "deadline": ""}
    ]
    assert extraction["next_steps"] == ["Revisar en la próxima reunión"]


def test_ollama_extractor_writes_extraction_and_summary(tmp_path: Path) -> None:
    transcript = type("T", (), {"text": "Definimos el roadmap.", "segments": []})()

    def fake_chat(payload: dict) -> dict:
        assert payload["model"] == "qwen2.5:7b"
        return {
            "message": {
                "content": json.dumps(
                    {
                        "topic": "Roadmap Q3",
                        "key_decisions": ["Priorizar extracción"],
                        "action_items": [],
                        "blockers_risks": [],
                        "status_updates": [],
                        "technical_details": [],
                        "open_questions": [],
                        "next_steps": ["Integrar Ollama"],
                    }
                )
            }
        }

    extractor = OllamaStructuredExtractor(
        {"ollama": {"base_url": "http://localhost:11434", "text_model": "qwen2.5:7b"}},
        chat_fn=fake_chat,
    )
    visual = [{"timestamp": "00:00:05", "type": "slide", "description": "Timeline"}]

    result = extractor.build_extraction(
        transcript,
        visual,
        "reunion-estrategica-20260619.mp4",
        tmp_path,
    )

    assert result["topic"] == "Roadmap Q3"
    assert result["meeting_date"] == "2026-06-19"
    saved = json.loads((tmp_path / "extraction.json").read_text(encoding="utf-8"))
    assert saved["key_decisions"] == ["Priorizar extracción"]
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Roadmap Q3" in summary
    assert "Priorizar extracción" in summary
    assert "Timeline" in summary


def test_ollama_extractor_raises_on_invalid_json(tmp_path: Path) -> None:
    transcript = type("T", (), {"text": "hola", "segments": []})()
    extractor = OllamaStructuredExtractor(
        {"ollama": {"unload_between_stages": False}},
        chat_fn=lambda *_a, **_k: {"message": {"content": "no json here"}},
    )

    with pytest.raises(RuntimeError, match="Empty extraction response"):
        extractor.build_extraction(transcript, [], "meeting.mp4", tmp_path)


def test_ollama_extractor_requests_json_schema_format(tmp_path: Path) -> None:
    transcript = type("T", (), {"text": "hola", "segments": []})()
    captured: dict = {}

    def fake_chat(payload: dict) -> dict:
        captured.update(payload)
        return {
            "message": {
                "content": json.dumps(
                    {
                        "topic": "Test",
                        "key_decisions": [],
                        "action_items": [],
                        "blockers_risks": [],
                        "status_updates": [],
                        "technical_details": [],
                        "open_questions": [],
                        "next_steps": [],
                    }
                )
            }
        }

    extractor = OllamaStructuredExtractor(
        {"ollama": {"unload_between_stages": False}}, chat_fn=fake_chat
    )
    extractor.build_extraction(transcript, [], "meeting.mp4", tmp_path)

    assert captured.get("format", {}).get("required") == list(extraction_json_schema()["required"])
    assert captured["messages"][0]["role"] == "system"


def test_runner_integration_with_ollama_extractor(tmp_path: Path) -> None:
    recording = tmp_path / "reunion-estrategica-20260619.mp4"
    recording.write_bytes(b"fake-video")
    settings = Settings(
        output_dir=tmp_path / "output",
        config={},
        ollama_base_url="http://localhost:11434",
        ollama_text_model="qwen2.5:7b",
        ollama_vision_model="qwen2.5vl:7b",
    )

    class FakeTranscriber:
        def transcribe(self, rec: Path, output_dir: Path, *, unload_after: bool = True) -> object:
            result = type("T", (), {"segments": [], "text": "Acordamos el piloto."})()
            write_json(output_dir / "transcript.json", {"segments": []})
            (output_dir / "transcript.txt").write_text("Acordamos el piloto.", encoding="utf-8")
            return result

    class FakeFrames:
        def extract(self, rec: Path, transcript: object, output_dir: Path) -> list[Path]:
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            path = frames_dir / "frame_0001.jpg"
            path.write_bytes(b"jpeg")
            write_json(
                output_dir / "frames.json",
                [{"timestamp": 1.0, "path": str(path), "trigger": "scene"}],
            )
            return [path]

    class FakeVision:
        def analyze(
            self, frame_paths: list[Path], output_dir: Path, *, unload_after: bool = True
        ) -> list[dict]:
            visual = [{"timestamp": "00:00:01", "type": "slide", "description": "Agenda"}]
            write_json(output_dir / "visual_content.json", visual)
            return visual

    def fake_chat(payload: dict) -> dict:
        return {
            "message": {
                "content": json.dumps(
                    {
                        "topic": "Piloto GPU",
                        "key_decisions": ["Correr en Windows"],
                        "action_items": [
                            {"owner": "Carlos", "task": "Validar extracción", "deadline": ""}
                        ],
                        "blockers_risks": [],
                        "status_updates": [],
                        "technical_details": [],
                        "open_questions": [],
                        "next_steps": [],
                    }
                )
            }
        }

    extractor = OllamaStructuredExtractor(
        {"ollama": {"text_model": "qwen2.5:7b"}},
        chat_fn=fake_chat,
    )
    runner = WorkflowRunner(
        settings,
        FakeTranscriber(),
        FakeFrames(),
        FakeVision(),
        extractor,
    )

    out = runner.run(recording)
    extraction = json.loads((out / "extraction.json").read_text(encoding="utf-8"))

    assert out == settings.output_dir / slugify(recording.name)
    assert extraction["meeting_date"] == "2026-06-19"
    assert extraction["topic"] == "Piloto GPU"
    assert extraction["visual_content"][0]["description"] == "Agenda"
    assert "Piloto GPU" in (out / "summary.md").read_text(encoding="utf-8")


def test_render_summary_matches_extraction_facts() -> None:
    extraction = normalize_extraction(
        {
            "topic": "Sync semanal",
            "key_decisions": ["Mantener español"],
            "action_items": [{"owner": "Luis", "task": "Revisar PR", "deadline": "viernes"}],
            "blockers_risks": ["GPU ocupada"],
            "status_updates": [],
            "technical_details": [],
            "open_questions": [],
            "next_steps": ["Merge"],
        },
        meeting_date="2026-06-19",
        visual_content=[],
    )

    summary = render_summary(extraction)

    assert "Sync semanal" in summary
    assert "Mantener español" in summary
    assert "Luis" in summary
    assert "Revisar PR" in summary
    assert "GPU ocupada" in summary
    assert "Merge" in summary
