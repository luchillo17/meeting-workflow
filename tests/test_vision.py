"""Tests for OllamaVisionAnalyzer."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from workflow.vision import OllamaVisionAnalyzer, build_vision_prompt, format_timestamp


def test_format_timestamp() -> None:
    assert format_timestamp(3661.5) == "01:01:01"


def test_build_vision_prompt_uses_output_language() -> None:
    prompt = build_vision_prompt(output_language="es")
    assert "Write the description in Spanish" in prompt


def test_analyze_writes_visual_content_json(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = frames_dir / "frame_0001.jpg"
    frame.write_bytes(b"jpeg-data")
    write_json = tmp_path / "frames.json"
    write_json.write_text(
        json.dumps([{"timestamp": 12.5, "path": str(frame), "trigger": "visual_cue"}]),
        encoding="utf-8",
    )

    def fake_chat(payload: dict) -> dict:
        assert payload["model"] == "qwen2.5vl:7b"
        images = payload["messages"][0]["images"]
        assert base64.b64encode(b"jpeg-data").decode() == images[0]
        return {
            "message": {
                "content": '{"type":"whiteboard","description":"Diagrama con tres bloques."}'
            }
        }

    analyzer = OllamaVisionAnalyzer(
        {
            "ollama": {
                "base_url": "http://localhost:11434",
                "vision_model": "qwen2.5vl:7b",
                "timeout_seconds": 30,
            },
            "output": {"language": "es"},
        },
        chat_fn=fake_chat,
    )

    result = analyzer.analyze([frame], tmp_path)

    assert result == [
        {
            "timestamp": "00:00:12",
            "type": "whiteboard",
            "description": "Diagrama con tres bloques.",
        }
    ]
    saved = json.loads((tmp_path / "visual_content.json").read_text(encoding="utf-8"))
    assert saved == result


def test_analyze_unloads_vision_model_when_configured(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = frames_dir / "frame_0001.jpg"
    frame.write_bytes(b"jpeg-data")
    (tmp_path / "frames.json").write_text(
        json.dumps([{"timestamp": 1.0, "path": str(frame), "trigger": "scene"}]),
        encoding="utf-8",
    )
    unloaded: list[str] = []

    analyzer = OllamaVisionAnalyzer(
        {"ollama": {"vision_model": "qwen2.5vl:7b", "unload_between_stages": True}},
        chat_fn=lambda _payload: {"message": {"content": '{"type":"other","description":"ok"}'}},
        unload_fn=unloaded.append,
    )

    analyzer.analyze([frame], tmp_path)

    assert unloaded == ["qwen2.5vl:7b"]


def test_analyze_skips_unload_when_unload_after_false(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = frames_dir / "frame_0001.jpg"
    frame.write_bytes(b"jpeg-data")
    (tmp_path / "frames.json").write_text(
        json.dumps([{"timestamp": 1.0, "path": str(frame), "trigger": "scene"}]),
        encoding="utf-8",
    )
    unloaded: list[str] = []

    analyzer = OllamaVisionAnalyzer(
        {"ollama": {"vision_model": "qwen2.5vl:7b", "unload_between_stages": True}},
        chat_fn=lambda _payload: {"message": {"content": '{"type":"other","description":"ok"}'}},
        unload_fn=unloaded.append,
    )

    analyzer.analyze([frame], tmp_path, unload_after=False)

    assert unloaded == []


def test_analyze_skips_low_value_participant_tile_descriptions(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = frames_dir / "frame_0001.jpg"
    frame.write_bytes(b"jpeg-data")
    (tmp_path / "frames.json").write_text(
        json.dumps([{"timestamp": 1.0, "path": str(frame), "trigger": "scene"}]),
        encoding="utf-8",
    )

    analyzer = OllamaVisionAnalyzer(
        {"ollama": {"unload_between_stages": False}},
        chat_fn=lambda _payload: {
            "message": {
                "content": (
                    '{"type":"other","description":'
                    '"La imagen muestra un fondo negro con burbuja circular read.ai meeting notes"}'
                )
            }
        },
    )

    assert analyzer.analyze([frame], tmp_path) == []
    saved = json.loads((tmp_path / "visual_content.json").read_text(encoding="utf-8"))
    assert saved == []


def test_analyze_skips_frame_after_empty_vision_responses(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = frames_dir / "frame_0001.jpg"
    frame.write_bytes(b"jpeg-data")
    (tmp_path / "frames.json").write_text(
        json.dumps([{"timestamp": 1.0, "path": str(frame), "trigger": "scene"}]),
        encoding="utf-8",
    )
    calls = {"n": 0}

    def empty_chat(_payload: dict) -> dict:
        calls["n"] += 1
        return {"message": {"content": ""}}

    analyzer = OllamaVisionAnalyzer(
        {"ollama": {"unload_between_stages": False}}, chat_fn=empty_chat
    )

    assert analyzer.analyze([frame], tmp_path) == []
    assert calls["n"] == 2


def test_analyze_retries_empty_vision_response_once(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = frames_dir / "frame_0001.jpg"
    frame.write_bytes(b"jpeg-data")
    (tmp_path / "frames.json").write_text(
        json.dumps([{"timestamp": 1.0, "path": str(frame), "trigger": "scene"}]),
        encoding="utf-8",
    )
    calls = {"n": 0}

    def flaky_chat(_payload: dict) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"message": {"content": ""}}
        return {"message": {"content": '{"type":"slide","description":"Tablero de tareas."}'}}

    analyzer = OllamaVisionAnalyzer(
        {"ollama": {"unload_between_stages": False}}, chat_fn=flaky_chat
    )

    result = analyzer.analyze([frame], tmp_path)

    assert calls["n"] == 2
    assert result[0]["type"] == "slide"


def test_analyze_skips_missing_frame_files(tmp_path: Path) -> None:
    missing = tmp_path / "frames" / "frame_0001.jpg"
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "frames.json").write_text(
        json.dumps([{"timestamp": 1.0, "path": str(missing), "trigger": "scene"}]),
        encoding="utf-8",
    )
    analyzer = OllamaVisionAnalyzer(
        {"ollama": {"unload_between_stages": False}}, chat_fn=lambda *_a, **_k: {}
    )

    assert analyzer.analyze([missing], tmp_path) == []


def test_analyze_removes_stale_visual_content_on_failure(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame = frames_dir / "frame_0001.jpg"
    frame.write_bytes(b"jpeg-data")
    (tmp_path / "frames.json").write_text(
        json.dumps([{"timestamp": 1.0, "path": str(frame), "trigger": "scene"}]),
        encoding="utf-8",
    )
    stale = tmp_path / "visual_content.json"
    stale.write_text('[{"timestamp": "00:00:00", "type": "other", "description": "stale"}]')

    def fail_chat(*_args: object, **_kwargs: object) -> dict:
        raise RuntimeError("vision failed")

    analyzer = OllamaVisionAnalyzer({"ollama": {"unload_between_stages": False}}, chat_fn=fail_chat)

    with pytest.raises(RuntimeError, match="vision failed"):
        analyzer.analyze([frame], tmp_path)

    assert not stale.exists()
    assert not (tmp_path / "extraction.json").exists()


def test_load_timestamps_tolerates_invalid_frames_json(tmp_path: Path) -> None:
    (tmp_path / "frames.json").write_text("{bad", encoding="utf-8")

    mapping = OllamaVisionAnalyzer._load_timestamps(tmp_path)

    assert mapping == {}
