"""Tests for OllamaVisionAnalyzer."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from workflow.vision import OllamaVisionAnalyzer, format_timestamp


def test_format_timestamp() -> None:
    assert format_timestamp(3661.5) == "01:01:01"


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

    def fake_chat(_url: str, payload: dict) -> dict:
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
            }
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


def test_analyze_skips_missing_frame_files(tmp_path: Path) -> None:
    missing = tmp_path / "frames" / "frame_0001.jpg"
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "frames.json").write_text(
        json.dumps([{"timestamp": 1.0, "path": str(missing), "trigger": "scene"}]),
        encoding="utf-8",
    )
    analyzer = OllamaVisionAnalyzer({"ollama": {}}, chat_fn=lambda *_a, **_k: {})

    assert analyzer.analyze([missing], tmp_path) == []
