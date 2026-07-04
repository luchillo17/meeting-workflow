"""Tests for WhisperTranscriber adapter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from workflow.transcriber import WhisperTranscriber


class FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class FakeWhisperModel:
    def transcribe(self, audio_path: str, language: str | None = None):
        segments = [
            FakeSegment(0.0, 2.5, "Hola, revisemos el tablero."),
            FakeSegment(2.5, 5.0, "Tenemos tres bloques en el diagrama."),
        ]
        return segments, object()


def _fake_run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    audio_path = Path(cmd[-1])
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake-wav")
    return subprocess.CompletedProcess(cmd, 0, "", "")


def _transcriber(
    *,
    run_cmd=_fake_run_cmd,
    model_factory=lambda *_a, **_k: FakeWhisperModel(),
) -> WhisperTranscriber:
    return WhisperTranscriber(
        {"whisper": {"model": "tiny", "device": "cpu", "compute_type": "int8", "language": "es"}},
        run_cmd_fn=run_cmd,
        model_factory=model_factory,
    )


def test_transcribe_writes_segmented_json_and_text(tmp_path: Path) -> None:
    recording = tmp_path / "reunion-20260619.mp4"
    recording.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"

    result = _transcriber().transcribe(recording, output_dir)

    assert result.text == "Hola, revisemos el tablero. Tenemos tres bloques en el diagrama."
    assert len(result.segments) == 2
    assert result.segments[0]["text"] == "Hola, revisemos el tablero."

    payload = json.loads((output_dir / "transcript.json").read_text(encoding="utf-8"))
    assert payload["segments"] == result.segments
    assert (output_dir / "transcript.txt").read_text(encoding="utf-8") == result.text


def test_transcribe_extracts_mono_audio_via_ffmpeg(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    captured: list[list[str]] = []

    def capture_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return _fake_run_cmd(cmd)

    _transcriber(run_cmd=capture_cmd).transcribe(recording, output_dir)

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == "ffmpeg"
    assert "-vn" in cmd
    assert "-ac" in cmd and "1" in cmd
    assert "-ar" in cmd and "16000" in cmd
    assert str(output_dir / "audio.wav") in cmd


def test_audio_extraction_failure_surfaces_clear_error(tmp_path: Path) -> None:
    recording = tmp_path / "broken.mp4"
    recording.write_bytes(b"fake-video")

    def fail_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("Command failed (1): ffmpeg ...\nstderr: Invalid data")

    transcriber = _transcriber(run_cmd=fail_cmd)

    with pytest.raises(RuntimeError, match="Audio extraction failed for broken.mp4"):
        transcriber.transcribe(recording, tmp_path / "out")


def test_cpu_device_downgrades_float16_compute_type() -> None:
    transcriber = WhisperTranscriber({"whisper": {"device": "cpu", "compute_type": "float16"}})
    assert transcriber._effective_compute_type() == "int8"


def test_whisper_model_unloaded_after_transcription(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    created_models: list[FakeWhisperModel] = []

    def factory(*_args, **_kwargs) -> FakeWhisperModel:
        model = FakeWhisperModel()
        created_models.append(model)
        return model

    transcriber = _transcriber(model_factory=factory)
    transcriber.transcribe(recording, tmp_path / "out")

    assert transcriber._model is None
    assert len(created_models) == 1
