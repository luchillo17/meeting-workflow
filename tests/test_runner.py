"""Tests for WorkflowRunner at the agreed seam."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from workflow.runner import WorkflowRunner
from workflow.settings import Settings
from workflow.summary import render_summary
from workflow.utils import slugify, write_json


@dataclass
class FakeTranscript:
    segments: list[dict]
    text: str


class CountingTranscriber:
    def __init__(self) -> None:
        self.calls = 0
        self.unloads = 0

    def transcribe(
        self, recording: Path, output_dir: Path, *, unload_after: bool = True
    ) -> FakeTranscript:
        self.calls += 1
        if unload_after and hasattr(self, "unload"):
            self.unload()
        result = FakeTranscript(segments=[{"start": 0, "end": 1, "text": "test"}], text="test")
        write_json(output_dir / "transcript.json", {"segments": result.segments})
        (output_dir / "transcript.txt").write_text("test", encoding="utf-8")
        return result

    def unload(self) -> None:
        self.unloads += 1


class CountingFrameExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, recording: Path, transcript: FakeTranscript, output_dir: Path) -> list[Path]:
        self.calls += 1
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        path = frames_dir / "f.jpg"
        path.write_bytes(b"x")
        return [path]


class CountingVision:
    def __init__(self) -> None:
        self.calls = 0
        self.unloads = 0

    def analyze(
        self, frame_paths: list[Path], output_dir: Path, *, unload_after: bool = True
    ) -> list[dict]:
        self.calls += 1
        if unload_after:
            self.unload()
        visual = [{"timestamp": "00:00:00", "type": "slide", "description": "Test slide"}]
        write_json(output_dir / "visual_content.json", visual)
        return visual

    def unload(self) -> None:
        self.unloads += 1


class CountingExtractor:
    def __init__(self) -> None:
        self.calls = 0
        self.unloads = 0

    def build_extraction(
        self,
        transcript,
        visual_content,
        recording_name: str,
        output_dir: Path,
        *,
        unload_after: bool = True,
    ) -> dict:
        self.calls += 1
        if unload_after:
            self.unload()
        extraction = {
            "meeting_date": "2026-06-19",
            "topic": "Test",
            "key_decisions": ["decide"],
            "action_items": [],
            "blockers_risks": [],
            "status_updates": [],
            "technical_details": [],
            "open_questions": [],
            "next_steps": [],
            "visual_content": visual_content,
        }
        write_json(output_dir / "extraction.json", extraction)
        (output_dir / "summary.md").write_text(render_summary(extraction), encoding="utf-8")
        return extraction

    def unload(self) -> None:
        self.unloads += 1


def _runner(
    tmp_path: Path, recording: Path
) -> tuple[WorkflowRunner, CountingTranscriber, Settings]:
    settings = Settings(
        output_dir=tmp_path / "output",
        config={},
        ollama_base_url="http://localhost:11434",
        ollama_text_model="qwen2.5:7b",
        ollama_vision_model="qwen2.5vl:7b",
    )
    transcriber = CountingTranscriber()
    frames = CountingFrameExtractor()
    vision = CountingVision()
    extractor = CountingExtractor()
    runner = WorkflowRunner(settings, transcriber, frames, vision, extractor)
    return runner, transcriber, settings


def test_run_writes_complete_extraction_bundle(tmp_path: Path) -> None:
    recording = tmp_path / "reunion-estrategica-20260619.mp4"
    recording.write_bytes(b"fake-video")
    runner, _, settings = _runner(tmp_path, recording)

    out = runner.run(recording)

    assert out == settings.output_dir / slugify(recording.name)
    assert (out / "transcript.json").exists()
    assert (out / "transcript.txt").exists()
    assert (out / "visual_content.json").exists()
    assert (out / "extraction.json").exists()
    assert (out / "summary.md").exists()
    assert "Test" in (out / "summary.md").read_text(encoding="utf-8")


def test_skip_when_extraction_exists(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    runner, transcriber, settings = _runner(tmp_path, recording)
    out_dir = settings.output_dir / slugify(recording.name)
    out_dir.mkdir(parents=True)
    write_json(out_dir / "extraction.json", {"topic": "existing"})

    runner.run(recording)

    assert transcriber.calls == 0


def test_force_reprocesses(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    runner, transcriber, settings = _runner(tmp_path, recording)
    out_dir = settings.output_dir / slugify(recording.name)
    out_dir.mkdir(parents=True)
    write_json(out_dir / "extraction.json", {"topic": "old"})

    runner.run(recording, force=True)

    assert transcriber.calls == 1
    assert (out_dir / "summary.md").exists()


def test_missing_recording_raises(tmp_path: Path) -> None:
    recording = tmp_path / "missing.mp4"
    runner, _, _ = _runner(tmp_path, recording)

    with pytest.raises(FileNotFoundError):
        runner.run(recording)


def test_slugify_and_meeting_date_from_filename() -> None:
    name = "Reunión Estrategica Vital Link-20260619.mp4"
    from workflow.utils import parse_meeting_date

    assert parse_meeting_date(name) == "2026-06-19"
    assert "20260619" in slugify(name)


def test_run_batch_processes_each_stage_once_per_recording(tmp_path: Path) -> None:
    recordings = [
        tmp_path / "meeting-a.mp4",
        tmp_path / "meeting-b.mp4",
    ]
    for recording in recordings:
        recording.write_bytes(b"fake-video")

    settings = Settings(
        output_dir=tmp_path / "output",
        config={},
        ollama_base_url="http://localhost:11434",
        ollama_text_model="qwen2.5:7b",
        ollama_vision_model="qwen2.5vl:7b",
    )
    transcriber = CountingTranscriber()
    frames = CountingFrameExtractor()
    vision = CountingVision()
    extractor = CountingExtractor()
    runner = WorkflowRunner(settings, transcriber, frames, vision, extractor)

    outputs = runner.run_batch(recordings)

    assert len(outputs) == 2
    assert transcriber.calls == 2
    assert frames.calls == 2
    assert vision.calls == 2
    assert extractor.calls == 2
    assert transcriber.unloads == 1
    assert vision.unloads == 1
    assert extractor.unloads == 1
    for out in outputs:
        assert (out / "extraction.json").exists()


def test_run_batch_skips_completed_extractions(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    runner, transcriber, settings = _runner(tmp_path, recording)
    out_dir = settings.output_dir / slugify(recording.name)
    out_dir.mkdir(parents=True)
    write_json(out_dir / "extraction.json", {"topic": "existing"})

    results = runner.run_batch([recording])

    assert results == []
    assert transcriber.calls == 0


def test_run_batch_reuses_transcript_and_frames_when_present(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    settings = Settings(
        output_dir=tmp_path / "output",
        config={},
        ollama_base_url="http://localhost:11434",
        ollama_text_model="qwen2.5:7b",
        ollama_vision_model="qwen2.5vl:7b",
    )
    transcriber = CountingTranscriber()
    frames = CountingFrameExtractor()
    vision = CountingVision()
    extractor = CountingExtractor()
    runner = WorkflowRunner(settings, transcriber, frames, vision, extractor)

    out_dir = settings.output_dir / slugify(recording.name)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame = frames_dir / "frame_0001.jpg"
    frame.write_bytes(b"x")
    write_json(out_dir / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "hola"}]})
    write_json(
        out_dir / "frames.json",
        [{"timestamp": 0.0, "path": str(frame), "trigger": "scene"}],
    )

    outputs = runner.run_batch([recording])

    assert len(outputs) == 1
    assert transcriber.calls == 0
    assert frames.calls == 0
    assert vision.calls == 1
    assert extractor.calls == 1


def test_run_batch_reuses_vision_when_present(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    settings = Settings(
        output_dir=tmp_path / "output",
        config={},
        ollama_base_url="http://localhost:11434",
        ollama_text_model="qwen2.5:7b",
        ollama_vision_model="qwen2.5vl:7b",
    )
    transcriber = CountingTranscriber()
    frames = CountingFrameExtractor()
    vision = CountingVision()
    extractor = CountingExtractor()
    runner = WorkflowRunner(settings, transcriber, frames, vision, extractor)

    out_dir = settings.output_dir / slugify(recording.name)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame = frames_dir / "frame_0001.jpg"
    frame.write_bytes(b"x")
    write_json(out_dir / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "hola"}]})
    write_json(
        out_dir / "frames.json",
        [{"timestamp": 0.0, "path": str(frame), "trigger": "scene"}],
    )
    write_json(
        out_dir / "visual_content.json",
        [{"timestamp": "00:00:00", "type": "slide", "description": "cached"}],
    )

    outputs = runner.run_batch([recording])

    assert len(outputs) == 1
    assert transcriber.calls == 0
    assert frames.calls == 0
    assert vision.calls == 0
    assert extractor.calls == 1


def test_run_batch_extract_only_reruns_extraction_without_upstream(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    settings = Settings(
        output_dir=tmp_path / "output",
        config={},
        ollama_base_url="http://localhost:11434",
        ollama_text_model="qwen2.5:7b",
        ollama_vision_model="qwen2.5vl:7b",
    )
    transcriber = CountingTranscriber()
    frames = CountingFrameExtractor()
    vision = CountingVision()
    extractor = CountingExtractor()
    runner = WorkflowRunner(settings, transcriber, frames, vision, extractor)

    out_dir = settings.output_dir / slugify(recording.name)
    out_dir.mkdir(parents=True)
    write_json(out_dir / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "hola"}]})
    write_json(
        out_dir / "visual_content.json", [{"timestamp": "00:00:00", "description": "cached"}]
    )
    write_json(out_dir / "extraction.json", {"topic": "stale"})

    outputs = runner.run_batch([recording], extract_only=True)

    assert len(outputs) == 1
    assert transcriber.calls == 0
    assert frames.calls == 0
    assert vision.calls == 0
    assert extractor.calls == 1
    assert (out_dir / "visual_content.json").exists()


def test_run_batch_vision_only_reruns_vision_and_extraction(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    settings = Settings(
        output_dir=tmp_path / "output",
        config={},
        ollama_base_url="http://localhost:11434",
        ollama_text_model="qwen2.5:7b",
        ollama_vision_model="qwen2.5vl:7b",
    )
    transcriber = CountingTranscriber()
    frames = CountingFrameExtractor()
    vision = CountingVision()
    extractor = CountingExtractor()
    runner = WorkflowRunner(settings, transcriber, frames, vision, extractor)

    out_dir = settings.output_dir / slugify(recording.name)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame_path = frames_dir / "frame_0001.jpg"
    frame_path.write_bytes(b"fake-jpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "hola"}]})
    write_json(
        out_dir / "frames.json",
        [{"timestamp": 0.0, "path": str(frame_path), "trigger": "scene"}],
    )
    write_json(out_dir / "visual_content.json", [{"timestamp": "00:00:00", "description": "stale"}])
    write_json(out_dir / "extraction.json", {"topic": "stale"})

    outputs = runner.run_batch([recording], vision_only=True)

    assert len(outputs) == 1
    assert transcriber.calls == 0
    assert frames.calls == 0
    assert vision.calls == 1
    assert extractor.calls == 1
