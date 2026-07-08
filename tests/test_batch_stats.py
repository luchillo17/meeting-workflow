"""Tests for batch run statistics (ADR 0010)."""

from __future__ import annotations

import json
from pathlib import Path

from workflow.batch_stats import (
    BatchRunRecorder,
    collect_recording_stats,
    load_run_history,
    median_stage_seconds_per_audio_minute,
    persist_run_record,
)
from workflow.settings import Settings
from workflow.utils import write_json


class _FakeJob:
    def __init__(self, recording_name: str, output_dir: Path) -> None:
        self.recording = type("R", (), {"name": recording_name})()
        self.output_dir = output_dir


def test_persist_run_record_writes_json_and_jsonl(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    record = {"run_id": "20260708T000000Z-abc", "status": "completed", "recordings": []}
    path = persist_run_record(output_root, record)
    assert path.is_file()
    jsonl = output_root / "batch-runs.jsonl"
    assert jsonl.is_file()
    assert "abc" in jsonl.read_text(encoding="utf-8")


def test_batch_run_recorder_collects_stage_timings(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    job_dir = output_root / "meeting-20260514"
    job_dir.mkdir(parents=True)
    write_json(
        job_dir / "transcript.json",
        {"segments": [{"start": 0, "end": 120, "text": "hola"}]},
    )
    write_json(
        job_dir / "extraction.json",
        {
            "topic": "Test",
            "key_decisions": ["decide"],
            "action_items": [],
            "blockers_risks": [],
            "status_updates": [],
            "technical_details": [],
            "open_questions": [],
            "next_steps": [],
        },
    )
    settings = Settings(
        output_dir=output_root,
        config={"whisper": {"model": "large-v3"}, "diarization": {"enabled": True}},
        ollama_base_url="http://localhost:11434",
        ollama_text_model="qwen3.5:9b",
        ollama_vision_model="qwen3.5:9b",
    )
    recorder = BatchRunRecorder(
        output_root,
        settings,
        [Path("meeting.mp4")],
    )
    recorder.add_stage_seconds(job_dir.name, "transcript", 42.5)
    recorder.mark_recording_completed(job_dir.name)
    job = _FakeJob("meeting.mp4", job_dir)
    run_path = recorder.finalize([job], status="completed", steps_completed=4, steps_total=4)
    data = json.loads(run_path.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["aggregates"]["recordings_completed"] == 1
    assert data["recordings"][0]["stage_seconds"]["transcript"] == 42.5
    assert data["recordings"][0]["audio_duration_seconds"] == 120.0


def test_median_stage_seconds_per_audio_minute(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    record = {
        "run_id": "run-1",
        "status": "completed",
        "recordings": [
            {
                "audio_duration_seconds": 60.0,
                "stage_seconds": {"transcript": 30.0},
            },
            {
                "audio_duration_seconds": 120.0,
                "stage_seconds": {"transcript": 60.0},
            },
        ],
    }
    persist_run_record(output_root, record)
    median = median_stage_seconds_per_audio_minute(output_root, stage="transcript")
    assert median == 30.0


def test_collect_recording_stats_counts_extraction_shape(tmp_path: Path) -> None:
    output_dir = tmp_path / "slug"
    output_dir.mkdir()
    write_json(
        output_dir / "extraction.json",
        {
            "topic": "T",
            "key_decisions": ["a", "b"],
            "action_items": [{"owner": "", "task": "x", "deadline": ""}],
            "blockers_risks": [],
            "status_updates": [],
            "technical_details": [],
            "open_questions": [],
            "next_steps": [],
        },
    )
    (output_dir / "transcript.txt").write_text("convenio ips portal", encoding="utf-8")
    stats = collect_recording_stats(
        output_dir,
        source_filename="m.mp4",
        stage_seconds={"extraction": 1.0},
        config={"extraction": {"grounding_min_overlap": 0.34}},
    )
    assert stats["counts"]["key_decisions"] == 2
    assert stats["counts"]["action_items"] == 1


def test_load_run_history_reads_jsonl(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    persist_run_record(output_root, {"run_id": "a", "status": "completed"})
    persist_run_record(output_root, {"run_id": "b", "status": "completed"})
    history = load_run_history(output_root, limit=5)
    assert [row["run_id"] for row in history] == ["a", "b"]


def test_batch_run_recorder_counts_only_marked_completions(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    settings = Settings(
        output_dir=output_root,
        config={},
        ollama_base_url="http://localhost:11434",
        ollama_text_model="qwen3.5:9b",
        ollama_vision_model="qwen3.5:9b",
    )
    recorder = BatchRunRecorder(output_root, settings, [Path("a.mp4"), Path("b.mp4")])
    done_dir = output_root / "done"
    done_dir.mkdir(parents=True)
    write_json(done_dir / "extraction.json", {"topic": "done"})
    recorder.mark_recording_completed(done_dir.name)
    pending_dir = output_root / "pending"
    pending_dir.mkdir()
    write_json(pending_dir / "transcript.json", {"segments": []})
    jobs = [_FakeJob("a.mp4", done_dir), _FakeJob("b.mp4", pending_dir)]
    run_path = recorder.finalize(jobs, status="interrupted", steps_completed=5, steps_total=8)
    data = json.loads(run_path.read_text(encoding="utf-8"))
    assert data["aggregates"]["recordings_completed"] == 1
