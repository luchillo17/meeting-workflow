"""Batch run statistics — performance and quality records (ADR 0010)."""

from __future__ import annotations

import json
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from workflow.extraction_eval import (
    PILOT_MEETING_CHECKS,
    evaluate_output_dir,
    load_chapter_count,
    load_transcript_chars,
    load_transcript_text,
    load_visual_frame_count,
)
from workflow.extraction_grounding import compute_grounding_ratio
from workflow.publish import filter_publishable_visual, load_visual_content
from workflow.settings import Settings
from workflow.utils import utc_now_iso, write_json

RunStatus = Literal["completed", "interrupted", "failed"]


def new_run_id() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def config_fingerprint(config: dict, *, settings: Settings) -> dict[str, Any]:
    whisper = config.get("whisper", {})
    diarization = config.get("diarization", {})
    extraction = config.get("extraction", {})
    batch = config.get("batch", {})
    return {
        "whisper_model": whisper.get("model"),
        "diarization_enabled": bool(diarization.get("enabled")),
        "diarization_model": diarization.get("model"),
        "ollama_text_model": settings.ollama_text_model,
        "ollama_vision_model": settings.ollama_vision_model,
        "extraction_mode": extraction.get("mode"),
        "deterministic_grounding": extraction.get("deterministic_grounding"),
        "pipeline_workers": batch.get("pipeline_workers"),
        "frame_workers": batch.get("frame_workers"),
        "max_frames": config.get("frames", {}).get("max_frames"),
    }


def audio_duration_seconds(output_dir: Path) -> float | None:
    transcript_path = output_dir / "transcript.json"
    if transcript_path.is_file():
        try:
            data = json.loads(transcript_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
        segments = data.get("segments", []) if isinstance(data, dict) else data
        if isinstance(segments, list) and segments:
            ends = [
                float(segment.get("end", 0.0)) for segment in segments if isinstance(segment, dict)
            ]
            if ends:
                return max(ends)
    audio_path = output_dir / "audio.wav"
    if audio_path.is_file():
        try:
            with wave.open(str(audio_path), "rb") as handle:
                return handle.getnframes() / float(handle.getframerate())
        except (OSError, wave.Error):
            return None
    return None


def distinct_speaker_count(output_dir: Path) -> int | None:
    diar_path = output_dir / "diarization.json"
    if not diar_path.is_file():
        return None
    try:
        data = json.loads(diar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    turns = data.get("turns") if isinstance(data, dict) else None
    if not isinstance(turns, list):
        return None
    speakers = {
        str(turn.get("speaker", ""))
        for turn in turns
        if isinstance(turn, dict) and turn.get("speaker")
    }
    return len(speakers)


def transcript_segment_count(output_dir: Path) -> int:
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.is_file():
        return 0
    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    segments = data.get("segments", []) if isinstance(data, dict) else data
    return len(segments) if isinstance(segments, list) else 0


def frame_count(output_dir: Path) -> int:
    frames_path = output_dir / "frames.json"
    if not frames_path.is_file():
        return 0
    try:
        data = json.loads(frames_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(data) if isinstance(data, list) else 0


def infer_extraction_mode(output_dir: Path, config: dict) -> str:
    extraction_cfg = config.get("extraction", {})
    mode = str(extraction_cfg.get("mode", "auto"))
    if mode != "auto":
        return mode
    trigger = int(extraction_cfg.get("chapter_trigger_chars", 12_000))
    if load_chapter_count(output_dir) > 0 or load_transcript_chars(output_dir) > trigger:
        return "map_reduce"
    return "single"


def pilot_eval_for_output(output_dir: Path) -> dict[str, Any] | None:
    for spec in PILOT_MEETING_CHECKS:
        prefix = str(spec["slug_prefix"])
        if output_dir.name.startswith(prefix):
            failures = evaluate_output_dir(output_dir, checks=spec)
            return {
                "slug_prefix": prefix,
                "passed": not failures,
                "failures": failures,
            }
    return None


def collect_recording_stats(
    output_dir: Path,
    *,
    source_filename: str,
    stage_seconds: dict[str, float],
    config: dict,
) -> dict[str, Any]:
    extraction_path = output_dir / "extraction.json"
    extraction: dict[str, Any] = {}
    if extraction_path.is_file():
        try:
            loaded = json.loads(extraction_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                extraction = loaded
        except json.JSONDecodeError:
            extraction = {}

    visual_raw = load_visual_content(output_dir)
    visual_publishable = filter_publishable_visual(visual_raw)
    transcript_text = load_transcript_text(output_dir)
    visual_text = " ".join(
        str(entry.get("description", "")) for entry in visual_raw if isinstance(entry, dict)
    )
    grounding_ratio: float | None = None
    grounding_items: str | None = None
    if extraction and transcript_text.strip():
        ratio, grounded, total = compute_grounding_ratio(
            extraction,
            transcript_text=transcript_text,
            visual_text=visual_text,
            min_overlap_ratio=float(
                config.get("extraction", {}).get("grounding_min_overlap", 0.34)
            ),
        )
        if total:
            grounding_ratio = round(ratio, 4)
            grounding_items = f"{grounded}/{total}"

    pilot = pilot_eval_for_output(output_dir)
    duration = audio_duration_seconds(output_dir)

    return {
        "source_filename": source_filename,
        "output_slug": output_dir.name,
        "audio_duration_seconds": round(duration, 2) if duration is not None else None,
        "stage_seconds": {key: round(value, 3) for key, value in stage_seconds.items()},
        "counts": {
            "transcript_segments": transcript_segment_count(output_dir),
            "diarization_speakers": distinct_speaker_count(output_dir),
            "frames_kept": frame_count(output_dir),
            "vision_frames": load_visual_frame_count(output_dir),
            "vision_frames_publishable": len(visual_publishable),
            "chapters": load_chapter_count(output_dir),
            "key_decisions": len(extraction.get("key_decisions") or []),
            "action_items": len(extraction.get("action_items") or []),
            "blockers_risks": len(extraction.get("blockers_risks") or []),
        },
        "quality": {
            "extraction_mode": infer_extraction_mode(output_dir, config),
            "grounding_ratio": grounding_ratio,
            "grounding_items": grounding_items,
            "pilot_eval": pilot,
            "diarization_present": (output_dir / "diarization.json").is_file(),
            "extraction_present": extraction_path.is_file(),
        },
    }


@dataclass
class BatchRunRecorder:
    """Collect per-stage timings and write a run record when the batch ends."""

    output_root: Path
    settings: Settings
    recording_paths: list[Path]
    force: bool = False
    extract_only: bool = False
    vision_only: bool = False
    run_id: str = field(default_factory=new_run_id)
    started_at: str = field(default_factory=utc_now_iso)
    _stage_seconds: dict[str, dict[str, float]] = field(default_factory=dict)
    _completed_slugs: set[str] = field(default_factory=set)
    _started_monotonic: float = field(default_factory=time.monotonic)

    def start_stage_timer(self, output_slug: str, stage: str) -> _StageTimer:
        return _StageTimer(self, output_slug, stage)

    def mark_recording_completed(self, output_slug: str) -> None:
        self._completed_slugs.add(output_slug)

    def add_stage_seconds(self, output_slug: str, stage: str, seconds: float) -> None:
        per_recording = self._stage_seconds.setdefault(output_slug, {})
        per_recording[stage] = per_recording.get(stage, 0.0) + max(0.0, seconds)

    def finalize(
        self,
        jobs: list[Any],
        *,
        status: RunStatus,
        steps_completed: int | None = None,
        steps_total: int | None = None,
    ) -> Path:
        finished_at = utc_now_iso()
        wall_seconds = round(time.monotonic() - self._started_monotonic, 3)
        recordings: list[dict[str, Any]] = []
        for job in jobs:
            stage_seconds = self._stage_seconds.get(job.output_dir.name, {})
            recordings.append(
                collect_recording_stats(
                    job.output_dir,
                    source_filename=job.recording.name,
                    stage_seconds=stage_seconds,
                    config=self.settings.config,
                )
            )

        record = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "status": status,
            "input": {
                "recording_count": len(self.recording_paths),
                "processed_count": len(jobs),
                "recordings": [path.name for path in self.recording_paths],
                "force": self.force,
                "extract_only": self.extract_only,
                "vision_only": self.vision_only,
            },
            "config": config_fingerprint(self.settings.config, settings=self.settings),
            "aggregates": {
                "wall_seconds": wall_seconds,
                "steps_completed": steps_completed,
                "steps_total": steps_total,
                "recordings_completed": len(self._completed_slugs),
            },
            "recordings": recordings,
        }
        return persist_run_record(self.output_root, record)


@dataclass
class _StageTimer:
    recorder: BatchRunRecorder
    output_slug: str
    stage: str
    _started: float = field(default_factory=time.monotonic)

    def stop(self) -> None:
        elapsed = time.monotonic() - self._started
        self.recorder.add_stage_seconds(self.output_slug, self.stage, elapsed)


def persist_run_record(output_root: Path, record: dict[str, Any]) -> Path:
    runs_dir = output_root / "batch-runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{record['run_id']}.json"
    write_json(run_path, record)
    jsonl_path = output_root / "batch-runs.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return run_path


def load_run_history(output_root: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    jsonl_path = output_root / "batch-runs.jsonl"
    if not jsonl_path.is_file():
        return []
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def median_stage_seconds_per_audio_minute(
    output_root: Path,
    *,
    stage: str,
    limit: int = 20,
) -> float | None:
    """Rolling median of stage wall seconds per audio minute (for ETA hints)."""
    ratios: list[float] = []
    for record in reversed(load_run_history(output_root, limit=limit)):
        if record.get("status") != "completed":
            continue
        for row in record.get("recordings") or []:
            if not isinstance(row, dict):
                continue
            duration = row.get("audio_duration_seconds")
            stage_seconds = (row.get("stage_seconds") or {}).get(stage)
            if not duration or not stage_seconds or duration <= 0:
                continue
            ratios.append(float(stage_seconds) / (float(duration) / 60.0))
    if not ratios:
        return None
    ratios.sort()
    mid = len(ratios) // 2
    if len(ratios) % 2:
        return ratios[mid]
    return (ratios[mid - 1] + ratios[mid]) / 2.0
