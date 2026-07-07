"""Orchestrates a Workflow Run on one Meeting Recording."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from workflow.frames import frames_bundle_valid, load_frame_paths, load_transcript_for_frames
from workflow.ports import Extractor, FrameExtractor, Transcriber, TranscriptResult, VisionAnalyzer
from workflow.settings import Settings
from workflow.utils import invalidate_downstream_artifacts, slugify


@dataclass
class _BatchJob:
    recording: Path
    output_dir: Path
    transcript: TranscriptResult | None = None
    frame_paths: list[Path] = field(default_factory=list)
    visual_content: list[dict[str, Any]] = field(default_factory=list)


class WorkflowRunner:
    def __init__(
        self,
        settings: Settings,
        transcriber: Transcriber,
        frame_extractor: FrameExtractor,
        vision: VisionAnalyzer,
        extractor: Extractor,
        console: Console | None = None,
    ) -> None:
        self.settings = settings
        self.transcriber = transcriber
        self.frame_extractor = frame_extractor
        self.vision = vision
        self.extractor = extractor
        self.console = console or Console()

    def run(self, recording_path: Path, *, force: bool = False) -> Path:
        results = self.run_batch([recording_path], force=force)
        if not results:
            return self.settings.output_dir / slugify(Path(recording_path).name)
        return results[0]

    def run_batch(
        self,
        recording_paths: list[Path],
        *,
        force: bool = False,
        extract_only: bool = False,
        vision_only: bool = False,
    ) -> list[Path]:
        if extract_only and vision_only:
            raise ValueError("Use only one of extract_only or vision_only")
        jobs = self._prepare_jobs(
            recording_paths,
            force=force,
            extract_only=extract_only,
            vision_only=vision_only,
        )
        if not jobs:
            return []

        count = len(jobs)
        if vision_only:
            label = "Vision-only"
        elif extract_only:
            label = "Extraction-only"
        else:
            label = "Batch Workflow Run"
        self.console.print(f"[bold]{label}[/bold] ({count} recording(s))")

        if vision_only:
            for job in jobs:
                job.transcript = load_transcript_for_frames(job.output_dir)
                job.frame_paths = load_frame_paths(job.output_dir)
                if not job.frame_paths:
                    raise FileNotFoundError(
                        f"No frames for vision-only rerun: {job.output_dir} "
                        "(run full process or frames first)"
                    )
            self._run_vision_stage(jobs, force=True)
        elif not extract_only:
            self._run_transcript_stage(jobs, force=force)
            self._run_frames_stage(jobs, force=force)
            self._run_vision_stage(jobs, force=force)

        self.console.print("[bold cyan]Stage 4/4[/bold cyan] Structured Extraction")
        for job in jobs:
            self.console.print(f"  [cyan]Extraction[/cyan] -> {job.recording.name}")
            transcript = job.transcript or load_transcript_for_frames(job.output_dir)
            visual = job.visual_content or self._load_visual_content(job.output_dir)
            self.extractor.build_extraction(
                transcript,
                visual,
                job.recording.name,
                job.output_dir,
                unload_after=False,
            )
        self._unload_adapter(self.extractor)

        for job in jobs:
            self.console.print(f"[green]Done[/green] -> {job.output_dir}")
        return [job.output_dir for job in jobs]

    def _run_transcript_stage(self, jobs: list[_BatchJob], *, force: bool) -> None:
        self.console.print("[bold cyan]Stage 1/4[/bold cyan] Transcript")
        transcribed_any = False
        for job in jobs:
            transcript_path = job.output_dir / "transcript.json"
            if not force and transcript_path.is_file():
                self.console.print(f"  [dim]Reusing transcript[/dim] -> {job.recording.name}")
                job.transcript = load_transcript_for_frames(job.output_dir)
                continue
            self.console.print(f"  [cyan]Transcript[/cyan] -> {job.recording.name}")
            job.transcript = self.transcriber.transcribe(
                job.recording, job.output_dir, unload_after=False
            )
            transcribed_any = True
        if transcribed_any:
            self._unload_adapter(self.transcriber)

    def _run_frames_stage(self, jobs: list[_BatchJob], *, force: bool) -> None:
        self.console.print("[bold cyan]Stage 2/4[/bold cyan] Visual Capture (frames)")
        for job in jobs:
            if not force and frames_bundle_valid(job.output_dir):
                self.console.print(f"  [dim]Reusing frames[/dim] -> {job.recording.name}")
                job.frame_paths = load_frame_paths(job.output_dir)
                continue
            self.console.print(f"  [cyan]Frames[/cyan] -> {job.recording.name}")
            transcript = job.transcript or load_transcript_for_frames(job.output_dir)
            job.frame_paths = self.frame_extractor.extract(
                job.recording, transcript, job.output_dir
            )

    def _run_vision_stage(self, jobs: list[_BatchJob], *, force: bool) -> None:
        self.console.print("[bold cyan]Stage 3/4[/bold cyan] Visual Capture (vision)")
        vision_ran = False
        for job in jobs:
            visual_path = job.output_dir / "visual_content.json"
            if not force and visual_path.is_file():
                self.console.print(f"  [dim]Reusing vision[/dim] -> {job.recording.name}")
                job.visual_content = self._load_visual_content(job.output_dir)
                continue
            self.console.print(f"  [cyan]Vision[/cyan] -> {job.recording.name}")
            job.visual_content = self.vision.analyze(
                job.frame_paths, job.output_dir, unload_after=False
            )
            vision_ran = True
        if vision_ran:
            self._unload_adapter(self.vision)

    def _prepare_jobs(
        self,
        recording_paths: list[Path],
        *,
        force: bool,
        extract_only: bool = False,
        vision_only: bool = False,
    ) -> list[_BatchJob]:
        jobs: list[_BatchJob] = []
        for recording_path in recording_paths:
            recording = Path(recording_path)
            if not recording.is_file():
                raise FileNotFoundError(f"Meeting Recording not found: {recording}")

            output_dir = self.settings.output_dir / slugify(recording.name)
            extraction_file = output_dir / "extraction.json"

            if extraction_file.exists() and not force and not extract_only and not vision_only:
                self.console.print(
                    f"[yellow]Skipping[/yellow] - Extraction already exists at {extraction_file}"
                )
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            invalidate_downstream_artifacts(
                output_dir,
                include_visual=vision_only or (force and not extract_only),
            )
            jobs.append(_BatchJob(recording=recording, output_dir=output_dir))
        return jobs

    @staticmethod
    def _load_visual_content(output_dir: Path) -> list[dict[str, Any]]:
        visual_path = output_dir / "visual_content.json"
        if not visual_path.is_file():
            return []
        try:
            data = json.loads(visual_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    @staticmethod
    def _unload_adapter(adapter: object) -> None:
        unload = getattr(adapter, "unload", None)
        if callable(unload):
            unload()
