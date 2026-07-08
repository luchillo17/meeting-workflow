"""Orchestrates a Workflow Run on one Meeting Recording."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from workflow.batch_progress import STAGE_NAMES, BatchProgress, BatchSettings
from workflow.frames import frames_bundle_valid, load_frame_paths, load_transcript_for_frames
from workflow.ports import Extractor, FrameExtractor, Transcriber, TranscriptResult, VisionAnalyzer
from workflow.settings import Settings
from workflow.shutdown import BatchInterrupted, get_shutdown_coordinator
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
        self._batch = BatchSettings.from_config(settings.config)

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
        workers_note = ""
        if not extract_only and not vision_only and self._batch.pipeline_workers > 1:
            workers_note = (
                f" · pipeline={self._batch.pipeline_workers}"
                f" · frame_workers={self._batch.frame_workers}"
            )
        self.console.print(f"[bold]{label}[/bold] ({count} recording(s)){workers_note}")

        stage_count = self._stage_count(extract_only=extract_only, vision_only=vision_only)
        shutdown = get_shutdown_coordinator()
        shutdown.acquire_batch_lock(self.settings.output_dir)
        self._register_shutdown_cleanup()

        try:
            with BatchProgress(self.console, count, stage_count=stage_count) as progress:
                if vision_only:
                    for job in jobs:
                        shutdown.check_interrupted()
                        job.transcript = load_transcript_for_frames(job.output_dir)
                        job.frame_paths = load_frame_paths(job.output_dir)
                        if not job.frame_paths:
                            raise FileNotFoundError(
                                f"No frames for vision-only rerun: {job.output_dir} "
                                "(run full process or frames first)"
                            )
                    self._run_vision_stage(jobs, force=True, progress=progress, stage=1)
                    self._run_extraction_stage(jobs, progress=progress, stage=2)
                elif extract_only:
                    self._run_extraction_stage(jobs, progress=progress, stage=1)
                elif self._batch.pipeline_workers > 1:
                    self._run_pipelined(jobs, force=force, progress=progress)
                else:
                    self._run_staged(jobs, force=force, progress=progress)
        except BatchInterrupted:
            self.console.print("[yellow]Batch interrupted — GPU models unloaded[/yellow]")
            raise
        finally:
            shutdown.release_batch_lock()

        for job in jobs:
            self.console.print(f"[green]Done[/green] -> {job.output_dir}")
        return [job.output_dir for job in jobs]

    @staticmethod
    def _stage_count(*, extract_only: bool, vision_only: bool) -> int:
        if extract_only:
            return 1
        if vision_only:
            return 2
        return len(STAGE_NAMES)

    def _register_shutdown_cleanup(self) -> None:
        shutdown = get_shutdown_coordinator()
        for adapter in (self.transcriber, self.vision, self.extractor):
            shutdown.register_cleanup(lambda adapter=adapter: self._unload_adapter(adapter))

    def _run_staged(self, jobs: list[_BatchJob], *, force: bool, progress: BatchProgress) -> None:
        self._run_transcript_stage(jobs, force=force, progress=progress)
        self._run_frames_stage(jobs, force=force, progress=progress)
        self._run_vision_stage(jobs, force=force, progress=progress)
        self._run_extraction_stage(jobs, progress=progress)

    def _run_pipelined(
        self, jobs: list[_BatchJob], *, force: bool, progress: BatchProgress
    ) -> None:
        progress.begin_pipeline()
        gpu_lock = threading.Lock()
        shutdown = get_shutdown_coordinator()
        executor: ThreadPoolExecutor | None = None

        def _work(job: _BatchJob, index: int) -> None:
            shutdown.check_interrupted()
            self._process_job_pipelined(
                job, force=force, gpu_lock=gpu_lock, progress=progress, index=index
            )

        try:
            executor = ThreadPoolExecutor(max_workers=self._batch.pipeline_workers)
            futures = [
                executor.submit(_work, job, index) for index, job in enumerate(jobs, start=1)
            ]
            for future in as_completed(futures):
                shutdown.check_interrupted()
                future.result()
        except BatchInterrupted:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            if executor is not None:
                executor.shutdown(wait=True)

    def _process_job_pipelined(
        self,
        job: _BatchJob,
        *,
        force: bool,
        gpu_lock: threading.Lock,
        progress: BatchProgress,
        index: int,
    ) -> None:
        name = job.recording.name
        progress.start_step(1, "Transcript", index, name)
        with gpu_lock:
            if self._transcribe_job(job, force=force):
                self._unload_adapter(self.transcriber)
        progress.complete_step(1, "Transcript", index, name)

        progress.start_step(2, "Frames", index, name)
        self._frames_job(job, force=force)
        progress.complete_step(2, "Frames", index, name)

        progress.start_step(3, "Vision", index, name)
        with gpu_lock:
            if self._vision_job(job, force=force):
                self._unload_adapter(self.vision)
        progress.complete_step(3, "Vision", index, name)

        progress.start_step(4, "Extraction", index, name)
        with gpu_lock:
            self._extract_job(job)
            self._unload_adapter(self.extractor)
        progress.complete_step(4, "Extraction", index, name)

    def _run_transcript_stage(
        self, jobs: list[_BatchJob], *, force: bool, progress: BatchProgress
    ) -> None:
        progress.begin_stage(1, "Transcript")
        transcribed_any = False
        shutdown = get_shutdown_coordinator()
        for index, job in enumerate(jobs, start=1):
            shutdown.check_interrupted()
            progress.update(1, "Transcript", index, job.recording.name)
            if self._transcribe_job(job, force=force):
                transcribed_any = True
            progress.advance_stage_item(1, "Transcript", index, job.recording.name)
        if transcribed_any:
            self._unload_adapter(self.transcriber)

    def _run_frames_stage(
        self, jobs: list[_BatchJob], *, force: bool, progress: BatchProgress
    ) -> None:
        progress.begin_stage(2, "Frames")
        shutdown = get_shutdown_coordinator()
        pending = [
            (index, job)
            for index, job in enumerate(jobs, start=1)
            if force or not frames_bundle_valid(job.output_dir)
        ]
        for index, job in enumerate(jobs, start=1):
            shutdown.check_interrupted()
            if not force and frames_bundle_valid(job.output_dir):
                progress.update(2, "Frames", index, job.recording.name)
                job.frame_paths = load_frame_paths(job.output_dir)
                progress.advance_stage_item(2, "Frames", index, job.recording.name)

        if not pending:
            return

        if self._batch.frame_workers <= 1 or len(pending) == 1:
            for index, job in pending:
                shutdown.check_interrupted()
                progress.update(2, "Frames", index, job.recording.name)
                self._frames_job(job, force=force)
                progress.advance_stage_item(2, "Frames", index, job.recording.name)
            return

        lock = threading.Lock()
        executor: ThreadPoolExecutor | None = None

        def _extract_frames(item: tuple[int, _BatchJob]) -> None:
            shutdown.check_interrupted()
            index, job = item
            progress.update(2, "Frames", index, job.recording.name)
            self._frames_job(job, force=force)
            with lock:
                progress.advance_stage_item(2, "Frames", index, job.recording.name)

        try:
            executor = ThreadPoolExecutor(max_workers=self._batch.frame_workers)
            futures = [executor.submit(_extract_frames, item) for item in pending]
            for future in as_completed(futures):
                shutdown.check_interrupted()
                future.result()
        except BatchInterrupted:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            if executor is not None:
                executor.shutdown(wait=True)

    def _run_vision_stage(
        self,
        jobs: list[_BatchJob],
        *,
        force: bool,
        progress: BatchProgress,
        stage: int = 3,
    ) -> None:
        progress.begin_stage(stage, "Vision")
        vision_ran = False
        shutdown = get_shutdown_coordinator()
        for index, job in enumerate(jobs, start=1):
            shutdown.check_interrupted()
            progress.update(stage, "Vision", index, job.recording.name)
            if self._vision_job(job, force=force):
                vision_ran = True
            progress.advance_stage_item(stage, "Vision", index, job.recording.name)
        if vision_ran:
            self._unload_adapter(self.vision)

    def _run_extraction_stage(
        self,
        jobs: list[_BatchJob],
        *,
        progress: BatchProgress,
        stage: int = 4,
    ) -> None:
        progress.begin_stage(stage, "Extraction")
        shutdown = get_shutdown_coordinator()
        for index, job in enumerate(jobs, start=1):
            shutdown.check_interrupted()
            progress.update(stage, "Extraction", index, job.recording.name)
            self._extract_job(job)
            progress.advance_stage_item(stage, "Extraction", index, job.recording.name)
        self._unload_adapter(self.extractor)

    def _transcribe_job(self, job: _BatchJob, *, force: bool) -> bool:
        transcript_path = job.output_dir / "transcript.json"
        if not force and transcript_path.is_file():
            self.console.print(f"  [dim]Reusing transcript[/dim] -> {job.recording.name}")
            job.transcript = load_transcript_for_frames(job.output_dir)
            return False
        self.console.print(f"  [cyan]Transcript[/cyan] -> {job.recording.name}")
        job.transcript = self.transcriber.transcribe(
            job.recording, job.output_dir, unload_after=False
        )
        return True

    def _frames_job(self, job: _BatchJob, *, force: bool) -> None:
        if not force and frames_bundle_valid(job.output_dir):
            self.console.print(f"  [dim]Reusing frames[/dim] -> {job.recording.name}")
            job.frame_paths = load_frame_paths(job.output_dir)
            return
        self.console.print(f"  [cyan]Frames[/cyan] -> {job.recording.name}")
        transcript = job.transcript or load_transcript_for_frames(job.output_dir)
        job.frame_paths = self.frame_extractor.extract(job.recording, transcript, job.output_dir)

    def _vision_job(self, job: _BatchJob, *, force: bool) -> bool:
        visual_path = job.output_dir / "visual_content.json"
        if not force and visual_path.is_file():
            self.console.print(f"  [dim]Reusing vision[/dim] -> {job.recording.name}")
            job.visual_content = self._load_visual_content(job.output_dir)
            return False
        self.console.print(f"  [cyan]Vision[/cyan] -> {job.recording.name}")
        job.visual_content = self.vision.analyze(
            job.frame_paths, job.output_dir, unload_after=False
        )
        return True

    def _extract_job(self, job: _BatchJob) -> None:
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
