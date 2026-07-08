"""Rich progress reporting for batch workflow runs."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

STAGE_NAMES = ("Transcript", "Frames", "Vision", "Extraction")


@dataclass(frozen=True)
class BatchSettings:
    pipeline_workers: int = 1

    frame_workers: int = 4

    @classmethod
    def from_config(cls, config: dict) -> BatchSettings:
        batch = config.get("batch", {})

        import os

        cpu_default = max(1, min(4, (os.cpu_count() or 4) // 2))

        pipeline_workers = int(batch.get("pipeline_workers", 1))

        frame_workers = int(batch.get("frame_workers", cpu_default))

        return cls(
            pipeline_workers=max(1, pipeline_workers),
            frame_workers=max(1, frame_workers),
        )


def _short_name(name: str, limit: int = 48) -> str:
    return name if len(name) <= limit else f"{name[: limit - 1]}…"


class BatchProgress:
    """Overall step progress across all recordings and stages."""

    def __init__(
        self,
        console: Console,
        total_recordings: int,
        *,
        stage_count: int = 4,
    ) -> None:
        self.console = console

        self.total_recordings = total_recordings

        self.stage_count = stage_count

        self.total_steps = max(1, total_recordings * stage_count)

        self.completed_steps = 0

        self._use_live_bar = console.is_terminal

        self._progress: Progress | None = None

        self._task_id: int | None = None

    def __enter__(self) -> BatchProgress:
        if self._use_live_bar:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
            )

            self._progress.start()

            self._task_id = self._progress.add_task("Starting…", total=self.total_steps)

        self._emit(0, "Starting", 0, "")

        return self

    def __exit__(self, *args: object) -> None:
        if self._progress is not None:
            self._progress.stop()

    def _emit(
        self,
        stage: int,
        stage_name: str,
        recording_index: int,
        recording_name: str,
        *,
        working: bool = False,
    ) -> None:
        current_step = self.completed_steps + (1 if working else 0)

        current_step = min(current_step, self.total_steps)

        pct = int((current_step / self.total_steps) * 100)

        rec_part = (
            f"recording [{recording_index}/{self.total_recordings}] {_short_name(recording_name)}"
            if recording_index
            else "initializing"
        )

        stage_part = f"stage {stage}/{self.stage_count} {stage_name}" if stage else "starting"

        status = "working" if working else "done"

        line = (
            f"Progress {pct:>3}% | step {current_step}/{self.total_steps} | "
            f"{stage_part} | {rec_part} | {status}"
        )

        self.console.print(f"[bold cyan]{line}[/bold cyan]")

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=current_step,
                total=self.total_steps,
                description=line[:100],
            )

    def begin_stage(self, stage: int, stage_name: str) -> None:
        return

    def start_step(
        self,
        stage: int,
        stage_name: str,
        recording_index: int,
        recording_name: str,
    ) -> None:
        self._emit(stage, stage_name, recording_index, recording_name, working=True)

    def complete_step(
        self,
        stage: int,
        stage_name: str,
        recording_index: int,
        recording_name: str,
    ) -> None:
        self.completed_steps = min(self.completed_steps + 1, self.total_steps)

        self._emit(stage, stage_name, recording_index, recording_name, working=False)

    def update(
        self,
        stage: int,
        stage_name: str,
        index: int,
        recording_name: str,
    ) -> None:
        self.start_step(stage, stage_name, index, recording_name)

    def advance_stage_item(
        self,
        stage: int,
        stage_name: str,
        index: int,
        recording_name: str,
    ) -> None:
        self.complete_step(stage, stage_name, index, recording_name)

    def begin_pipeline(self) -> None:
        self._emit(0, "Pipeline", 0, "")

    def update_pipeline(self, index: int, step: str, recording_name: str) -> None:
        stage_map = {
            "transcript": 1,
            "frames": 2,
            "vision": 3,
            "extraction": 4,
        }

        stage = stage_map.get(step.lower(), 0)

        stage_name = step.capitalize()

        self.start_step(stage, stage_name, index, recording_name)

    def advance_pipeline(self, index: int, recording_name: str) -> None:
        self.completed_steps = min(self.completed_steps + 1, self.total_steps)

        self._emit(4, "Done", index, recording_name, working=False)
