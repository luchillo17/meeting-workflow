"""Orchestrates a Workflow Run on one Meeting Recording."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from workflow.ports import Extractor, FrameExtractor, Transcriber, VisionAnalyzer
from workflow.settings import Settings
from workflow.utils import slugify


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
        recording = Path(recording_path)
        if not recording.is_file():
            raise FileNotFoundError(f"Meeting Recording not found: {recording}")

        output_dir = self.settings.output_dir / slugify(recording.name)
        extraction_file = output_dir / "extraction.json"

        if extraction_file.exists() and not force:
            self.console.print(
                f"[yellow]Skipping[/yellow] - Extraction already exists at {extraction_file}"
            )
            return output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        self.console.print(f"[bold]Workflow Run[/bold] -> {recording.name}")

        self.console.print("  [cyan]Transcript[/cyan]")
        transcript = self.transcriber.transcribe(recording, output_dir)

        self.console.print("  [cyan]Visual Capture[/cyan]")
        frame_paths = self.frame_extractor.extract(recording, transcript, output_dir)
        visual_content = self.vision.analyze(frame_paths, output_dir)

        self.console.print("  [cyan]Structured Extraction[/cyan]")
        self.extractor.build_extraction(transcript, visual_content, recording.name, output_dir)

        self.console.print(f"[green]Done[/green] -> {output_dir}")
        return output_dir
