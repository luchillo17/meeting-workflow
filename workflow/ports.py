"""Ports for injectable workflow adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TranscriptResult(Protocol):
    segments: list[dict]
    text: str


class Transcriber(Protocol):
    def transcribe(
        self, recording: Path, output_dir: Path, *, unload_after: bool = True
    ) -> TranscriptResult: ...


class FrameExtractor(Protocol):
    def extract(
        self, recording: Path, transcript: TranscriptResult, output_dir: Path
    ) -> list[Path]: ...


class VisionAnalyzer(Protocol):
    def analyze(
        self, frame_paths: list[Path], output_dir: Path, *, unload_after: bool = True
    ) -> list[dict]: ...


class Extractor(Protocol):
    def build_extraction(
        self,
        transcript: TranscriptResult,
        visual_content: list[dict],
        recording_name: str,
        output_dir: Path,
        *,
        unload_after: bool = True,
    ) -> dict: ...
