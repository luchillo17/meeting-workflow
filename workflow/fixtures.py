"""Fixture adapters for slice-1 Workflow Runs (no GPU/ffmpeg/Ollama)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from workflow.summary import render_summary
from workflow.utils import parse_meeting_date, write_json


@dataclass
class SimpleTranscript:
    segments: list[dict]
    text: str


class FixtureTranscriber:
    def transcribe(self, recording: Path, output_dir: Path) -> SimpleTranscript:
        result = SimpleTranscript(
            segments=[{"start": 0.0, "end": 5.0, "text": "Hola, revisemos el tablero."}],
            text="Hola, revisemos el tablero.",
        )
        write_json(output_dir / "transcript.json", {"segments": result.segments})
        (output_dir / "transcript.txt").write_text(result.text, encoding="utf-8")
        return result


class FixtureFrameExtractor:
    def extract(self, recording: Path, transcript: SimpleTranscript, output_dir: Path) -> list[Path]:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / "frame_0001.jpg"
        frame_path.write_bytes(b"fixture-frame")
        write_json(
            output_dir / "frames.json",
            [{"timestamp": 1.0, "path": str(frame_path), "trigger": "fixture"}],
        )
        return [frame_path]


class FixtureVisionAnalyzer:
    def analyze(self, frame_paths: list[Path], output_dir: Path) -> list[dict]:
        visual = [
            {
                "timestamp": "00:00:01",
                "type": "whiteboard",
                "description": "Diagrama de arquitectura con tres bloques.",
            }
        ]
        write_json(output_dir / "visual_content.json", visual)
        return visual


class FixtureExtractor:
    def build_extraction(
        self,
        transcript: SimpleTranscript,
        visual_content: list[dict],
        recording_name: str,
        output_dir: Path,
    ) -> dict:
        extraction = {
            "meeting_date": parse_meeting_date(recording_name),
            "topic": "Fixture meeting",
            "key_decisions": ["Aprobar el enfoque híbrido transcript + visual."],
            "action_items": [
                {"owner": "Carlos", "task": "Pilotar el workflow", "deadline": "2026-07-10"}
            ],
            "blockers_risks": [],
            "status_updates": ["Workflow runner scaffold complete."],
            "technical_details": ["WorkflowRunner seam with injectable adapters."],
            "open_questions": [],
            "next_steps": ["Integrar transcripción real."],
            "visual_content": visual_content,
        }
        write_json(output_dir / "extraction.json", extraction)
        (output_dir / "summary.md").write_text(render_summary(extraction), encoding="utf-8")
        return extraction


def fixture_adapters() -> tuple[FixtureTranscriber, FixtureFrameExtractor, FixtureVisionAnalyzer, FixtureExtractor]:
    return FixtureTranscriber(), FixtureFrameExtractor(), FixtureVisionAnalyzer(), FixtureExtractor()
