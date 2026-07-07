"""Split Whisper segments into on-disk chapters for map-reduce extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from workflow.utils import write_json

_TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")


@dataclass(frozen=True)
class TranscriptChapter:
    index: int
    start_seconds: float
    end_seconds: float
    text: str
    path: str = ""

    @property
    def time_range_label(self) -> str:
        return f"{format_timestamp(self.start_seconds)}–{format_timestamp(self.end_seconds)}"


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_timestamp(timestamp: str) -> float:
    match = _TIMESTAMP_RE.match(timestamp.strip())
    if not match:
        return 0.0
    hours, minutes, seconds = (int(match.group(i)) for i in range(1, 4))
    return float(hours * 3600 + minutes * 60 + seconds)


def split_segments_into_chapters(
    segments: list[dict],
    *,
    target_chars: int = 8_000,
    min_chars: int = 1_500,
) -> list[TranscriptChapter]:
    """Group adjacent Whisper segments into chapters under target_chars."""
    if target_chars <= 0:
        target_chars = 8_000

    chapters: list[TranscriptChapter] = []
    bucket_segments: list[dict] = []
    bucket_chars = 0

    def flush_bucket() -> None:
        nonlocal bucket_segments, bucket_chars
        if not bucket_segments:
            return
        text = " ".join(str(seg.get("text", "")).strip() for seg in bucket_segments).strip()
        if not text:
            bucket_segments = []
            bucket_chars = 0
            return
        chapters.append(
            TranscriptChapter(
                index=len(chapters) + 1,
                start_seconds=float(bucket_segments[0].get("start", 0.0)),
                end_seconds=float(bucket_segments[-1].get("end", 0.0)),
                text=text,
            )
        )
        bucket_segments = []
        bucket_chars = 0

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        piece = str(segment.get("text", "")).strip()
        if not piece:
            continue
        extra = len(piece) + (1 if bucket_segments else 0)
        if bucket_segments and bucket_chars + extra > target_chars:
            flush_bucket()
        bucket_segments.append(segment)
        bucket_chars += extra

    flush_bucket()

    if len(chapters) >= 2 and len(chapters[-1].text) < min_chars:
        tail = chapters.pop()
        merged_text = f"{chapters[-1].text} {tail.text}".strip()
        chapters[-1] = TranscriptChapter(
            index=chapters[-1].index,
            start_seconds=chapters[-1].start_seconds,
            end_seconds=tail.end_seconds,
            text=merged_text,
        )

    return [
        TranscriptChapter(
            index=index,
            start_seconds=chapter.start_seconds,
            end_seconds=chapter.end_seconds,
            text=chapter.text,
        )
        for index, chapter in enumerate(chapters, start=1)
    ]


def write_chapter_bundle(
    output_dir: Path, chapters: list[TranscriptChapter]
) -> list[TranscriptChapter]:
    """Persist chapter text files and chapters.json metadata."""
    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    written: list[TranscriptChapter] = []
    meta: list[dict] = []

    for chapter in chapters:
        filename = f"chapter_{chapter.index:03d}.txt"
        rel_path = f"chapters/{filename}"
        path = output_dir / rel_path
        path.write_text(chapter.text, encoding="utf-8")
        updated = TranscriptChapter(
            index=chapter.index,
            start_seconds=chapter.start_seconds,
            end_seconds=chapter.end_seconds,
            text=chapter.text,
            path=rel_path,
        )
        written.append(updated)
        meta.append(
            {
                "index": updated.index,
                "start_seconds": updated.start_seconds,
                "end_seconds": updated.end_seconds,
                "path": rel_path,
                "char_count": len(updated.text),
                "time_range": updated.time_range_label,
            }
        )

    write_json(output_dir / "chapters.json", meta)
    return written


def load_chapter_bundle(output_dir: Path) -> list[TranscriptChapter]:
    meta_path = output_dir / "chapters.json"
    if not meta_path.is_file():
        return []
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    chapters: list[TranscriptChapter] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        rel_path = str(entry.get("path", ""))
        text_path = output_dir / rel_path
        if not text_path.is_file():
            continue
        chapters.append(
            TranscriptChapter(
                index=int(entry.get("index", len(chapters) + 1)),
                start_seconds=float(entry.get("start_seconds", 0.0)),
                end_seconds=float(entry.get("end_seconds", 0.0)),
                text=text_path.read_text(encoding="utf-8"),
                path=rel_path,
            )
        )
    return chapters


def visual_for_time_range(
    visual_content: list[dict],
    start_seconds: float,
    end_seconds: float,
    *,
    padding_seconds: float = 30.0,
) -> list[dict]:
    """Return visual captures whose timestamp falls within the chapter window."""
    window_start = max(0.0, start_seconds - padding_seconds)
    window_end = end_seconds + padding_seconds
    matched: list[dict] = []
    for entry in visual_content:
        if not isinstance(entry, dict):
            continue
        ts = parse_timestamp(str(entry.get("timestamp", "")))
        if window_start <= ts <= window_end:
            matched.append(entry)
    return matched
