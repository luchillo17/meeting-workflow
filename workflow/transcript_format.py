"""Format Whisper segments for humans and extraction prompts."""

from __future__ import annotations

from workflow.transcript_filters import join_segment_texts


def segment_has_speakers(segments: list[dict]) -> bool:
    return any(isinstance(segment, dict) and segment.get("speaker") for segment in segments)


def segments_to_display_text(segments: list[dict]) -> str:
    """Plain text for transcript.txt; speaker-prefixed lines when diarized."""
    if not segment_has_speakers(segments):
        return join_segment_texts(segments)

    lines: list[str] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if not current_parts:
            return
        speaker = current_speaker or "Speaker ?"
        lines.append(f"[{speaker}] {' '.join(current_parts)}")

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker = str(segment.get("speaker", "")).strip() or "Speaker ?"
        if speaker == current_speaker:
            current_parts.append(text)
        else:
            flush()
            current_speaker = speaker
            current_parts = [text]
    flush()
    return "\n".join(lines)
