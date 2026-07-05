"""Post-transcription filters for known Whisper silence hallucinations."""

from __future__ import annotations

import re

# Short phrases Whisper often emits on silence/noise (language -> normalized forms).
_HALLUCINATION_PHRASES: dict[str, frozenset[str]] = {
    "es": frozenset(
        {
            "gracias",
            "gracias.",
            "muchas gracias",
            "muchas gracias.",
            "gracias por ver",
            "gracias por ver.",
            "gracias por su atención",
            "gracias por su atención.",
            "subtítulos realizados por",
            "subtitulos realizados por",
        }
    ),
    "en": frozenset(
        {
            "thank you",
            "thank you.",
            "thanks for watching",
            "thanks for watching.",
            "thanks for listening",
            "thanks for listening.",
            "subtitles by",
            "subtitles by the amara.org community",
        }
    ),
}

_NON_WORD_RE = re.compile(r"[^\w\s.]", re.UNICODE)


def normalize_segment_text(text: str) -> str:
    cleaned = _NON_WORD_RE.sub("", text.strip().lower())
    return cleaned.strip()


def is_hallucination_phrase(text: str, language: str) -> bool:
    norm = normalize_segment_text(text)
    if not norm:
        return True
    phrases = _HALLUCINATION_PHRASES.get(language, frozenset())
    bare = norm.rstrip(".")
    return norm in phrases or bare in phrases


def filter_hallucination_segments(
    segments: list[dict],
    *,
    language: str,
    max_duration_seconds: float = 2.0,
    min_repeat_count: int = 3,
) -> list[dict]:
    """Drop known silence hallucinations and short identical repeated segments."""
    if not segments:
        return segments

    kept: list[dict] = []
    for segment in segments:
        duration = float(segment["end"]) - float(segment["start"])
        text = str(segment["text"])
        if duration <= max_duration_seconds and is_hallucination_phrase(text, language):
            continue
        kept.append(segment)

    if min_repeat_count < 2:
        return kept

    deduped: list[dict] = []
    run_text: str | None = None
    run_segments: list[dict] = []

    def flush_run() -> None:
        nonlocal run_text, run_segments
        if not run_segments:
            return
        if len(run_segments) >= min_repeat_count and run_text is not None:
            max_dur = max(float(s["end"]) - float(s["start"]) for s in run_segments)
            if max_dur <= max_duration_seconds:
                run_text = None
                run_segments = []
                return
        deduped.extend(run_segments)
        run_text = None
        run_segments = []

    for segment in kept:
        duration = float(segment["end"]) - float(segment["start"])
        norm = normalize_segment_text(str(segment["text"]))
        if duration <= max_duration_seconds and norm:
            if norm == run_text:
                run_segments.append(segment)
                continue
            flush_run()
            run_text = norm
            run_segments = [segment]
            continue
        flush_run()
        deduped.append(segment)

    flush_run()
    return deduped


def join_segment_texts(segments: list[dict]) -> str:
    return " ".join(
        str(segment["text"]).strip() for segment in segments if str(segment["text"]).strip()
    )
