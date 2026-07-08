"""Tests for diarized transcript display formatting."""

from __future__ import annotations

from workflow.transcript_format import segment_has_speakers, segments_to_display_text


def test_segments_to_display_text_plain_without_speakers() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hola."},
        {"start": 2.0, "end": 4.0, "text": "Adiós."},
    ]

    assert segments_to_display_text(segments) == "Hola. Adiós."
    assert segment_has_speakers(segments) is False


def test_segments_to_display_text_merges_same_speaker_lines() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Primera frase.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "text": "Segunda frase.", "speaker": "Speaker 1"},
        {"start": 4.0, "end": 6.0, "text": "Otra voz.", "speaker": "Speaker 2"},
    ]

    text = segments_to_display_text(segments)

    assert text == "[Speaker 1] Primera frase. Segunda frase.\n[Speaker 2] Otra voz."
    assert segment_has_speakers(segments) is True
