"""Tests for output language resolution."""

from __future__ import annotations

from workflow.output_language import (
    default_meeting_topic,
    language_display_name,
    resolve_output_language,
)


def test_resolve_output_language_prefers_output_config() -> None:
    assert (
        resolve_output_language({"output": {"language": "en"}, "whisper": {"language": "es"}})
        == "en"
    )


def test_resolve_output_language_falls_back_to_whisper() -> None:
    assert resolve_output_language({"whisper": {"language": "es"}}) == "es"


def test_language_display_name() -> None:
    assert language_display_name("es") == "Spanish"
    assert language_display_name("en") == "English"


def test_default_meeting_topic_by_language() -> None:
    assert default_meeting_topic("es") == "Reunión"
    assert default_meeting_topic("en") == "Meeting"
