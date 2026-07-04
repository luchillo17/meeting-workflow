"""Output language resolution for LLM prompts (prompts stay English; output follows config)."""

from __future__ import annotations

_LANGUAGE_NAMES: dict[str, str] = {
    "es": "Spanish",
    "en": "English",
}

_DEFAULT_TOPICS: dict[str, str] = {
    "es": "Reunión",
    "en": "Meeting",
}


def resolve_output_language(config: dict) -> str:
    """ISO 639-1 code for extraction/vision text output."""
    configured = config.get("output", {}).get("language")
    if configured:
        return str(configured)
    return str(config.get("whisper", {}).get("language", "es"))


def language_display_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


def default_meeting_topic(code: str) -> str:
    return _DEFAULT_TOPICS.get(code, "Meeting")
