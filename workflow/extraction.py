"""Structured Extraction via Ollama text model."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from workflow.ollama_client import OllamaClient, resolve_ollama_settings
from workflow.output_language import (
    default_meeting_topic,
    language_display_name,
    resolve_output_language,
)
from workflow.summary import render_summary
from workflow.utils import parse_meeting_date, write_json

_MAX_TRANSCRIPT_CHARS = 24_000

_EXTRACTION_PROMPT = """You summarize meetings from a transcript and optional visual captures.

Write all JSON string values in {output_language}.

Return a single JSON object with this exact shape:
{{
  "topic": "short meeting title",
  "key_decisions": ["decision 1", "..."],
  "action_items": [{{"owner": "name or empty", "task": "...", "deadline": "date or empty"}}],
  "blockers_risks": ["..."],
  "status_updates": ["..."],
  "technical_details": ["..."],
  "open_questions": ["..."],
  "next_steps": ["..."]
}}

Fidelity rules (mandatory):
- Include only facts explicitly stated in the transcript or visual captures.
  Do not invent names, dates, numbers, or commitments.
- When numbers appear, keep what they measure (progress, budget, timeline, headcount,
  ownership, etc.). Do not move a figure from one topic to another.
- Do not infer unstated quantities, roles, or deadlines. If something was not said,
  omit it or add it to open_questions.
- Do not split or multiply a group-level figure across individuals unless the
  transcript gives that breakdown.
- Ignore meeting notetaker bots and UI labels (e.g. read.ai) as participants.
  Only include people explicitly named in speech.

Use empty lists when a section has no information.
Return only the JSON object, no markdown or extra text.

Transcript:
{transcript}

Visual captures:
{visual}
"""

_SYSTEM_PROMPT = (
    "You summarize meetings. Respond with valid JSON only. "
    "Write ALL string values in {output_language} only — never use another language. "
    "Ground every fact in the transcript; do not invent or re-label numbers."
)


def extraction_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "key_decisions": {"type": "array", "items": {"type": "string"}},
            "action_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "task": {"type": "string"},
                        "deadline": {"type": "string"},
                    },
                    "required": ["owner", "task", "deadline"],
                },
            },
            "blockers_risks": {"type": "array", "items": {"type": "string"}},
            "status_updates": {"type": "array", "items": {"type": "string"}},
            "technical_details": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
            "next_steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "topic",
            "key_decisions",
            "action_items",
            "blockers_risks",
            "status_updates",
            "technical_details",
            "open_questions",
            "next_steps",
        ],
    }


def build_extraction_prompt(
    transcript_text: str,
    visual_content: list[dict],
    *,
    output_language: str = "es",
    max_transcript_chars: int = _MAX_TRANSCRIPT_CHARS,
) -> str:
    text = transcript_text.strip()
    if max_transcript_chars > 0 and len(text) > max_transcript_chars:
        text = text[:max_transcript_chars] + "\n[... transcript truncated ...]"
    visual_lines = []
    for entry in visual_content:
        ts = entry.get("timestamp", "")
        kind = entry.get("type", "other")
        desc = entry.get("description", "")
        visual_lines.append(f"- [{ts}] ({kind}) {desc}")
    visual = "\n".join(visual_lines) if visual_lines else "(no visual captures)"
    return _EXTRACTION_PROMPT.format(
        transcript=text,
        visual=visual,
        output_language=language_display_name(output_language),
    )


def build_extraction_system_prompt(*, output_language: str = "es") -> str:
    return _SYSTEM_PROMPT.format(output_language=language_display_name(output_language))


def _parse_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    preview = content[:200].replace("\n", " ")
    raise ValueError(f"Ollama response did not contain valid extraction JSON. Preview: {preview!r}")


def _coerce_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _optional_text(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text == "None":
        return default
    return text


def _as_string_list(value: Any) -> list[str]:
    items: list[str] = []
    for entry in _coerce_to_list(value):
        if isinstance(entry, dict):
            text = entry.get("text") or entry.get("value") or entry.get("description")
            if text is not None:
                text = str(text).strip()
                if text:
                    items.append(text)
            continue
        text = _optional_text(entry)
        if text:
            items.append(text)
    return items


def _as_action_items(value: Any) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for entry in _coerce_to_list(value):
        if isinstance(entry, dict):
            items.append(
                {
                    "owner": _optional_text(entry.get("owner")),
                    "task": _optional_text(entry.get("task")),
                    "deadline": _optional_text(entry.get("deadline")),
                }
            )
        else:
            text = _optional_text(entry)
            if text:
                items.append({"owner": "", "task": text, "deadline": ""})
    return [item for item in items if item["task"]]


def normalize_extraction(
    raw: dict[str, Any],
    *,
    meeting_date: str | None,
    visual_content: list[dict],
    output_language: str = "es",
) -> dict[str, Any]:
    fallback_topic = default_meeting_topic(output_language)
    return {
        "meeting_date": meeting_date,
        "topic": _optional_text(raw.get("topic"), default=fallback_topic) or fallback_topic,
        "key_decisions": _as_string_list(raw.get("key_decisions")),
        "action_items": _as_action_items(raw.get("action_items")),
        "blockers_risks": _as_string_list(raw.get("blockers_risks")),
        "status_updates": _as_string_list(raw.get("status_updates")),
        "technical_details": _as_string_list(raw.get("technical_details")),
        "open_questions": _as_string_list(raw.get("open_questions")),
        "next_steps": _as_string_list(raw.get("next_steps")),
        "visual_content": visual_content,
    }


def extraction_schema_keys() -> tuple[str, ...]:
    return (
        "meeting_date",
        "topic",
        "key_decisions",
        "action_items",
        "blockers_risks",
        "status_updates",
        "technical_details",
        "open_questions",
        "next_steps",
        "visual_content",
    )


class OllamaStructuredExtractor:
    def __init__(
        self,
        config: dict,
        *,
        chat_fn: Callable[[dict], dict] | None = None,
        unload_fn: Callable[[str], None] | None = None,
    ) -> None:
        ollama_cfg = config.get("ollama", {})
        extraction_cfg = config.get("extraction", {})
        self._settings = resolve_ollama_settings(config)
        self._client = OllamaClient(self._settings)
        self._model = ollama_cfg.get("text_model", "qwen2.5:7b")
        self._max_transcript_chars = int(
            extraction_cfg.get("max_transcript_chars", _MAX_TRANSCRIPT_CHARS)
        )
        self._output_language = resolve_output_language(config)
        self._chat_fn = chat_fn
        self._unload = unload_fn or self._client.unload

    def build_extraction(
        self,
        transcript: object,
        visual_content: list[dict],
        recording_name: str,
        output_dir: Path,
    ) -> dict:
        text = str(getattr(transcript, "text", ""))
        prompt = build_extraction_prompt(
            text,
            visual_content,
            output_language=self._output_language,
            max_transcript_chars=self._max_transcript_chars,
        )
        payload = {
            "model": self._model,
            "stream": False,
            "format": extraction_json_schema(),
            "messages": [
                {
                    "role": "system",
                    "content": build_extraction_system_prompt(
                        output_language=self._output_language
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        try:
            if self._chat_fn is not None:
                response = self._chat_fn(payload)
            else:
                response = self._client.chat(payload, keep_alive=0)
            content = response.get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("Empty extraction response from Ollama")
            raw = _parse_json_object(content)
            extraction = normalize_extraction(
                raw,
                meeting_date=parse_meeting_date(recording_name),
                visual_content=visual_content,
                output_language=self._output_language,
            )
            write_json(output_dir / "extraction.json", extraction)
            (output_dir / "summary.md").write_text(render_summary(extraction), encoding="utf-8")
            return extraction
        finally:
            if self._settings.unload_between_stages:
                self._unload(self._model)
