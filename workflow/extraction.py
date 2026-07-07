"""Structured Extraction via Ollama text model."""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

_MAX_TRANSCRIPT_CHARS = 24_000
_MIN_TRANSCRIPT_FOR_EMPTY_RETRY = 5_000
_EXTRACTION_EMPTY_RETRIES = 3

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

Source priority (mandatory):
- key_decisions, action_items, blockers_risks, status_updates, open_questions, and next_steps
  MUST come from what was explicitly SAID in the transcript.
- Do NOT treat UI walkthroughs, screen descriptions, or tool names visible on screen as decisions
  unless a participant explicitly agreed to them in speech.
- Visual captures are supplementary: use them only for technical_details and to clarify the topic.
  Never invent decisions or action items from visuals alone.

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
{correction_hint}

Transcript:
{transcript}

Visual captures (supplementary — max {visual_count} shown):
{visual}
"""

_SYSTEM_PROMPT = (
    "You are a JSON extraction API. Output MUST be a single raw JSON object — "
    "no markdown, no headings, no prose before or after the JSON. "
    "Write ALL string values in {output_language} only — never use another language. "
    "Ground decisions and action items in speech from the transcript; "
    "do not invent or re-label numbers."
)

_TYPE_PRIORITY = {"whiteboard": 4, "diagram": 3, "slide": 2, "other": 1}

_LOW_VALUE_VISUAL_RE = re.compile(
    r"(?i)(read\.ai|meeting notes|otter|fireflies|"
    r"c[ií]rculos?.*(iniciales|siglas|colores)|participantes visibles|"
    r"fondo negro.*foto circular|burbuja circular|avatar y nombre asociado)"
)

_EN_MARKERS = re.compile(
    r"\b(the|and|with|will|should|includes|portal|interface|section|users)\b", re.I
)
_ES_MARKERS = re.compile(
    r"\b(el|la|los|las|de|para|con|reunión|decisión|acordamos|pendiente)\b", re.I
)
_ES_CHARS = re.compile(r"[áéíóúñ¿¡]", re.I)


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


def is_low_value_visual_description(description: str) -> bool:
    text = description.strip()
    if not text:
        return True
    if _LOW_VALUE_VISUAL_RE.search(text):
        shared_signals = (
            "whiteboard",
            "pizarra",
            "navegador",
            "pantalla",
            "figma",
            "powerpoint",
            "diagrama",
            "convenio",
            "parámetros",
        )
        lowered = text.lower()
        if not any(signal in lowered for signal in shared_signals):
            return True
    return False


def _visual_priority(entry: dict) -> int:
    frame_type = str(entry.get("type", "other")).split("|")[0].strip().lower()
    return _TYPE_PRIORITY.get(frame_type, 1)


def select_visual_for_extraction(
    visual_content: list[dict],
    *,
    max_frames: int = 8,
    max_description_chars: int = 400,
) -> list[dict]:
    if max_frames <= 0:
        return []
    candidates = [
        entry
        for entry in visual_content
        if isinstance(entry, dict)
        and not is_low_value_visual_description(str(entry.get("description", "")))
    ]
    ranked = sorted(
        candidates,
        key=lambda entry: (-_visual_priority(entry), str(entry.get("timestamp", ""))),
    )
    selected = ranked[:max_frames]
    selected.sort(key=lambda entry: str(entry.get("timestamp", "")))
    trimmed: list[dict] = []
    for entry in selected:
        desc = str(entry.get("description", "")).strip()
        if max_description_chars > 0 and len(desc) > max_description_chars:
            desc = desc[:max_description_chars] + "..."
        trimmed.append(
            {
                "timestamp": entry.get("timestamp", ""),
                "type": entry.get("type", "other"),
                "description": desc,
            }
        )
    return trimmed


def sample_transcript_text(
    text: str,
    max_chars: int,
    *,
    sampling: str = "head_tail",
    head_ratio: float = 0.7,
) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if sampling != "head_tail":
        return text[:max_chars] + "\n[... transcript truncated ...]"

    marker = "\n[... middle omitted ...]\n"
    tail_chars = max(1_000, int(max_chars * (1.0 - head_ratio)))
    head_chars = max_chars - tail_chars - len(marker)
    if head_chars < 1_000:
        head_chars = max_chars - len(marker) - 1_000
        tail_chars = 1_000
    return text[:head_chars] + marker + text[-tail_chars:]


def _extraction_string_fields(raw: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    topic = raw.get("topic")
    if topic:
        fields.append(str(topic))
    for key in (
        "key_decisions",
        "blockers_risks",
        "status_updates",
        "technical_details",
        "open_questions",
        "next_steps",
    ):
        fields.extend(str(item) for item in _coerce_to_list(raw.get(key)))
    for item in _coerce_to_list(raw.get("action_items")):
        if isinstance(item, dict):
            fields.extend(
                str(item.get(field, ""))
                for field in ("owner", "task", "deadline")
                if item.get(field)
            )
    return [field for field in fields if field.strip()]


def extraction_has_language_drift(raw: dict[str, Any], *, output_language: str) -> bool:
    combined = " ".join(_extraction_string_fields(raw))
    if not combined.strip():
        return False
    if output_language == "es":
        es_score = len(_ES_MARKERS.findall(combined)) + len(_ES_CHARS.findall(combined)) * 2
        en_score = len(_EN_MARKERS.findall(combined))
        return en_score > es_score and en_score >= 3
    if output_language == "en":
        es_score = len(_ES_MARKERS.findall(combined)) + len(_ES_CHARS.findall(combined))
        en_score = len(_EN_MARKERS.findall(combined))
        return es_score > en_score and es_score >= 3
    return False


def extraction_is_structurally_empty(raw: dict[str, Any]) -> bool:
    structured_keys = (
        "key_decisions",
        "action_items",
        "blockers_risks",
        "status_updates",
        "technical_details",
        "open_questions",
        "next_steps",
    )
    return all(not _coerce_to_list(raw.get(key)) for key in structured_keys)


def build_extraction_prompt(
    transcript_text: str,
    visual_content: list[dict],
    *,
    output_language: str = "es",
    max_transcript_chars: int = _MAX_TRANSCRIPT_CHARS,
    transcript_sampling: str = "head_tail",
    head_tail_start_ratio: float = 0.7,
    max_visual_frames: int = 8,
    max_visual_description_chars: int = 400,
    correction_hint: str = "",
) -> str:
    text = sample_transcript_text(
        transcript_text,
        max_transcript_chars,
        sampling=transcript_sampling,
        head_ratio=head_tail_start_ratio,
    )
    visual_subset = select_visual_for_extraction(
        visual_content,
        max_frames=max_visual_frames,
        max_description_chars=max_visual_description_chars,
    )
    visual_lines = []
    for entry in visual_subset:
        ts = entry.get("timestamp", "")
        kind = entry.get("type", "other")
        desc = entry.get("description", "")
        visual_lines.append(f"- [{ts}] ({kind}) {desc}")
    visual = "\n".join(visual_lines) if visual_lines else "(no visual captures)"
    hint_block = f"\n{correction_hint.strip()}\n" if correction_hint.strip() else ""
    return _EXTRACTION_PROMPT.format(
        transcript=text,
        visual=visual,
        visual_count=len(visual_subset),
        output_language=language_display_name(output_language),
        correction_hint=hint_block,
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


def _qwen_no_think_prefix(model: str) -> str:
    return "/no_think\n" if "qwen3" in model.lower() else ""


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
        self._model = ollama_cfg.get("text_model", "qwen3.5:9b")
        self._max_transcript_chars = int(
            extraction_cfg.get("max_transcript_chars", _MAX_TRANSCRIPT_CHARS)
        )
        self._transcript_sampling = str(extraction_cfg.get("transcript_sampling", "head_tail"))
        self._head_tail_start_ratio = float(extraction_cfg.get("head_tail_start_ratio", 0.7))
        self._max_visual_frames = int(extraction_cfg.get("max_visual_frames", 8))
        self._max_visual_description_chars = int(
            extraction_cfg.get("max_visual_description_chars", 400)
        )
        self._retry_on_language_drift = bool(extraction_cfg.get("retry_on_language_drift", True))
        self._retry_on_empty_extraction = bool(
            extraction_cfg.get("retry_on_empty_extraction", True)
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
        *,
        unload_after: bool = True,
    ) -> dict:
        text = str(getattr(transcript, "text", ""))
        try:
            raw = self._run_extraction_pass(text, visual_content, correction_hint="")
            if self._retry_on_language_drift and extraction_has_language_drift(
                raw, output_language=self._output_language
            ):
                raw = self._run_extraction_pass(
                    text,
                    visual_content,
                    correction_hint=(
                        f"CRITICAL: Your previous response used the wrong language. "
                        f"Rewrite ALL string values in "
                        f"{language_display_name(self._output_language)} only."
                    ),
                )
            if (
                self._retry_on_empty_extraction
                and len(text) >= _MIN_TRANSCRIPT_FOR_EMPTY_RETRY
                and extraction_is_structurally_empty(raw)
            ):
                raw = self._run_extraction_pass(
                    text,
                    visual_content,
                    correction_hint=(
                        "CRITICAL: The transcript is long and contains spoken decisions, "
                        "tasks, and follow-ups. Extract key_decisions, action_items, "
                        "blockers_risks, and next_steps from what participants SAID. "
                        "Do not leave all structured lists empty."
                    ),
                )
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
            if unload_after and self._settings.unload_between_stages:
                self.unload()

    def _run_extraction_pass(
        self,
        text: str,
        visual_content: list[dict],
        *,
        correction_hint: str,
    ) -> dict[str, Any]:
        prompt = _qwen_no_think_prefix(self._model) + build_extraction_prompt(
            text,
            visual_content,
            output_language=self._output_language,
            max_transcript_chars=self._max_transcript_chars,
            transcript_sampling=self._transcript_sampling,
            head_tail_start_ratio=self._head_tail_start_ratio,
            max_visual_frames=self._max_visual_frames,
            max_visual_description_chars=self._max_visual_description_chars,
            correction_hint=correction_hint,
        )
        payload = {
            "model": self._model,
            "stream": False,
            "format": extraction_json_schema(),
            "options": {"num_predict": 4096, "num_ctx": 32768},
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
        json_correction = ""
        for attempt in range(1, _EXTRACTION_EMPTY_RETRIES + 1):
            if json_correction:
                prompt = _qwen_no_think_prefix(self._model) + build_extraction_prompt(
                    text,
                    visual_content,
                    output_language=self._output_language,
                    max_transcript_chars=self._max_transcript_chars,
                    transcript_sampling=self._transcript_sampling,
                    head_tail_start_ratio=self._head_tail_start_ratio,
                    max_visual_frames=self._max_visual_frames,
                    max_visual_description_chars=self._max_visual_description_chars,
                    correction_hint=correction_hint + json_correction,
                )
                payload["messages"] = [
                    payload["messages"][0],
                    {"role": "user", "content": prompt},
                ]
            if self._chat_fn is not None:
                response = self._chat_fn(payload)
            else:
                response = self._client.chat(payload)
            content = response.get("message", {}).get("content", "")
            if not content.strip():
                logger.warning(
                    "Empty extraction response (attempt %d/%d)",
                    attempt,
                    _EXTRACTION_EMPTY_RETRIES,
                )
                json_correction = (
                    "\nCRITICAL: Return ONLY a raw JSON object matching the schema. "
                    "No markdown and no explanatory text."
                )
                continue
            try:
                return _parse_json_object(content)
            except ValueError as exc:
                logger.warning(
                    "Invalid extraction JSON (attempt %d/%d): %s",
                    attempt,
                    _EXTRACTION_EMPTY_RETRIES,
                    exc,
                )
                json_correction = (
                    "\nCRITICAL: Your previous reply was not valid JSON. "
                    "Return ONLY a raw JSON object matching the schema. "
                    "No markdown, headings, or prose."
                )
        raise RuntimeError("Empty extraction response from Ollama")

    def unload(self) -> None:
        """Release the text model from VRAM."""
        if self._settings.unload_between_stages:
            self._unload(self._model)
