"""Structured Extraction via Ollama text model."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from workflow.extraction_grounding import (
    apply_deterministic_grounding,
    dedupe_similar_strings,
)
from workflow.ollama_client import OllamaClient, resolve_ollama_settings
from workflow.output_language import (
    default_meeting_topic,
    language_display_name,
    resolve_output_language,
)
from workflow.summary import render_summary
from workflow.transcript_chapters import (
    TranscriptChapter,
    load_chapter_bundle,
    split_segments_into_chapters,
    visual_for_time_range,
    write_chapter_bundle,
)
from workflow.utils import parse_meeting_date, write_json

logger = logging.getLogger(__name__)

_MAX_TRANSCRIPT_CHARS = 24_000
_MIN_TRANSCRIPT_FOR_EMPTY_RETRY = 5_000
_EXTRACTION_EMPTY_RETRIES = 3

_CLASSIFICATION_RULES = """\
Section definitions (apply strictly):
- topic: one short title for the meeting's main subject — theme or outcome, not a feature spec.
- key_decisions: outcomes the group explicitly agreed or committed to in speech.
  INCLUDE:
  (a) team priority or sequencing with buy-in ("primero lo normativo", "después convenios",
      "reenfoquemos prioridades", "terminar X antes de Y") when others align or confirm;
  (b) explicit agreement on business rules the team accepts ("de acuerdo", "listo", "sí",
      "correcto", "exacto", "acordamos", "quedamos en").
  EXCLUDE:
  (a) a single speaker's feature brainstorm without confirmation ("se me ocurre", "yo creo",
      "podríamos", "sería bueno", "lo que propongo", "habría que") — those go in
      technical_details or open_questions;
  (b) UI field specs or implementation detail — technical_details, not key_decisions.
- action_items: concrete post-meeting tasks with a clear deliverable — a meeting to schedule,
  a diagram to draw, a review to perform, a document to validate.
  Use an imperative or "Name: deliverable" phrasing. owner only when speech assigns a person.
  Do NOT list product requirements, field names, or UI components here.
  BAD: "Crear campo Convenio opcional en historial clínico"
  GOOD: "Sergio: dibujar el flujo de convenio para revisión conjunta"
  GOOD: "Lucho: valorar impacto en cronograma RDA antes del 15"
  Put field/requirement lists in technical_details.
- technical_details: requirements, fields, flows, business rules, and design ideas discussed —
  including unconfirmed proposals and screen-walkthrough content.
- status_updates: material project, customer, or timeline progress — NOT call logistics
  (who joined, screen share, recording, audio/video quality, greetings, small talk).
- blockers_risks: impediments, external dependencies, regulatory uncertainty, migration gaps,
  or delivery risks explicitly discussed (e.g. missing onboarding features blocking a go-live).
- open_questions: unresolved questions explicitly raised (including normative/regulatory doubts).
- next_steps: stated follow-ups (meetings, diagrams, reviews, evaluations) even if unassigned.
"""

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

{_classification_rules}
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
- For action_items: when participants state a deadline, week, or calendar date for a task,
  put it in deadline (ISO date, dd/mm, or spoken phrase like "el lunes"). Leave deadline empty
  when no timing was stated. Do not invent deadlines.
- owner is optional: set only when speech explicitly assigns a person to that task.

Speaker turns (when [Speaker N] or [Name] labels appear):
- Use turn boundaries: proposals (one speaker monologue) vs agreement (another confirms).
- Put only explicit group agreement in key_decisions.

Use empty lists when a section has no information.
Return only the JSON object, no markdown or extra text.
{correction_hint}

Transcript:
{transcript}

Visual captures (supplementary — max {visual_count} shown):
{visual}
"""

_MERGE_EXTRACTION_PROMPT = """\
Merge partial meeting extractions from sequential chapters into one JSON object.

Write all JSON string values in {output_language}.

Return a single JSON object with this exact shape:
{{
  "topic": "short meeting title for the whole meeting",
  "key_decisions": ["..."],
  "action_items": [{{"owner": "name or empty", "task": "...", "deadline": "date or empty"}}],
  "blockers_risks": ["..."],
  "status_updates": ["..."],
  "technical_details": ["..."],
  "open_questions": ["..."],
  "next_steps": ["..."]
}}

Rules:
- Combine chapter findings; remove duplicates and near-duplicates.
{_classification_rules}
- During merge, review key_decisions: demote ONLY items that are clearly a single
  speaker's unconfirmed proposal. Keep team priorities, deferrals, and agreed business rules.
- Keep the most specific wording when two items describe the same fact.
- Do not invent facts that are absent from the chapter extractions or visual captures.
- Preserve action_item deadlines from chapter extractions when merging duplicates.
- Use visual captures only for technical_details and topic context.

Return only the JSON object, no markdown or extra text.
{correction_hint}

Chapter extractions:
{chapters}

Visual captures (supplementary — max {visual_count} shown):
{visual}
"""

_GROUNDING_PROMPT = """Review a merged meeting extraction against chapter transcript text.

Write all JSON string values in {output_language}.

Remove bullets NOT supported by the chapter transcripts below.
Also RECLASSIFY mislabeled items when obvious:
- key_decisions that are only unconfirmed feature specs from one speaker → technical_details
- action_items that are field/requirement lists without a deliverable → technical_details
- status_updates that are call logistics (joins, screen share, AV) → remove
Do not remove valid team priorities or deferrals from key_decisions.
Keep supported, correctly labeled items unchanged. Synthesize one concise topic.

Return only the corrected JSON object matching the same schema. No markdown or prose.

Merged extraction to verify:
{merged}

Chapter transcripts:
{chapters}
"""

_SYSTEM_PROMPT = (
    "You are a JSON extraction API. Output MUST be a single raw JSON object — "
    "no markdown, no headings, no prose before or after the JSON. "
    "Write ALL string values in {output_language} only — never use another language. "
    "Classify strictly: key_decisions need explicit agreement; proposals and requirements "
    "belong in technical_details; action_items need deliverable tasks, not feature lists. "
    "Ground every bullet in transcript speech; do not invent or re-label numbers."
)

_TYPE_PRIORITY = {"whiteboard": 4, "diagram": 3, "slide": 2, "other": 1}

STRUCTURED_VISION_TYPES = frozenset({"whiteboard", "diagram", "slide"})

SHARED_CONTENT_SIGNALS = (
    "whiteboard",
    "pizarra",
    "navegador",
    "pantalla",
    "compartiendo",
    "presentación",
    "figma",
    "powerpoint",
    "diagrama",
    "convenio",
    "parámetros",
    "historia",
    "clínica",
    "laboral",
    "formulario",
    "mockup",
    "prototipo",
    "e-learning",
    "portal",
    "tablero",
    "matriz",
    "flujo",
    "arquitectura",
    "documento",
    "excel",
    "hoja de cálculo",
    "miro",
    "screen share",
)

_BOT_UI_RE = re.compile(r"(?i)read\.ai|meeting notes|otter|fireflies")
_TILE_ONLY_RE = re.compile(
    r"(?i)(?:solo |únicamente )?(?:participantes visibles|avatares|fondo negro|burbuja circular)"
)
_CIRCLE_INITIALS_RE = re.compile(r"(?i)c[ií]rculos?.*(iniciales|siglas|colores)")

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


def sanitize_visual_description(description: str) -> str:
    """Unwrap JSON-shaped vision output and return plain description text."""
    text = description.strip()
    if not text:
        return ""
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            inner = data.get("description")
            if inner is not None and str(inner).strip():
                return str(inner).strip()
    return text


def has_shared_content_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in SHARED_CONTENT_SIGNALS)


def is_low_value_visual_description(description: str, *, frame_type: str = "other") -> bool:
    text = sanitize_visual_description(description)
    if not text:
        return True
    normalized_type = frame_type.split("|")[0].strip().lower()
    if normalized_type in STRUCTURED_VISION_TYPES and len(text) >= 20:
        return False
    if has_shared_content_signal(text):
        return False
    if _BOT_UI_RE.search(text):
        return True
    if _CIRCLE_INITIALS_RE.search(text):
        return True
    if _TILE_ONLY_RE.search(text):
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
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    time_padding_seconds: float = 30.0,
) -> list[dict]:
    if max_frames <= 0:
        return []
    scoped = visual_content
    if start_seconds is not None and end_seconds is not None:
        scoped = visual_for_time_range(
            visual_content,
            start_seconds,
            end_seconds,
            padding_seconds=time_padding_seconds,
        )
    candidates = [
        entry
        for entry in scoped
        if isinstance(entry, dict)
        and not is_low_value_visual_description(
            str(entry.get("description", "")),
            frame_type=str(entry.get("type", "other")),
        )
    ]
    ranked = sorted(
        candidates,
        key=lambda entry: (-_visual_priority(entry), str(entry.get("timestamp", ""))),
    )
    selected = ranked[:max_frames]
    selected.sort(key=lambda entry: str(entry.get("timestamp", "")))
    trimmed: list[dict] = []
    for entry in selected:
        desc = sanitize_visual_description(str(entry.get("description", "")))
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
    visual_time_padding_seconds: float = 30.0,
    visual_start_seconds: float | None = None,
    visual_end_seconds: float | None = None,
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
        start_seconds=visual_start_seconds,
        end_seconds=visual_end_seconds,
        time_padding_seconds=visual_time_padding_seconds,
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
        _classification_rules=_CLASSIFICATION_RULES,
    )


def build_extraction_system_prompt(*, output_language: str = "es") -> str:
    return _SYSTEM_PROMPT.format(output_language=language_display_name(output_language))


def _format_visual_lines(visual_subset: list[dict]) -> str:
    visual_lines = []
    for entry in visual_subset:
        ts = entry.get("timestamp", "")
        kind = entry.get("type", "other")
        desc = entry.get("description", "")
        visual_lines.append(f"- [{ts}] ({kind}) {desc}")
    return "\n".join(visual_lines) if visual_lines else "(no visual captures)"


def build_merge_extraction_prompt(
    chapter_extractions: list[tuple[TranscriptChapter, dict[str, Any]]],
    visual_content: list[dict],
    *,
    output_language: str = "es",
    max_visual_frames: int = 8,
    max_visual_description_chars: int = 400,
    correction_hint: str = "",
) -> str:
    chapter_blocks: list[str] = []
    for chapter, raw in chapter_extractions:
        chapter_blocks.append(
            f"### {chapter.time_range_label}\n{json.dumps(raw, ensure_ascii=False, indent=2)}"
        )
    visual_subset = select_visual_for_extraction(
        visual_content,
        max_frames=max_visual_frames,
        max_description_chars=max_visual_description_chars,
    )
    hint_block = f"\n{correction_hint.strip()}\n" if correction_hint.strip() else ""
    return _MERGE_EXTRACTION_PROMPT.format(
        chapters="\n\n".join(chapter_blocks),
        visual=_format_visual_lines(visual_subset),
        visual_count=len(visual_subset),
        output_language=language_display_name(output_language),
        correction_hint=hint_block,
        _classification_rules=_CLASSIFICATION_RULES,
    )


def build_grounding_prompt(
    merged_raw: dict[str, Any],
    chapters: list[TranscriptChapter],
    *,
    output_language: str = "es",
) -> str:
    chapter_blocks = [f"### {chapter.time_range_label}\n{chapter.text}" for chapter in chapters]
    return _GROUNDING_PROMPT.format(
        merged=json.dumps(merged_raw, ensure_ascii=False, indent=2),
        chapters="\n\n".join(chapter_blocks),
        output_language=language_display_name(output_language),
    )


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = re.sub(r"\s+", " ", item.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return dedupe_similar_strings(result)


def _visual_text_blob(visual_content: list[dict]) -> str:
    parts: list[str] = []
    for entry in visual_content:
        if isinstance(entry, dict):
            desc = str(entry.get("description", "")).strip()
            if desc:
                parts.append(desc)
    return " ".join(parts)


def _apply_list_dedup(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    for key in (
        "key_decisions",
        "blockers_risks",
        "status_updates",
        "technical_details",
        "open_questions",
        "next_steps",
    ):
        result[key] = _dedupe_strings(_as_string_list(result.get(key)))
    return result


def merge_chapter_extractions_deterministic(
    chapter_extractions: list[tuple[TranscriptChapter, dict[str, Any]]],
) -> dict[str, Any]:
    """Fallback merge used in tests when LLM merge is unavailable."""
    merged: dict[str, Any] = {
        "topic": "",
        "key_decisions": [],
        "action_items": [],
        "blockers_risks": [],
        "status_updates": [],
        "technical_details": [],
        "open_questions": [],
        "next_steps": [],
    }
    for _chapter, raw in chapter_extractions:
        if not merged["topic"] and raw.get("topic"):
            merged["topic"] = raw.get("topic")
        for key in (
            "key_decisions",
            "blockers_risks",
            "status_updates",
            "technical_details",
            "open_questions",
            "next_steps",
        ):
            merged[key].extend(_as_string_list(raw.get(key)))
        merged["action_items"].extend(_as_action_items(raw.get("action_items")))
    for key in (
        "key_decisions",
        "blockers_risks",
        "status_updates",
        "technical_details",
        "open_questions",
        "next_steps",
    ):
        merged[key] = _dedupe_strings(merged[key])
    deduped_actions: list[dict[str, str]] = []
    seen_tasks: set[str] = set()
    for item in merged["action_items"]:
        task_key = re.sub(r"\s+", " ", item["task"].strip().lower())
        if not task_key or task_key in seen_tasks:
            continue
        seen_tasks.add(task_key)
        deduped_actions.append(item)
    merged["action_items"] = deduped_actions
    return merged


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
        self._extraction_mode = str(extraction_cfg.get("mode", "auto"))
        self._chapter_trigger_chars = int(
            extraction_cfg.get("chapter_trigger_chars", self._max_transcript_chars)
        )
        self._chapter_target_chars = int(extraction_cfg.get("chapter_target_chars", 8_000))
        self._chapter_min_chars = int(extraction_cfg.get("chapter_min_chars", 1_500))
        self._grounding_check = bool(extraction_cfg.get("grounding_check", True))
        self._deterministic_grounding = bool(extraction_cfg.get("deterministic_grounding", True))
        self._grounding_min_overlap = float(extraction_cfg.get("grounding_min_overlap", 0.34))
        self._grounding_min_token_length = int(extraction_cfg.get("grounding_min_token_length", 4))
        self._visual_time_padding_seconds = float(
            extraction_cfg.get("visual_time_padding_seconds", 30.0)
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
        segments = getattr(transcript, "segments", None)
        if not isinstance(segments, list):
            segments = []
        try:
            raw = self._extract_raw(text, segments, visual_content, output_dir)
            raw = self._post_process_raw(raw, text, visual_content, output_dir)
            cleaned_visual = [
                {
                    **entry,
                    "description": sanitize_visual_description(str(entry.get("description", ""))),
                }
                for entry in visual_content
                if isinstance(entry, dict)
            ]
            extraction = normalize_extraction(
                raw,
                meeting_date=parse_meeting_date(recording_name),
                visual_content=cleaned_visual,
                output_language=self._output_language,
            )
            write_json(output_dir / "extraction.json", extraction)
            (output_dir / "summary.md").write_text(render_summary(extraction), encoding="utf-8")
            return extraction
        finally:
            if unload_after and self._settings.unload_between_stages:
                self.unload()

    def _resolve_extraction_mode(self, text_len: int) -> str:
        if self._extraction_mode == "auto":
            return "map_reduce" if text_len > self._chapter_trigger_chars else "single"
        return self._extraction_mode

    def _extract_raw(
        self,
        text: str,
        segments: list[dict],
        visual_content: list[dict],
        output_dir: Path,
    ) -> dict[str, Any]:
        if self._resolve_extraction_mode(len(text)) == "map_reduce" and segments:
            return self._map_reduce_extraction(segments, visual_content, output_dir)
        return self._run_extraction_pass(text, visual_content, correction_hint="")

    def _post_process_raw(
        self,
        raw: dict[str, Any],
        text: str,
        visual_content: list[dict],
        output_dir: Path,
    ) -> dict[str, Any]:
        if self._retry_on_language_drift and extraction_has_language_drift(
            raw, output_language=self._output_language
        ):
            raw = self._rerun_single_or_merge(
                text,
                visual_content,
                output_dir,
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
            raw = self._rerun_single_or_merge(
                text,
                visual_content,
                output_dir,
                correction_hint=(
                    "CRITICAL: The transcript is long and contains spoken decisions, "
                    "tasks, and follow-ups. Extract key_decisions only when the group "
                    "explicitly agreed; put proposals in technical_details. "
                    "Put deliverable tasks in action_items, not feature specs. "
                    "Do not leave all structured lists empty."
                ),
            )
        if self._deterministic_grounding and text.strip():
            raw = apply_deterministic_grounding(
                raw,
                transcript_text=text,
                visual_text=_visual_text_blob(visual_content),
                min_overlap_ratio=self._grounding_min_overlap,
                min_token_length=self._grounding_min_token_length,
            )
        return _apply_list_dedup(raw)

    def _rerun_single_or_merge(
        self,
        text: str,
        visual_content: list[dict],
        output_dir: Path,
        *,
        correction_hint: str,
    ) -> dict[str, Any]:
        chapters = load_chapter_bundle(output_dir)
        if chapters:
            chapter_extractions = self._load_chapter_extractions(output_dir, chapters)
            if chapter_extractions:
                return self._run_merge_pass(chapter_extractions, visual_content, correction_hint)
        return self._run_extraction_pass(text, visual_content, correction_hint=correction_hint)

    def _map_reduce_extraction(
        self,
        segments: list[dict],
        visual_content: list[dict],
        output_dir: Path,
    ) -> dict[str, Any]:
        chapters = split_segments_into_chapters(
            segments,
            target_chars=self._chapter_target_chars,
            min_chars=self._chapter_min_chars,
        )
        if len(chapters) <= 1:
            only = chapters[0].text if chapters else ""
            return self._run_extraction_pass(only, visual_content, correction_hint="")
        chapters = write_chapter_bundle(output_dir, chapters)
        chapter_extractions: list[tuple[TranscriptChapter, dict[str, Any]]] = []
        partial_dir = output_dir / "extraction" / "chapters"
        partial_dir.mkdir(parents=True, exist_ok=True)
        for chapter in chapters:
            partial_path = partial_dir / f"chapter_{chapter.index:03d}.json"
            chapter_visual = visual_for_time_range(
                visual_content,
                chapter.start_seconds,
                chapter.end_seconds,
                padding_seconds=self._visual_time_padding_seconds,
            )
            if partial_path.is_file():
                raw = json.loads(partial_path.read_text(encoding="utf-8"))
            else:
                raw = self._run_chapter_pass(
                    chapter,
                    chapter_total=len(chapters),
                    visual_content=chapter_visual,
                )
                write_json(partial_path, raw)
            chapter_extractions.append((chapter, raw))
        merged = self._run_merge_pass(chapter_extractions, visual_content)
        if self._grounding_check:
            merged = self._run_grounding_pass(merged, chapters)
        return merged

    def _load_chapter_extractions(
        self,
        output_dir: Path,
        chapters: list[TranscriptChapter],
    ) -> list[tuple[TranscriptChapter, dict[str, Any]]]:
        partial_dir = output_dir / "extraction" / "chapters"
        loaded: list[tuple[TranscriptChapter, dict[str, Any]]] = []
        for chapter in chapters:
            partial_path = partial_dir / f"chapter_{chapter.index:03d}.json"
            if not partial_path.is_file():
                return []
            try:
                raw = json.loads(partial_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
            if isinstance(raw, dict):
                loaded.append((chapter, raw))
        return loaded

    def _run_chapter_pass(
        self,
        chapter: TranscriptChapter,
        *,
        chapter_total: int,
        visual_content: list[dict],
    ) -> dict[str, Any]:
        hint = (
            f"Section {chapter.index} of {chapter_total} ({chapter.time_range_label}). "
            "Extract only facts spoken in this section. "
            "Proposals without group agreement → technical_details, not key_decisions. "
            "Skip call logistics in status_updates."
        )
        return self._run_extraction_pass(
            chapter.text,
            visual_content,
            correction_hint=hint,
            max_transcript_chars=0,
            visual_start_seconds=chapter.start_seconds,
            visual_end_seconds=chapter.end_seconds,
        )

    def _run_merge_pass(
        self,
        chapter_extractions: list[tuple[TranscriptChapter, dict[str, Any]]],
        visual_content: list[dict],
        correction_hint: str = "",
    ) -> dict[str, Any]:
        prompt = _qwen_no_think_prefix(self._model) + build_merge_extraction_prompt(
            chapter_extractions,
            visual_content,
            output_language=self._output_language,
            max_visual_frames=self._max_visual_frames,
            max_visual_description_chars=self._max_visual_description_chars,
            correction_hint=correction_hint,
        )
        return self._chat_json(prompt)

    def _run_grounding_pass(
        self,
        merged_raw: dict[str, Any],
        chapters: list[TranscriptChapter],
    ) -> dict[str, Any]:
        prompt = _qwen_no_think_prefix(self._model) + build_grounding_prompt(
            merged_raw,
            chapters,
            output_language=self._output_language,
        )
        return self._chat_json(prompt)

    def _run_extraction_pass(
        self,
        text: str,
        visual_content: list[dict],
        *,
        correction_hint: str,
        max_transcript_chars: int | None = None,
        visual_start_seconds: float | None = None,
        visual_end_seconds: float | None = None,
    ) -> dict[str, Any]:
        prompt = _qwen_no_think_prefix(self._model) + build_extraction_prompt(
            text,
            visual_content,
            output_language=self._output_language,
            max_transcript_chars=(
                self._max_transcript_chars if max_transcript_chars is None else max_transcript_chars
            ),
            transcript_sampling=self._transcript_sampling,
            head_tail_start_ratio=self._head_tail_start_ratio,
            max_visual_frames=self._max_visual_frames,
            max_visual_description_chars=self._max_visual_description_chars,
            visual_time_padding_seconds=self._visual_time_padding_seconds,
            visual_start_seconds=visual_start_seconds,
            visual_end_seconds=visual_end_seconds,
            correction_hint=correction_hint,
        )
        return self._chat_json(prompt)

    def _chat_json(self, prompt: str) -> dict[str, Any]:
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
                payload["messages"] = [
                    payload["messages"][0],
                    {
                        "role": "user",
                        "content": prompt + json_correction,
                    },
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
