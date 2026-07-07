"""Regression spot-checks for Structured Extraction quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workflow.extraction_grounding import compute_grounding_ratio

# Pilot meetings: count/structure guards + domain terms (not golden-file accuracy).
PILOT_MEETING_CHECKS: tuple[dict[str, Any], ...] = (
    {
        "slug_prefix": "funcionalidad-ipss",
        "min_transcript_chars": 20_000,
        "min_decisions": 2,
        "min_action_items": 2,
        "min_chapters": 4,
        "min_visual_frames": 2,
        "must_contain": ("convenio",),
        "transcript_terms": ("convenio", "ips"),
        "must_not_contain": ("read.ai", "otter.ai", "fireflies"),
        "min_grounding_ratio": 0.45,
    },
    {
        "slug_prefix": "revisión-aspectos-relevantes-poc",
        "min_transcript_chars": 40_000,
        "min_decisions": 2,
        "min_action_items": 2,
        "min_chapters": 4,
        "min_visual_frames": 3,
        "must_contain": ("portal",),
        "transcript_terms": ("portal",),
        "must_not_contain": ("read.ai",),
        "min_grounding_ratio": 0.45,
    },
    {
        "slug_prefix": "revision-avances-mvp",
        "min_transcript_chars": 40_000,
        "min_decisions": 2,
        "min_action_items": 2,
        "min_chapters": 4,
        "min_visual_frames": 3,
        "transcript_terms": ("mvp",),
        "must_not_contain": ("read.ai",),
        "min_grounding_ratio": 0.45,
    },
    {
        "slug_prefix": "revision-formato-hc-laboral",
        "min_transcript_chars": 30_000,
        "min_decisions": 1,
        "min_action_items": 2,
        "min_chapters": 4,
        "min_visual_frames": 1,
        "must_contain": ("historia",),
        "transcript_terms": ("historia", "laboral"),
        "must_not_contain": ("read.ai",),
        "min_grounding_ratio": 0.45,
    },
)

_CHAPTER_TRIGGER_CHARS = 12_000


def load_transcript_text(output_dir: Path) -> str:
    transcript_path = output_dir / "transcript.txt"
    if transcript_path.is_file():
        return transcript_path.read_text(encoding="utf-8")
    json_path = output_dir / "transcript.json"
    if not json_path.is_file():
        return ""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    if isinstance(data, dict):
        segments = data.get("segments") or []
    elif isinstance(data, list):
        segments = data
    else:
        return ""
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, dict):
            text = str(segment.get("text", "")).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def load_transcript_chars(output_dir: Path) -> int:
    return len(load_transcript_text(output_dir))


def load_chapter_count(output_dir: Path) -> int:
    chapters_path = output_dir / "chapters.json"
    if not chapters_path.is_file():
        return 0
    try:
        data = json.loads(chapters_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(data) if isinstance(data, list) else 0


def load_visual_frame_count(output_dir: Path) -> int:
    visual_path = output_dir / "visual_content.json"
    if not visual_path.is_file():
        return 0
    try:
        data = json.loads(visual_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(data) if isinstance(data, list) else 0


def load_visual_text(output_dir: Path) -> str:
    visual_path = output_dir / "visual_content.json"
    if not visual_path.is_file():
        return ""
    try:
        data = json.loads(visual_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, list):
        return ""
    parts: list[str] = []
    for entry in data:
        if isinstance(entry, dict):
            desc = str(entry.get("description", "")).strip()
            if desc:
                parts.append(desc)
    return " ".join(parts)


def _extraction_blob(extraction: dict[str, Any]) -> str:
    parts: list[str] = [str(extraction.get("topic", ""))]
    for key in (
        "key_decisions",
        "blockers_risks",
        "status_updates",
        "technical_details",
        "open_questions",
        "next_steps",
    ):
        parts.extend(str(item) for item in extraction.get(key) or [])
    for item in extraction.get("action_items") or []:
        if isinstance(item, dict):
            parts.extend(str(item.get(field, "")) for field in ("owner", "task", "deadline"))
    return " ".join(parts).lower()


def evaluate_extraction(
    extraction: dict[str, Any],
    *,
    checks: dict[str, Any],
    transcript_chars: int = 0,
    transcript_text: str = "",
    chapter_count: int = 0,
    visual_frame_count: int = 0,
    visual_text: str = "",
    grounding_min_overlap: float = 0.34,
) -> list[str]:
    """Return human-readable failure messages; empty list means all checks passed."""
    failures: list[str] = []
    blob = _extraction_blob(extraction)
    topic = str(extraction.get("topic", "")).strip()
    min_chars = int(checks.get("min_transcript_chars", 0))
    long_meeting = bool(min_chars and transcript_chars >= min_chars)

    if long_meeting:
        min_decisions = int(checks.get("min_decisions", 0))
        decisions = extraction.get("key_decisions") or []
        if min_decisions and len(decisions) < min_decisions:
            failures.append(
                f"expected at least {min_decisions} key_decisions, got {len(decisions)}"
            )
        min_actions = int(checks.get("min_action_items", 0))
        actions = extraction.get("action_items") or []
        if min_actions and len(actions) < min_actions:
            failures.append(f"expected at least {min_actions} action_items, got {len(actions)}")
        empty_tasks = sum(
            1
            for item in actions
            if isinstance(item, dict) and not str(item.get("task", "")).strip()
        )
        if actions and empty_tasks:
            failures.append(f"expected action_items with tasks, got {empty_tasks} empty task(s)")

        min_chapters = int(checks.get("min_chapters", 0))
        if min_chapters and transcript_chars >= _CHAPTER_TRIGGER_CHARS:
            if chapter_count < min_chapters:
                failures.append(f"expected at least {min_chapters} chapters, got {chapter_count}")

        min_visual = int(checks.get("min_visual_frames", 0))
        if min_visual and visual_frame_count < min_visual:
            failures.append(
                f"expected at least {min_visual} visual_content frames, got {visual_frame_count}"
            )

    min_topic_len = int(checks.get("min_topic_len", 8))
    if long_meeting and len(topic) < min_topic_len:
        failures.append(f"topic too short ({len(topic)} chars)")

    for needle in checks.get("must_contain", ()):
        if str(needle).lower() not in blob:
            failures.append(f"missing expected term: {needle!r}")

    for needle in checks.get("must_not_contain", ()):
        if str(needle).lower() in blob:
            failures.append(f"found forbidden term: {needle!r}")

    transcript_lower = transcript_text.lower()
    for needle in checks.get("transcript_terms", ()):
        term = str(needle).lower()
        if term in transcript_lower and term not in blob:
            failures.append(f"transcript mentions {needle!r} but extraction does not")

    min_grounding = float(checks.get("min_grounding_ratio", 0))
    if long_meeting and min_grounding > 0 and transcript_text.strip():
        ratio, grounded, total = compute_grounding_ratio(
            extraction,
            transcript_text=transcript_text,
            visual_text=visual_text,
            min_overlap_ratio=grounding_min_overlap,
        )
        if total and ratio < min_grounding:
            failures.append(
                f"grounding ratio {ratio:.2f} below minimum {min_grounding:.2f} "
                f"({grounded}/{total} items)"
            )

    return failures


def evaluate_output_dir(output_dir: Path, *, checks: dict[str, Any]) -> list[str]:
    extraction_path = output_dir / "extraction.json"
    if not extraction_path.is_file():
        return ["missing extraction.json"]
    try:
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["extraction.json is not valid JSON"]
    if not isinstance(extraction, dict):
        return ["extraction.json root must be an object"]

    transcript_text = load_transcript_text(output_dir)
    return evaluate_extraction(
        extraction,
        checks=checks,
        transcript_chars=len(transcript_text),
        transcript_text=transcript_text,
        chapter_count=load_chapter_count(output_dir),
        visual_frame_count=load_visual_frame_count(output_dir),
        visual_text=load_visual_text(output_dir),
    )


def evaluate_pilot_outputs(output_root: Path) -> list[str]:
    """Run pilot spot-checks against all matching Extraction folders."""
    failures: list[str] = []
    if not output_root.is_dir():
        return [f"output root not found: {output_root}"]

    for spec in PILOT_MEETING_CHECKS:
        prefix = str(spec["slug_prefix"])
        matches = sorted(
            path for path in output_root.iterdir() if path.is_dir() and path.name.startswith(prefix)
        )
        if not matches:
            failures.append(f"no output folder for slug prefix {prefix!r}")
            continue
        for folder in matches:
            folder_failures = evaluate_output_dir(folder, checks=spec)
            for message in folder_failures:
                failures.append(f"{folder.name}: {message}")
    return failures
