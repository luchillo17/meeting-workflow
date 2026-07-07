"""Regression spot-checks for Structured Extraction quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Slug prefix checks for pilot meetings (no golden files — substring/count guards).
PILOT_MEETING_CHECKS: tuple[dict[str, Any], ...] = (
    {
        "slug_prefix": "funcionalidad-ipss",
        "min_transcript_chars": 20_000,
        "min_decisions": 1,
        "must_contain": ("convenio",),
    },
    {
        "slug_prefix": "revisión-aspectos-relevantes-poc",
        "min_transcript_chars": 40_000,
        "min_decisions": 2,
        "must_contain": ("portal",),
    },
    {
        "slug_prefix": "revision-avances-mvp",
        "min_transcript_chars": 40_000,
        "min_decisions": 2,
    },
    {
        "slug_prefix": "revision-formato-hc-laboral",
        "min_transcript_chars": 30_000,
        "min_decisions": 1,
    },
)


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
) -> list[str]:
    """Return human-readable failure messages; empty list means all checks passed."""
    failures: list[str] = []
    blob = _extraction_blob(extraction)

    min_chars = int(checks.get("min_transcript_chars", 0))
    if min_chars and transcript_chars >= min_chars:
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

    for needle in checks.get("must_contain", ()):
        if str(needle).lower() not in blob:
            failures.append(f"missing expected term: {needle!r}")

    for needle in checks.get("must_not_contain", ()):
        if str(needle).lower() in blob:
            failures.append(f"found forbidden term: {needle!r}")

    return failures


def evaluate_output_dir(output_dir: Path, *, checks: dict[str, Any]) -> list[str]:
    extraction_path = output_dir / "extraction.json"
    transcript_path = output_dir / "transcript.txt"
    if not extraction_path.is_file():
        return ["missing extraction.json"]
    try:
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["extraction.json is not valid JSON"]
    if not isinstance(extraction, dict):
        return ["extraction.json root must be an object"]

    transcript_chars = 0
    if transcript_path.is_file():
        transcript_chars = len(transcript_path.read_text(encoding="utf-8"))

    return evaluate_extraction(extraction, checks=checks, transcript_chars=transcript_chars)


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
