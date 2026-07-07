"""Infer display names for diarized speakers from transcript speech patterns.

Always runs after diarization. Names are replaced only when speech evidence is
strong; otherwise labels stay Speaker 1, Speaker 2, …
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

_SPEAKER_LABEL_RE = re.compile(r"^Speaker (\d+)$")

# Discourse markers / domain terms that Whisper often capitalizes at line starts.
_NAME_BLOCKLIST = frozenset(
    {
        "ah",
        "ahí",
        "allá",
        "allí",
        "arl",
        "buenas",
        "bueno",
        "cierto",
        "claro",
        "colombia",
        "correcto",
        "cruz",
        "digamos",
        "entonces",
        "epm",
        "eps",
        "exactamente",
        "glamping",
        "gracias",
        "hola",
        "listo",
        "meditic",
        "mire",
        "no",
        "o",
        "ojo",
        "porque",
        "pero",
        "rips",
        "rda",
        "roja",
        "sería",
        "sí",
        "soy",
        "supongamos",
        "vitalink",
        "yo",
    }
)

_SELF_INTRO_PATTERNS = (
    re.compile(r"\bsoy\s+([A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bme llamo\s+([A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bmi nombre es\s+([A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ-]{1,30})\b", re.IGNORECASE),
)

_VOCATIVE_START_RE = re.compile(
    r"^([A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ-]{2,30})(?:,|\s)",
)

_SELF_INTRO_WEIGHT = 10
_VOCATIVE_WEIGHT = 3
_MIN_ASSIGN_SCORE = 5


def _normalize_name(name: str) -> str:
    cleaned = name.strip().strip(".,;:!?\"'")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def _looks_like_person_name(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(("emos", "amos", "imos", "aba", "ían")):
        return False
    return True


def _is_plausible_name(name: str, *, roster: set[str]) -> bool:
    normalized = _normalize_name(name)
    if len(normalized) < 2:
        return False
    if normalized.lower() in _NAME_BLOCKLIST:
        return False
    if normalized.lower() in roster:
        return True
    if not _looks_like_person_name(normalized):
        return False
    if not normalized[0].isupper():
        return False
    if normalized.isupper() and len(normalized) <= 4:
        return False
    return True


def _speaker_labels(segments: list[dict]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        speaker = str(segment.get("speaker", "")).strip()
        if speaker and _SPEAKER_LABEL_RE.match(speaker) and speaker not in seen:
            labels.append(speaker)
            seen.add(speaker)
    return labels


def _find_self_intro_names(text: str, *, roster: set[str]) -> list[str]:
    found: list[str] = []
    for pattern in _SELF_INTRO_PATTERNS:
        for match in pattern.finditer(text):
            name = _normalize_name(match.group(1))
            if _is_plausible_name(name, roster=roster):
                found.append(name)
    return found


def _find_vocative_name(text: str, *, roster: set[str]) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    match = _VOCATIVE_START_RE.match(stripped)
    if not match:
        return None
    name = _normalize_name(match.group(1))
    if _is_plausible_name(name, roster=roster):
        return name
    return None


def _score_name_candidates(
    segments: list[dict],
    *,
    roster: set[str],
) -> dict[str, dict[str, int]]:
    labels = _speaker_labels(segments)
    if not labels:
        return {}

    scores: dict[str, dict[str, int]] = {label: defaultdict(int) for label in labels}

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        speaker = str(segment.get("speaker", "")).strip()
        if speaker not in scores:
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        for name in _find_self_intro_names(text, roster=roster):
            scores[speaker][name] += _SELF_INTRO_WEIGHT

        vocative = _find_vocative_name(text, roster=roster)
        if vocative:
            for other in labels:
                if other != speaker:
                    scores[other][vocative] += _VOCATIVE_WEIGHT
            next_speaker = None
            for future in segments[index + 1 :]:
                if not isinstance(future, dict):
                    continue
                candidate = str(future.get("speaker", "")).strip()
                if candidate in scores and candidate != speaker:
                    next_speaker = candidate
                    break
            if next_speaker is not None:
                scores[next_speaker][vocative] += _VOCATIVE_WEIGHT

    return scores


def _pick_speaker_map(scores: dict[str, dict[str, int]]) -> dict[str, str]:
    """Map Speaker N labels to names when evidence is strong and unambiguous."""
    assignments: dict[str, str] = {}
    used_names: set[str] = set()

    ranked: list[tuple[int, str, str]] = []
    for label, name_scores in scores.items():
        for name, value in name_scores.items():
            if value >= _MIN_ASSIGN_SCORE:
                ranked.append((value, label, name))
    ranked.sort(reverse=True)

    for value, label, name in ranked:
        if label in assignments or name in used_names:
            continue
        competitors = [
            other_label
            for other_label, other_scores in scores.items()
            if other_label != label and other_scores.get(name, 0) >= value - 1
        ]
        if competitors:
            continue
        assignments[label] = name
        used_names.add(name)

    return assignments


def infer_speaker_names(
    segments: list[dict],
    config: dict[str, Any] | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Try to map Speaker N labels to spoken names; keep Speaker N when not inferable."""
    diar_cfg = (config or {}).get("diarization", {})
    roster_raw = diar_cfg.get("roster") or diar_cfg.get("speaker_hints") or []
    roster = {
        _normalize_name(str(name))
        for name in roster_raw
        if isinstance(name, str) and str(name).strip()
    }

    scores = _score_name_candidates(segments, roster=roster)
    if not scores:
        return [dict(segment) for segment in segments], {}

    speaker_map = _pick_speaker_map(scores)
    if not speaker_map:
        return [dict(segment) for segment in segments], {}

    labeled: list[dict] = []
    for segment in segments:
        updated = dict(segment)
        speaker = str(updated.get("speaker", "")).strip()
        if speaker in speaker_map:
            updated["speaker"] = speaker_map[speaker]
            updated["speaker_id"] = speaker
        labeled.append(updated)

    logger.info("Inferred speaker names: %s", speaker_map)
    return labeled, speaker_map
