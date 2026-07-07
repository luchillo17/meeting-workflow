"""Deterministic transcript grounding for Structured Extraction bullets."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

_SPANISH_STOPWORDS = frozenset(
    """
    el la los las un una unos unas de del al lo le les que y o en con por para sin sobre
    como mas muy tambien este esta estos estas ese esa esos esas aqui hay ser es son fue
    serán debe deben puede pueden cuando donde quien cual cuales todo todos toda todas
    """.split()
)

_SPEECH_GROUNDED_KEYS = (
    "key_decisions",
    "blockers_risks",
    "status_updates",
    "open_questions",
    "next_steps",
)

_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def normalize_for_grounding(text: str) -> str:
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def grounding_tokens(
    text: str,
    *,
    min_token_length: int = 4,
    include_short: bool = False,
) -> list[str]:
    """Extract content tokens for overlap checks (Spanish-aware, accent-stripped)."""
    normalized = normalize_for_grounding(text)
    raw_tokens = _TOKEN_RE.findall(normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        if token in _SPANISH_STOPWORDS:
            continue
        if len(token) >= min_token_length or (include_short and len(token) >= 3):
            tokens.append(token)
    if tokens:
        return tokens
    # Very short bullets: fall back to any non-stopword token length >= 3
    return [token for token in raw_tokens if token not in _SPANISH_STOPWORDS and len(token) >= 3]


def token_overlap_ratio(tokens: list[str], source_text: str) -> float:
    if not tokens:
        return 1.0
    source = normalize_for_grounding(source_text)
    hits = sum(1 for token in tokens if token in source)
    return hits / len(tokens)


def is_grounded_in_source(
    item: str,
    source_text: str,
    *,
    min_overlap_ratio: float = 0.34,
    min_token_length: int = 4,
) -> bool:
    text = item.strip()
    if not text:
        return False
    tokens = grounding_tokens(text, min_token_length=min_token_length)
    if not tokens:
        tokens = grounding_tokens(text, min_token_length=3, include_short=True)
    if not tokens:
        return True
    return token_overlap_ratio(tokens, source_text) >= min_overlap_ratio


def is_deadline_grounded(deadline: str, source_text: str) -> bool:
    deadline = deadline.strip()
    if not deadline:
        return True
    source = normalize_for_grounding(source_text)
    deadline_norm = normalize_for_grounding(deadline)
    if deadline_norm in source:
        return True
    digits = re.findall(r"\d+", deadline_norm)
    if digits and any(digit in source for digit in digits if len(digit) >= 2):
        return True
    # Weekday / relative deadline words often paraphrased — require at least one token hit
    tokens = grounding_tokens(deadline_norm, min_token_length=3, include_short=True)
    return bool(tokens) and token_overlap_ratio(tokens, source_text) >= 0.34


def compute_grounding_ratio(
    extraction: dict[str, Any],
    *,
    transcript_text: str,
    visual_text: str = "",
    min_overlap_ratio: float = 0.34,
    min_token_length: int = 4,
) -> tuple[float, int, int]:
    """Return (ratio, grounded_count, total_count) for speech-derived fields."""
    grounded = 0
    total = 0
    tech_source = f"{transcript_text}\n{visual_text}".strip()

    for key in _SPEECH_GROUNDED_KEYS:
        for item in extraction.get(key) or []:
            text = str(item).strip()
            if not text:
                continue
            total += 1
            if is_grounded_in_source(
                text,
                transcript_text,
                min_overlap_ratio=min_overlap_ratio,
                min_token_length=min_token_length,
            ):
                grounded += 1

    for item in extraction.get("action_items") or []:
        if not isinstance(item, dict):
            continue
        task = str(item.get("task", "")).strip()
        if not task:
            continue
        total += 1
        if is_grounded_in_source(
            task,
            transcript_text,
            min_overlap_ratio=min_overlap_ratio,
            min_token_length=min_token_length,
        ):
            grounded += 1
        deadline = str(item.get("deadline", "")).strip()
        if deadline:
            total += 1
            if is_deadline_grounded(deadline, transcript_text):
                grounded += 1

    for item in extraction.get("technical_details") or []:
        text = str(item).strip()
        if not text:
            continue
        total += 1
        if is_grounded_in_source(
            text,
            tech_source,
            min_overlap_ratio=min_overlap_ratio,
            min_token_length=min_token_length,
        ):
            grounded += 1

    if total == 0:
        return 1.0, 0, 0
    return grounded / total, grounded, total


def apply_deterministic_grounding(
    raw: dict[str, Any],
    *,
    transcript_text: str,
    visual_text: str = "",
    min_overlap_ratio: float = 0.34,
    min_token_length: int = 4,
) -> dict[str, Any]:
    """Drop bullets whose content tokens are not supported by transcript (or vision for tech)."""
    result = dict(raw)
    tech_source = f"{transcript_text}\n{visual_text}".strip()

    for key in _SPEECH_GROUNDED_KEYS:
        kept: list[str] = []
        for item in raw.get(key) or []:
            text = str(item).strip()
            if not text:
                continue
            if is_grounded_in_source(
                text,
                transcript_text,
                min_overlap_ratio=min_overlap_ratio,
                min_token_length=min_token_length,
            ):
                kept.append(text)
            else:
                logger.debug("Dropped ungrounded %s: %s", key, text[:80])
        result[key] = kept

    kept_actions: list[dict[str, str]] = []
    for entry in raw.get("action_items") or []:
        if not isinstance(entry, dict):
            continue
        task = str(entry.get("task", "")).strip()
        if not task:
            continue
        if not is_grounded_in_source(
            task,
            transcript_text,
            min_overlap_ratio=min_overlap_ratio,
            min_token_length=min_token_length,
        ):
            logger.debug("Dropped ungrounded action_item: %s", task[:80])
            continue
        owner = str(entry.get("owner", "")).strip()
        if owner and not is_grounded_in_source(
            owner,
            transcript_text,
            min_overlap_ratio=0.34,
            min_token_length=3,
        ):
            owner = ""
        deadline = str(entry.get("deadline", "")).strip()
        if deadline and not is_deadline_grounded(deadline, transcript_text):
            logger.debug("Cleared ungrounded deadline on task: %s", task[:80])
            deadline = ""
        kept_actions.append({"owner": owner, "task": task, "deadline": deadline})
    result["action_items"] = kept_actions

    kept_tech: list[str] = []
    for item in raw.get("technical_details") or []:
        text = str(item).strip()
        if not text:
            continue
        if is_grounded_in_source(
            text,
            tech_source,
            min_overlap_ratio=min_overlap_ratio,
            min_token_length=min_token_length,
        ):
            kept_tech.append(text)
        else:
            logger.debug("Dropped ungrounded technical_details: %s", text[:80])
    result["technical_details"] = kept_tech

    return result


def dedupe_similar_strings(
    items: list[str],
    *,
    similarity_threshold: float = 0.72,
) -> list[str]:
    """Drop near-duplicate bullets using token-set Jaccard similarity."""
    result: list[str] = []
    seen_token_sets: list[set[str]] = []

    for item in items:
        text = item.strip()
        if not text:
            continue
        tokens = set(grounding_tokens(text, min_token_length=3, include_short=True))
        if not tokens:
            result.append(text)
            continue
        duplicate = False
        for prior in seen_token_sets:
            union = tokens | prior
            if not union:
                continue
            jaccard = len(tokens & prior) / len(union)
            if jaccard >= similarity_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        seen_token_sets.append(tokens)
        result.append(text)
    return result
