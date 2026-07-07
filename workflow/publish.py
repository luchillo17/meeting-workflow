"""Publish extraction briefs to docs/meetings for Cursor context."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from workflow.summary import render_summary
from workflow.utils import parse_meeting_date, utc_now_iso, write_json

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DATE_SUFFIX_RE = re.compile(r"-20\d{6}.*$")


def short_publish_slug(folder_name: str, *, max_len: int = 50) -> str:
    slug = _DATE_SUFFIX_RE.sub("", folder_name).strip("-")
    return (slug or folder_name)[:max_len]


def publish_basename(output_dir: Path, extraction: dict[str, Any]) -> str:
    meeting_date = (
        extraction.get("meeting_date") or parse_meeting_date(output_dir.name) or "unknown"
    )
    slug = short_publish_slug(output_dir.name)
    return f"{meeting_date}-{slug}"


def build_publish_markdown(
    extraction: dict[str, Any],
    *,
    summary_body: str,
    source_slug: str,
) -> str:
    frontmatter = {
        "meeting_date": extraction.get("meeting_date"),
        "topic": extraction.get("topic"),
        "source_slug": source_slug,
        "published_at": utc_now_iso(),
    }
    yaml_block = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    body = summary_body.strip() or render_summary(extraction)
    return f"---\n{yaml_block}\n---\n\n{body.rstrip()}\n"


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta = yaml.safe_load(match.group(1))
    body = text[match.end() :].lstrip("\n")
    return (meta if isinstance(meta, dict) else {}), body


def load_extraction_bundle(output_dir: Path) -> tuple[dict[str, Any], str]:
    extraction_path = output_dir / "extraction.json"
    if not extraction_path.is_file():
        raise FileNotFoundError(f"missing extraction.json in {output_dir}")
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if not isinstance(extraction, dict):
        raise ValueError(f"extraction.json root must be an object in {output_dir}")
    summary_path = output_dir / "summary.md"
    summary_body = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
    return extraction, summary_body


def publish_output_dir(
    output_dir: Path,
    meetings_dir: Path,
    *,
    include_json: bool = False,
) -> Path:
    """Write agent brief (+ optional JSON sidecar) and refresh index.md."""
    extraction, summary_body = load_extraction_bundle(output_dir)
    meetings_dir.mkdir(parents=True, exist_ok=True)
    basename = publish_basename(output_dir, extraction)
    markdown_path = meetings_dir / f"{basename}.md"
    markdown_path.write_text(
        build_publish_markdown(
            extraction,
            summary_body=summary_body,
            source_slug=output_dir.name,
        ),
        encoding="utf-8",
    )
    if include_json:
        write_json(meetings_dir / f"{basename}.json", extraction)
    rebuild_meetings_index(meetings_dir)
    return markdown_path


def rebuild_meetings_index(meetings_dir: Path) -> None:
    entries: list[tuple[str, str, str, str]] = []
    for path in sorted(meetings_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        meeting_date = str(meta.get("meeting_date") or "unknown")
        topic = str(meta.get("topic") or path.stem)
        entries.append((meeting_date, topic, path.name, str(meta.get("source_slug", ""))))

    entries.sort(key=lambda row: (row[0], row[2]), reverse=True)
    lines = [
        "# Meeting briefs",
        "",
        "Published extractions for Cursor context. Full artifacts stay in local `output/`.",
        "",
        "| Date | Topic | Brief | Source slug |",
        "| ---- | ----- | ----- | ----------- |",
    ]
    for meeting_date, topic, filename, source_slug in entries:
        safe_topic = topic.replace("|", "\\|")
        lines.append(
            f"| {meeting_date} | {safe_topic} | [{filename}]({filename}) | `{source_slug}` |"
        )
    lines.append("")
    (meetings_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def discover_publishable(output_root: Path) -> list[Path]:
    if not output_root.is_dir():
        return []
    return sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "extraction.json").is_file()
    )


def published_source_slugs(meetings_dir: Path) -> set[str]:
    if not meetings_dir.is_dir():
        return set()
    slugs: set[str] = set()
    for path in meetings_dir.glob("*.md"):
        if path.name == "index.md":
            continue
        meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        slug = meta.get("source_slug")
        if slug:
            slugs.add(str(slug))
    return slugs


def discover_unpublished(output_root: Path, meetings_dir: Path) -> list[Path]:
    published = published_source_slugs(meetings_dir)
    return [path for path in discover_publishable(output_root) if path.name not in published]
