"""Publish extraction briefs to docs/meetings for Cursor context."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from workflow.extraction import has_shared_content_signal, is_low_value_visual_description
from workflow.summary import render_summary
from workflow.transcript_chapters import format_timestamp, parse_timestamp
from workflow.utils import parse_meeting_date, utc_now_iso, write_json

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DATE_SUFFIX_RE = re.compile(r"-20\d{6}.*$")
_FRAME_MATCH_TOLERANCE_SECONDS = 2.0
BUNDLE_BRIEF_NAME = "brief.md"
BUNDLE_EXTRACTION_NAME = "extraction.json"
_NO_SHARED_CONTENT_RE = re.compile(
    r"(?i)(no hay contenido compartido|sin contenido compartido|"
    r"no hay (?:presentación|diapositivas) visible|"
    r"comparte su cámara y avatar)"
)
_TILE_LAYOUT_RE = re.compile(
    r"(?i)(?:pantalla de )?videollamada con la interfaz|"
    r"participantes visibles en sus propias ventanas|"
    r"avatares de \w+ participantes"
)


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


def load_visual_content(output_dir: Path) -> list[dict[str, Any]]:
    visual_path = output_dir / "visual_content.json"
    if not visual_path.is_file():
        return []
    try:
        data = json.loads(visual_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def is_publishable_visual_entry(entry: dict[str, Any]) -> bool:
    """Return True when a visual capture is worth publishing for agent context."""
    frame_type = str(entry.get("type", "other"))
    if frame_type.strip().lower() == "skip":
        return False
    description = str(entry.get("description", ""))
    if is_low_value_visual_description(description, frame_type=frame_type):
        return False
    text = description.strip()
    if _NO_SHARED_CONTENT_RE.search(text):
        return False
    if _TILE_LAYOUT_RE.search(text) and not has_shared_content_signal(text):
        return False
    return True


def filter_publishable_visual(visual_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop Teams tile-only / bot chrome captures; keep slides, whiteboards, screen shares."""
    return [entry for entry in visual_content if is_publishable_visual_entry(entry)]


def load_frame_catalog(output_dir: Path) -> list[dict[str, Any]]:
    meta_path = output_dir / "frames.json"
    if not meta_path.is_file():
        return []
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    catalog: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        raw_path = Path(str(entry["path"]))
        frame_path = raw_path if raw_path.is_file() else output_dir / "frames" / raw_path.name
        if not frame_path.is_file():
            continue
        seconds = float(entry.get("timestamp", 0.0))
        catalog.append(
            {
                "timestamp": seconds,
                "timestamp_label": format_timestamp(seconds),
                "path": frame_path,
                "name": frame_path.name,
            }
        )
    return catalog


def resolve_frame_for_visual(
    entry: dict[str, Any],
    catalog: list[dict[str, Any]],
    *,
    tolerance_seconds: float = _FRAME_MATCH_TOLERANCE_SECONDS,
) -> Path | None:
    """Map a visual_content entry to its source frame file."""
    label = str(entry.get("timestamp", "")).strip()
    if label:
        for item in catalog:
            if item["timestamp_label"] == label:
                return item["path"]
        target = parse_timestamp(label)
        best: dict[str, Any] | None = None
        best_delta = tolerance_seconds
        for item in catalog:
            delta = abs(float(item["timestamp"]) - target)
            if delta <= best_delta:
                best_delta = delta
                best = item
        if best is not None:
            return best["path"]
    return None


def publish_context_bundle(
    output_dir: Path,
    bundle_dir: Path,
    visual_content: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write agent-context artifacts (visual, chapters, curated frames) under ``bundle_dir``."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    visual = filter_publishable_visual(visual_content or load_visual_content(output_dir))
    catalog = load_frame_catalog(output_dir)
    frames_dir = bundle_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    published_visual: list[dict[str, Any]] = []
    published_frame_names: set[str] = set()
    for entry in visual:
        frame_path = resolve_frame_for_visual(entry, catalog)
        published_entry = {
            "timestamp": entry.get("timestamp"),
            "type": entry.get("type"),
            "description": entry.get("description"),
        }
        if frame_path is not None:
            dest = frames_dir / frame_path.name
            shutil.copy2(frame_path, dest)
            published_entry["frame"] = f"frames/{frame_path.name}"
            published_frame_names.add(frame_path.name)
        published_visual.append(published_entry)

    for stale in frames_dir.iterdir():
        if stale.is_file() and stale.name not in published_frame_names:
            stale.unlink(missing_ok=True)

    catalog_by_name = {item["name"]: item for item in catalog}
    published_frames_meta: list[dict[str, Any]] = []
    for name in sorted(published_frame_names):
        item = catalog_by_name.get(name)
        if item is None:
            continue
        published_frames_meta.append(
            {
                "timestamp": item["timestamp"],
                "path": f"frames/{name}",
            }
        )
    write_json(bundle_dir / "frames.json", published_frames_meta)

    write_json(bundle_dir / "visual.json", published_visual)

    chapters_src = output_dir / "chapters"
    chapters_dest = bundle_dir / "chapters"
    if chapters_src.is_dir():
        chapters_dest.mkdir(parents=True, exist_ok=True)
        source_chapter_names = {chapter.name for chapter in chapters_src.glob("*.txt")}
        for chapter in chapters_src.glob("*.txt"):
            shutil.copy2(chapter, chapters_dest / chapter.name)
        for stale in chapters_dest.glob("*.txt"):
            if stale.name not in source_chapter_names:
                stale.unlink(missing_ok=True)
    elif chapters_dest.is_dir():
        for stale in chapters_dest.glob("*.txt"):
            stale.unlink(missing_ok=True)

    return {
        "visual_count": len(published_visual),
        "frame_count": sum(1 for entry in published_visual if entry.get("frame")),
        "chapter_count": len(list((bundle_dir / "chapters").glob("*.txt")))
        if (bundle_dir / "chapters").is_dir()
        else 0,
    }


def _prune_bundle_artifacts(
    bundle_dir: Path,
    *,
    include_json: bool,
    include_bundle: bool,
) -> None:
    """Remove bundle files omitted by the current publish mode."""
    if not include_json:
        (bundle_dir / BUNDLE_EXTRACTION_NAME).unlink(missing_ok=True)
    if not include_bundle:
        for name in ("visual.json", "frames.json"):
            (bundle_dir / name).unlink(missing_ok=True)
        for subdir in ("frames", "chapters"):
            path = bundle_dir / subdir
            if path.is_dir():
                shutil.rmtree(path)


def _remove_legacy_flat_publish_files(meetings_dir: Path, basename: str) -> None:
    """Drop pre-bundle flat ``{basename}.md`` / ``.json`` files at ``meetings_dir`` root."""
    for suffix in (".md", ".json"):
        legacy = meetings_dir / f"{basename}{suffix}"
        if legacy.is_file():
            legacy.unlink(missing_ok=True)


def iter_published_briefs(meetings_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return ``(brief_path, frontmatter)`` for each published meeting bundle."""
    if not meetings_dir.is_dir():
        return []
    briefs: list[tuple[Path, dict[str, Any]]] = []
    bundle_names: set[str] = set()
    for bundle_dir in sorted(meetings_dir.iterdir()):
        if not bundle_dir.is_dir():
            continue
        brief_path = bundle_dir / BUNDLE_BRIEF_NAME
        if brief_path.is_file():
            meta, _ = parse_frontmatter(brief_path.read_text(encoding="utf-8"))
            briefs.append((brief_path, meta))
            bundle_names.add(bundle_dir.name)
    for path in sorted(meetings_dir.glob("*.md")):
        if path.name == "index.md" or path.stem in bundle_names:
            continue
        meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        briefs.append((path, meta))
    return briefs


def publish_output_dir(
    output_dir: Path,
    meetings_dir: Path,
    *,
    include_json: bool = True,
    include_bundle: bool = True,
) -> Path:
    """Write one meeting bundle under ``meetings_dir/{date}-{slug}/``; refresh index.md."""
    extraction, summary_body = load_extraction_bundle(output_dir)
    meetings_dir.mkdir(parents=True, exist_ok=True)
    basename = publish_basename(output_dir, extraction)
    bundle_dir = meetings_dir / basename
    bundle_dir.mkdir(parents=True, exist_ok=True)

    brief_path = bundle_dir / BUNDLE_BRIEF_NAME
    brief_path.write_text(
        build_publish_markdown(
            extraction,
            summary_body=summary_body,
            source_slug=output_dir.name,
        ),
        encoding="utf-8",
    )
    if include_json:
        write_json(bundle_dir / BUNDLE_EXTRACTION_NAME, extraction)
    if include_bundle:
        publish_context_bundle(output_dir, bundle_dir)
    _prune_bundle_artifacts(
        bundle_dir,
        include_json=include_json,
        include_bundle=include_bundle,
    )

    _remove_legacy_flat_publish_files(meetings_dir, basename)
    rebuild_meetings_index(meetings_dir)
    return brief_path


def rebuild_meetings_index(meetings_dir: Path) -> None:
    entries: list[tuple[str, str, str, str]] = []
    seen_slugs: set[str] = set()
    for brief_path, meta in iter_published_briefs(meetings_dir):
        source_slug = str(meta.get("source_slug", ""))
        if source_slug and source_slug in seen_slugs:
            continue
        if source_slug:
            seen_slugs.add(source_slug)
        meeting_date = str(meta.get("meeting_date") or "unknown")
        topic = str(meta.get("topic") or brief_path.parent.name)
        rel_link = brief_path.relative_to(meetings_dir).as_posix()
        entries.append((meeting_date, topic, rel_link, source_slug))

    entries.sort(key=lambda row: (row[0], row[2]), reverse=True)
    lines = [
        "# Meeting briefs",
        "",
        "Published extractions for Cursor context. "
        "Full pipeline artifacts stay in local `output/`.",
        "Each meeting is a `{date}-{slug}/` folder with `brief.md`, `extraction.json`,",
        "`visual.json`, `chapters/`, and curated `frames/`.",
        "",
        "| Date | Topic | Brief | Source slug |",
        "| ---- | ----- | ----- | ----------- |",
    ]
    for meeting_date, topic, rel_link, source_slug in entries:
        safe_topic = topic.replace("|", "\\|")
        label = Path(rel_link).parent.name
        lines.append(f"| {meeting_date} | {safe_topic} | [{label}]({rel_link}) | `{source_slug}` |")
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
    slugs: set[str] = set()
    for _path, meta in iter_published_briefs(meetings_dir):
        slug = meta.get("source_slug")
        if slug:
            slugs.add(str(slug))
    return slugs


def discover_unpublished(output_root: Path, meetings_dir: Path) -> list[Path]:
    published = published_source_slugs(meetings_dir)
    return [path for path in discover_publishable(output_root) if path.name not in published]
