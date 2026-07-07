"""Tests for meeting publish."""

from __future__ import annotations

from pathlib import Path

import pytest

from workflow.publish import (
    build_publish_markdown,
    discover_publishable,
    publish_basename,
    publish_output_dir,
    rebuild_meetings_index,
    short_publish_slug,
)
from workflow.summary import render_summary
from workflow.utils import write_json


def test_short_publish_slug_strips_recording_date_suffix() -> None:
    name = "funcionalidad-ipss-trabajo-_-reenfoque-prioridades-20260311_183233-grabación"
    assert short_publish_slug(name) == "funcionalidad-ipss-trabajo-_-reenfoque-prioridades"


def test_publish_basename_uses_meeting_date_and_slug() -> None:
    output_dir = Path("revision-avances-mvp-20260514_170309-grabación")
    extraction = {"meeting_date": "2026-05-14", "topic": "Avances MVP"}
    assert publish_basename(output_dir, extraction) == "2026-05-14-revision-avances-mvp"


def test_build_publish_markdown_includes_frontmatter_and_summary() -> None:
    extraction = {
        "meeting_date": "2026-05-09",
        "topic": "POC review",
        "key_decisions": ["Ship portal responsive fixes"],
        "action_items": [],
        "blockers_risks": [],
        "status_updates": [],
        "technical_details": [],
        "open_questions": [],
        "next_steps": [],
        "visual_content": [],
    }
    body = render_summary(extraction)
    text = build_publish_markdown(
        extraction, summary_body=body, source_slug="revisión-aspectos-relevantes-poc"
    )
    assert text.startswith("---\n")
    assert "meeting_date: '2026-05-09'" in text or "meeting_date: 2026-05-09" in text
    assert "source_slug: revisión-aspectos-relevantes-poc" in text
    assert "# POC review" in text


def test_publish_output_dir_writes_markdown_json_and_index(tmp_path: Path) -> None:
    output_dir = tmp_path / "output" / "revision-avances-mvp-20260514_170309-grabación"
    meetings_dir = tmp_path / "docs" / "meetings"
    output_dir.mkdir(parents=True)
    extraction = {
        "meeting_date": "2026-05-14",
        "topic": "Avances MVP",
        "key_decisions": ["Priorizar onboarding IPS"],
        "action_items": [{"owner": "Ana", "task": "Revisar PR", "deadline": ""}],
        "blockers_risks": [],
        "status_updates": [],
        "technical_details": [],
        "open_questions": [],
        "next_steps": [],
        "visual_content": [],
    }
    write_json(output_dir / "extraction.json", extraction)
    (output_dir / "summary.md").write_text(render_summary(extraction), encoding="utf-8")

    markdown_path = publish_output_dir(output_dir, meetings_dir, include_json=True)

    assert markdown_path.is_file()
    assert (meetings_dir / "2026-05-14-revision-avances-mvp.json").is_file()
    index = (meetings_dir / "index.md").read_text(encoding="utf-8")
    assert "2026-05-14-revision-avances-mvp.md" in index
    assert "Avances MVP" in index


def test_rebuild_meetings_index_sorts_by_date_desc(tmp_path: Path) -> None:
    meetings_dir = tmp_path / "docs" / "meetings"
    meetings_dir.mkdir(parents=True)
    older = build_publish_markdown(
        {"meeting_date": "2026-03-11", "topic": "Older"},
        summary_body="# Older\n",
        source_slug="older",
    )
    newer = build_publish_markdown(
        {"meeting_date": "2026-05-14", "topic": "Newer"},
        summary_body="# Newer\n",
        source_slug="newer",
    )
    (meetings_dir / "2026-03-11-older.md").write_text(older, encoding="utf-8")
    (meetings_dir / "2026-05-14-newer.md").write_text(newer, encoding="utf-8")
    rebuild_meetings_index(meetings_dir)
    index = (meetings_dir / "index.md").read_text(encoding="utf-8")
    assert index.index("2026-05-14-newer.md") < index.index("2026-03-11-older.md")


def test_discover_publishable_finds_extraction_folders(tmp_path: Path) -> None:
    root = tmp_path / "output"
    ready = root / "ready-meeting-20260101"
    empty = root / "empty-meeting"
    ready.mkdir(parents=True)
    empty.mkdir()
    write_json(ready / "extraction.json", {"topic": "x"})
    assert discover_publishable(root) == [ready]


def test_publish_output_dir_requires_extraction(tmp_path: Path) -> None:
    output_dir = tmp_path / "output" / "missing"
    output_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        publish_output_dir(output_dir, tmp_path / "docs" / "meetings")
