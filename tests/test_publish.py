"""Tests for meeting publish."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow.publish import (
    build_publish_markdown,
    discover_publishable,
    filter_publishable_visual,
    publish_basename,
    publish_context_bundle,
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

    markdown_path = publish_output_dir(output_dir, meetings_dir)

    bundle_dir = meetings_dir / "2026-05-14-revision-avances-mvp"
    assert markdown_path == bundle_dir / "brief.md"
    assert markdown_path.is_file()
    assert (bundle_dir / "extraction.json").is_file()
    assert not (meetings_dir / "2026-05-14-revision-avances-mvp.md").exists()
    assert not (meetings_dir / "2026-05-14-revision-avances-mvp.json").exists()
    assert (bundle_dir / "visual.json").is_file()
    index = (meetings_dir / "index.md").read_text(encoding="utf-8")
    assert "2026-05-14-revision-avances-mvp/brief.md" in index
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
    (meetings_dir / "2026-03-11-older" / "brief.md").parent.mkdir(parents=True)
    (meetings_dir / "2026-03-11-older" / "brief.md").write_text(older, encoding="utf-8")
    (meetings_dir / "2026-05-14-newer" / "brief.md").parent.mkdir(parents=True)
    (meetings_dir / "2026-05-14-newer" / "brief.md").write_text(newer, encoding="utf-8")
    rebuild_meetings_index(meetings_dir)
    index = (meetings_dir / "index.md").read_text(encoding="utf-8")
    assert index.index("2026-05-14-newer/brief.md") < index.index("2026-03-11-older/brief.md")


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


def test_filter_publishable_visual_drops_tile_only() -> None:
    visual = [
        {
            "timestamp": "00:01:00",
            "type": "other",
            "description": "Solo participantes visibles en la videollamada",
        },
        {
            "timestamp": "00:08:37",
            "type": "whiteboard",
            "description": "Pantalla compartida con parámetros del convenio",
        },
        {
            "timestamp": "00:49:58",
            "type": "other",
            "description": (
                "Microsoft Teams: un participante comparte su cámara y avatar; "
                "no hay contenido compartido visible"
            ),
        },
    ]
    filtered = filter_publishable_visual(visual)
    assert len(filtered) == 1
    assert filtered[0]["type"] == "whiteboard"


def test_publish_context_bundle_copies_curated_frames(tmp_path: Path) -> None:
    output_dir = tmp_path / "output" / "meeting-20260514"
    frames_dir = output_dir / "frames"
    chapters_dir = output_dir / "chapters"
    frames_dir.mkdir(parents=True)
    chapters_dir.mkdir()
    frame_path = frames_dir / "frame_0001.jpg"
    frame_path.write_bytes(b"jpg")
    write_json(
        output_dir / "frames.json",
        [{"timestamp": 517.0, "path": str(frame_path), "trigger": "visual_cue"}],
    )
    write_json(
        output_dir / "visual_content.json",
        [
            {
                "timestamp": "00:08:37",
                "type": "whiteboard",
                "description": "Pantalla compartida con parámetros del convenio",
            },
            {
                "timestamp": "00:49:58",
                "type": "other",
                "description": "Solo participantes visibles en cámara sin contenido compartido",
            },
        ],
    )
    (chapters_dir / "chapter_001.txt").write_text("convenio", encoding="utf-8")

    bundle_dir = tmp_path / "docs" / "meetings" / "2026-05-14-meeting"
    stats = publish_context_bundle(output_dir, bundle_dir)

    assert stats["visual_count"] == 1
    assert stats["frame_count"] == 1
    assert (bundle_dir / "frames" / "frame_0001.jpg").is_file()
    assert (bundle_dir / "chapters" / "chapter_001.txt").is_file()
    frames_meta = json.loads((bundle_dir / "frames.json").read_text(encoding="utf-8"))
    assert frames_meta[0]["path"] == "frames/frame_0001.jpg"
    visual = json.loads((bundle_dir / "visual.json").read_text(encoding="utf-8"))
    assert visual[0]["frame"] == "frames/frame_0001.jpg"


def test_publish_context_bundle_removes_stale_frames(tmp_path: Path) -> None:
    output_dir = tmp_path / "output" / "meeting-20260514"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame_path = frames_dir / "frame_0001.jpg"
    frame_path.write_bytes(b"jpg")
    write_json(
        output_dir / "frames.json",
        [{"timestamp": 517.0, "path": str(frame_path), "trigger": "visual_cue"}],
    )
    write_json(
        output_dir / "visual_content.json",
        [
            {
                "timestamp": "00:08:37",
                "type": "whiteboard",
                "description": "Pantalla compartida con parámetros del convenio",
            },
        ],
    )

    bundle_dir = tmp_path / "bundle"
    stale = bundle_dir / "frames" / "frame_9999.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    publish_context_bundle(output_dir, bundle_dir)

    assert (bundle_dir / "frames" / "frame_0001.jpg").is_file()
    assert not stale.is_file()


def test_publish_brief_only_removes_stale_bundle_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "output" / "revision-avances-mvp-20260514_170309-grabación"
    meetings_dir = tmp_path / "docs" / "meetings"
    output_dir.mkdir(parents=True)
    extraction = {
        "meeting_date": "2026-05-14",
        "topic": "Avances MVP",
        "key_decisions": [],
        "action_items": [],
        "blockers_risks": [],
        "status_updates": [],
        "technical_details": [],
        "open_questions": [],
        "next_steps": [],
        "visual_content": [],
    }
    write_json(output_dir / "extraction.json", extraction)
    (output_dir / "summary.md").write_text(render_summary(extraction), encoding="utf-8")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "frame_0001.jpg"
    frame_path.write_bytes(b"jpg")
    write_json(
        output_dir / "frames.json",
        [{"timestamp": 1.0, "path": str(frame_path), "trigger": "scene"}],
    )
    write_json(
        output_dir / "visual_content.json",
        [
            {
                "timestamp": "00:00:01",
                "type": "slide",
                "description": "Pantalla compartida",
            }
        ],
    )

    publish_output_dir(output_dir, meetings_dir)
    bundle_dir = meetings_dir / "2026-05-14-revision-avances-mvp"
    assert (bundle_dir / "extraction.json").is_file()
    assert (bundle_dir / "visual.json").is_file()

    publish_output_dir(output_dir, meetings_dir, include_json=False, include_bundle=False)

    assert (bundle_dir / "brief.md").is_file()
    assert not (bundle_dir / "extraction.json").exists()
    assert not (bundle_dir / "visual.json").exists()
    assert not (bundle_dir / "frames").exists()
