"""Tests for inbox orchestration."""

from __future__ import annotations

from pathlib import Path

from workflow.inbox import plan_inbox, publish_targets
from workflow.publish import build_publish_markdown, discover_unpublished, published_source_slugs
from workflow.utils import slugify, write_json


def test_published_source_slugs_reads_frontmatter(tmp_path: Path) -> None:
    meetings_dir = tmp_path / "docs" / "meetings"
    meetings_dir.mkdir(parents=True)
    brief = build_publish_markdown(
        {"meeting_date": "2026-05-14", "topic": "MVP"},
        summary_body="# MVP\n",
        source_slug="revision-avances-mvp-20260514",
    )
    bundle_dir = meetings_dir / "2026-05-14-revision-avances-mvp"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "brief.md").write_text(brief, encoding="utf-8")
    assert published_source_slugs(meetings_dir) == {"revision-avances-mvp-20260514"}


def test_discover_unpublished_skips_already_published(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    meetings_dir = tmp_path / "docs" / "meetings"
    published_dir = output_root / "done-meeting-20260101"
    pending_dir = output_root / "new-meeting-20260202"
    published_dir.mkdir(parents=True)
    pending_dir.mkdir(parents=True)
    write_json(published_dir / "extraction.json", {"topic": "done"})
    write_json(pending_dir / "extraction.json", {"topic": "new"})
    brief = build_publish_markdown(
        {"meeting_date": "2026-01-01", "topic": "done"},
        summary_body="# done\n",
        source_slug=published_dir.name,
    )
    meetings_dir.mkdir(parents=True)
    bundle_dir = meetings_dir / "2026-01-01-done"
    bundle_dir.mkdir()
    (bundle_dir / "brief.md").write_text(brief, encoding="utf-8")

    assert discover_unpublished(output_root, meetings_dir) == [pending_dir]


def test_plan_inbox_splits_pending_and_unpublished(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output_root = tmp_path / "output"
    meetings_dir = tmp_path / "docs" / "meetings"
    inbox.mkdir()
    meetings_dir.mkdir(parents=True)

    pending_video = inbox / "pending.mp4"
    done_video = inbox / "done.mp4"
    pending_video.write_bytes(b"1")
    done_video.write_bytes(b"2")

    done_dir = output_root / slugify(done_video.name)
    done_dir.mkdir(parents=True)
    write_json(done_dir / "extraction.json", {"topic": "done"})

    plan = plan_inbox([inbox], output_root, meetings_dir)
    assert plan.pending == [pending_video]
    assert plan.unpublished == [done_dir]
    assert plan.done_count == 1
    assert plan.total_count == 2


def test_publish_targets_writes_briefs(tmp_path: Path) -> None:
    output_dir = tmp_path / "output" / "meeting-20260101"
    meetings_dir = tmp_path / "docs" / "meetings"
    output_dir.mkdir(parents=True)
    write_json(output_dir / "extraction.json", {"meeting_date": "2026-01-01", "topic": "Test"})
    (output_dir / "summary.md").write_text("# Test\n", encoding="utf-8")

    paths = publish_targets([output_dir], meetings_dir)
    assert len(paths) == 1
    assert paths[0].name == "brief.md"
    bundle_dir = meetings_dir / "2026-01-01-meeting"
    assert (bundle_dir / "extraction.json").is_file()
    assert (meetings_dir / "index.md").is_file()
