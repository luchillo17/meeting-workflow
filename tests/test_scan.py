"""Tests for folder scan."""

from __future__ import annotations

from pathlib import Path

from workflow.scan import (
    discover_recordings,
    parse_watch_folders,
    pending_recordings,
    resolve_watch_folders,
    scan_recordings,
)
from workflow.utils import write_json


def test_parse_watch_folders_splits_comma_and_semicolon() -> None:
    raw = r"C:\a\Grabaciones, D:\other; E:\third"
    paths = parse_watch_folders(raw)
    assert len(paths) == 3
    assert paths[0] == Path(r"C:\a\Grabaciones")


def test_resolve_watch_folders_prefers_cli_over_env() -> None:
    folders = resolve_watch_folders(
        cli_folders=[Path("/cli")],
        watch_folders_env="/env",
        onedrive_root="/onedrive",
    )
    assert folders == [Path("/cli")]


def test_resolve_watch_folders_uses_onedrive_grabaciones(tmp_path: Path) -> None:
    root = tmp_path / "onedrive"
    grabaciones = root / "Grabaciones"
    grabaciones.mkdir(parents=True)
    folders = resolve_watch_folders(onedrive_root=str(root))
    assert folders == [grabaciones]


def test_discover_recordings_finds_video_files(tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    video = folder / "meeting-a.mp4"
    video.write_bytes(b"video")
    (folder / "notes.txt").write_text("skip", encoding="utf-8")
    assert discover_recordings([folder]) == [video]


def test_discover_recordings_recursive(tmp_path: Path) -> None:
    nested = tmp_path / "inbox" / "2026"
    nested.mkdir(parents=True)
    video = nested / "nested.mp4"
    video.write_bytes(b"video")
    assert discover_recordings([tmp_path / "inbox"], recursive=True) == [video]


def test_scan_recordings_marks_done_when_extraction_exists(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output_root = tmp_path / "output"
    inbox.mkdir()
    video = inbox / "meeting-a.mp4"
    video.write_bytes(b"video")
    from workflow.utils import slugify

    out_dir = output_root / slugify(video.name)
    out_dir.mkdir(parents=True)
    write_json(out_dir / "extraction.json", {"topic": "done"})

    results = scan_recordings([inbox], output_root)
    assert len(results) == 1
    assert results[0].status == "done"


def test_pending_recordings_returns_only_unprocessed(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    output_root = tmp_path / "output"
    inbox.mkdir()
    done = inbox / "done.mp4"
    pending = inbox / "pending.mp4"
    done.write_bytes(b"1")
    pending.write_bytes(b"2")
    from workflow.utils import slugify

    out_dir = output_root / slugify(done.name)
    out_dir.mkdir(parents=True)
    write_json(out_dir / "extraction.json", {"topic": "done"})

    assert pending_recordings([inbox], output_root) == [pending]
