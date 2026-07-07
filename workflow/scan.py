"""Discover Meeting Recordings in watch folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from workflow.utils import slugify

RECORDING_SUFFIXES = frozenset({".mp4", ".mkv", ".webm", ".mov", ".m4v"})


@dataclass(frozen=True)
class RecordingScanResult:
    recording: Path
    status: str  # "pending" | "done"
    output_dir: Path


def parse_watch_folders(raw: str | None) -> list[Path]:
    if not raw or not raw.strip():
        return []
    parts = [part.strip().strip('"').strip("'") for part in raw.replace(";", ",").split(",")]
    return [Path(part) for part in parts if part]


def resolve_watch_folders(
    *,
    cli_folders: list[Path] | None = None,
    watch_folders_env: str | None = None,
    onedrive_root: str | None = None,
) -> list[Path]:
    folders: list[Path] = list(cli_folders or [])
    if not folders:
        folders.extend(parse_watch_folders(watch_folders_env))
    if not folders and onedrive_root:
        grabaciones = Path(onedrive_root) / "Grabaciones"
        if grabaciones.is_dir():
            folders.append(grabaciones)
    return folders


def discover_recordings(
    folders: list[Path],
    *,
    recursive: bool = False,
) -> list[Path]:
    recordings: list[Path] = []
    seen: set[str] = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        iterator = folder.rglob("*") if recursive else folder.iterdir()
        for path in sorted(iterator):
            if not path.is_file():
                continue
            if path.suffix.lower() not in RECORDING_SUFFIXES:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            recordings.append(path)
    return sorted(recordings, key=lambda path: path.name.lower())


def scan_recordings(
    folders: list[Path],
    output_root: Path,
    *,
    recursive: bool = False,
) -> list[RecordingScanResult]:
    results: list[RecordingScanResult] = []
    for recording in discover_recordings(folders, recursive=recursive):
        output_dir = output_root / slugify(recording.name)
        status = "done" if (output_dir / "extraction.json").is_file() else "pending"
        results.append(
            RecordingScanResult(recording=recording, status=status, output_dir=output_dir)
        )
    return results


def pending_recordings(
    folders: list[Path],
    output_root: Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    return [
        result.recording
        for result in scan_recordings(folders, output_root, recursive=recursive)
        if result.status == "pending"
    ]
