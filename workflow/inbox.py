"""Orchestrate scan → process → publish for watch folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from workflow.publish import discover_unpublished, publish_output_dir
from workflow.scan import scan_recordings


@dataclass(frozen=True)
class InboxPlan:
    pending: list[Path]
    unpublished: list[Path]
    done_count: int
    total_count: int


def plan_inbox(
    folders: list[Path],
    output_root: Path,
    meetings_dir: Path,
    *,
    recursive: bool = False,
) -> InboxPlan:
    results = scan_recordings(folders, output_root, recursive=recursive)
    pending = [result.recording for result in results if result.status == "pending"]
    unpublished = discover_unpublished(output_root, meetings_dir)
    return InboxPlan(
        pending=pending,
        unpublished=unpublished,
        done_count=len(results) - len(pending),
        total_count=len(results),
    )


def publish_targets(
    targets: list[Path],
    meetings_dir: Path,
    *,
    include_json: bool = False,
) -> list[Path]:
    published: list[Path] = []
    for output_dir in targets:
        published.append(publish_output_dir(output_dir, meetings_dir, include_json=include_json))
    return published
