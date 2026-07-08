"""Tests for batch progress settings."""

from __future__ import annotations

from workflow.batch_progress import BatchSettings


def test_batch_settings_defaults() -> None:
    settings = BatchSettings.from_config({})
    assert settings.pipeline_workers == 1
    assert settings.frame_workers >= 1


def test_batch_settings_from_config() -> None:
    settings = BatchSettings.from_config({"batch": {"pipeline_workers": 3, "frame_workers": 6}})
    assert settings.pipeline_workers == 3
    assert settings.frame_workers == 6


def test_batch_settings_clamps_minimum() -> None:
    settings = BatchSettings.from_config({"batch": {"pipeline_workers": 0, "frame_workers": 0}})
    assert settings.pipeline_workers == 1
    assert settings.frame_workers == 1


def test_batch_progress_step_percentages() -> None:
    from io import StringIO

    from rich.console import Console

    from workflow.batch_progress import BatchProgress

    console = Console(file=StringIO(), force_terminal=False, width=120)
    with BatchProgress(console, total_recordings=2, stage_count=2) as progress:
        progress.start_step(1, "Transcript", 1, "meeting-a.mp4")
        progress.complete_step(1, "Transcript", 1, "meeting-a.mp4")
        progress.start_step(2, "Frames", 1, "meeting-a.mp4")

    output = console.file.getvalue()
    assert "Progress  25%" in output or "Progress 25%" in output
    assert "step 1/4" in output
    assert "step 2/4" in output
