"""Tests for transcript chapter splitting."""

from __future__ import annotations

from workflow.transcript_chapters import (
    parse_timestamp,
    split_segments_into_chapters,
    visual_for_time_range,
    write_chapter_bundle,
)


def test_split_segments_into_chapters_preserves_speaker_labels() -> None:
    segments = [
        {"start": 0.0, "end": 5.0, "text": "propuesta", "speaker": "Speaker 1"},
        {"start": 5.0, "end": 10.0, "text": "de acuerdo", "speaker": "Speaker 2"},
    ]

    chapters = split_segments_into_chapters(segments, target_chars=500, min_chars=1)

    assert "[Speaker 1]" in chapters[0].text
    assert "[Speaker 2]" in chapters[0].text


def test_split_segments_into_chapters_respects_target_size() -> None:
    segments = [{"start": i * 10.0, "end": i * 10.0 + 9.0, "text": "x" * 2_000} for i in range(6)]

    chapters = split_segments_into_chapters(segments, target_chars=5_000, min_chars=500)

    assert len(chapters) >= 2
    assert all(len(chapter.text) <= 5_500 for chapter in chapters)
    assert chapters[0].start_seconds == 0.0
    assert chapters[-1].end_seconds == segments[-1]["end"]


def test_write_chapter_bundle_persists_files(tmp_path) -> None:
    chapters = split_segments_into_chapters(
        [
            {"start": 0.0, "end": 5.0, "text": "inicio"},
            {"start": 5.0, "end": 10.0, "text": "cierre"},
        ],
        target_chars=10,
        min_chars=1,
    )

    written = write_chapter_bundle(tmp_path, chapters)

    assert (tmp_path / "chapters.json").is_file()
    assert written[0].path.startswith("chapters/")
    assert (tmp_path / written[0].path).read_text(encoding="utf-8")


def test_visual_for_time_range_filters_by_timestamp() -> None:
    visual = [
        {"timestamp": "00:00:10", "type": "slide", "description": "a"},
        {"timestamp": "00:10:00", "type": "slide", "description": "b"},
    ]

    matched = visual_for_time_range(visual, 0.0, 120.0, padding_seconds=0.0)

    assert len(matched) == 1
    assert matched[0]["description"] == "a"


def test_parse_timestamp() -> None:
    assert parse_timestamp("01:02:03") == 3723.0
