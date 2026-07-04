"""Tests for frame extraction helpers and FfmpegFrameExtractor."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from workflow.frames import (
    FfmpegFrameExtractor,
    average_hash,
    dedupe_near_duplicate_frames,
    gap_fill_interval_timestamps,
    hamming_distance,
    layout_hash,
    merge_capture_timestamps,
    visual_cue_timestamps,
)


def test_load_transcript_for_frames_reads_segments(tmp_path: Path) -> None:
    from workflow.frames import load_transcript_for_frames

    out = tmp_path / "extraction"
    out.mkdir()
    (out / "transcript.json").write_text(
        '{"segments": [{"start": 5.0, "end": 8.0, "text": "veamos el tablero"}]}',
        encoding="utf-8",
    )

    transcript = load_transcript_for_frames(out)

    assert len(transcript.segments) == 1
    assert "tablero" in transcript.text


def test_load_transcript_for_frames_missing_returns_empty(tmp_path: Path) -> None:
    from workflow.frames import load_transcript_for_frames

    transcript = load_transcript_for_frames(tmp_path / "missing")

    assert transcript.segments == []
    assert transcript.text == ""


def test_visual_cue_timestamps_finds_matching_segments() -> None:
    segments = [
        {"start": 0.0, "end": 5.0, "text": "Hola a todos."},
        {"start": 10.0, "end": 15.0, "text": "Miremos el tablero en Miro."},
        {"start": 20.0, "end": 25.0, "text": "Siguiente tema."},
    ]
    cues = ["tablero", "miro"]

    times = visual_cue_timestamps(segments, cues)

    assert times == [10.0]


def test_merge_capture_timestamps_respects_max_and_priority() -> None:
    merged = merge_capture_timestamps(
        duration=120.0,
        scene_times=[0.0, 10.0, 10.2],
        interval_times=[0.0, 25.0, 50.0],
        visual_cue_times=[10.1],
        max_frames=3,
        min_spacing_seconds=0,
    )

    assert len(merged) == 3
    assert merged[0] == (0.0, "scene")
    assert merged[1][0] == pytest.approx(10.0)
    assert merged[1][1] == "visual_cue"
    assert merged[2] == (25.0, "interval")


def test_merge_capture_timestamps_keeps_all_visual_cues_when_over_budget() -> None:
    merged = merge_capture_timestamps(
        duration=600.0,
        scene_times=[float(i) for i in range(0, 60, 2)],
        interval_times=[float(i) for i in range(0, 600, 25)],
        visual_cue_times=[100.0, 200.0, 300.0],
        max_frames=5,
        min_spacing_seconds=0,
        cluster_seconds=0,
    )

    cue_times = [ts for ts, trigger in merged if trigger == "visual_cue"]
    assert cue_times == [100.0, 200.0, 300.0]


def test_merge_capture_timestamps_keeps_scene_even_when_close_to_interval() -> None:
    merged = merge_capture_timestamps(
        duration=120.0,
        scene_times=[2.0],
        interval_times=[0.0, 25.0, 50.0],
        visual_cue_times=[],
        max_frames=10,
        min_spacing_seconds=30.0,
    )

    triggers = dict(merged)
    assert triggers[2.0] == "scene"
    assert 25.0 not in triggers  # interval too soon after scene
    assert 50.0 in triggers


def test_merge_capture_timestamps_enforces_min_spacing() -> None:
    merged = merge_capture_timestamps(
        duration=300.0,
        scene_times=[2.0],
        interval_times=[0.0, 25.0, 50.0, 100.0],
        visual_cue_times=[],
        max_frames=10,
        min_spacing_seconds=30.0,
    )

    timestamps = [ts for ts, _ in merged]
    assert 2.0 in timestamps
    assert 25.0 not in timestamps  # only 23s after scene; static-screen oversample
    assert 50.0 in timestamps
    assert all(b - a >= 30.0 for a, b in zip(timestamps, timestamps[1:], strict=False))


def test_average_hash_detects_identical_images(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(path, format="JPEG")
    path2 = tmp_path / "frame2.jpg"
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(path2, format="JPEG", quality=50)

    assert hamming_distance(average_hash(path), average_hash(path2)) <= 2


def test_dedupe_near_duplicate_frames_keeps_first() -> None:
    timestamps = [0.0, 1.0, 30.0]
    hashes = {0: 0b1111, 1: 0b1110, 2: 0b1110}

    kept = dedupe_near_duplicate_frames(
        timestamps,
        hashes,
        threshold=1,
        dedup_window_seconds=20.0,
        global_dedup=False,
    )

    assert kept == [0, 2]  # index 1 dropped (similar + within window); index 2 kept (30s later)


def test_dedupe_global_collapses_same_layout_far_apart() -> None:
    timestamps = [2.0, 150.0, 400.0]
    hashes = {0: 0b1111, 1: 0b1110, 2: 0b1110}

    kept = dedupe_near_duplicate_frames(
        timestamps,
        hashes,
        threshold=5,
        global_dedup=True,
    )

    assert kept == [0]


def test_dedupe_window_keeps_similar_frames_far_apart() -> None:
    timestamps = [2.0, 150.0]
    hashes = {0: 0b1111, 1: 0b1110}

    kept = dedupe_near_duplicate_frames(
        timestamps,
        hashes,
        threshold=5,
        dedup_window_seconds=20.0,
        global_dedup=False,
    )

    assert kept == [0, 1]


def test_dedupe_keeps_protected_visual_cue_even_when_duplicate() -> None:
    timestamps = [0.0, 10.0]
    hashes = {0: 0b1111, 1: 0b1111}
    triggers = ["interval", "visual_cue"]

    kept = dedupe_near_duplicate_frames(
        timestamps,
        hashes,
        threshold=5,
        global_dedup=True,
        triggers=triggers,
    )

    assert kept == [0, 1]


def test_gap_fill_interval_timestamps_only_in_long_gaps() -> None:
    times = gap_fill_interval_timestamps(
        [2.0, 130.0],
        duration=300.0,
        gap_seconds=120.0,
        interval_seconds=60.0,
    )

    assert 62.0 in times
    assert 190.0 in times
    assert 250.0 in times
    assert 0.0 not in times
    assert 25.0 not in times


def test_layout_hash_ignores_small_highlight_change(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    base = tmp_path / "base.jpg"
    highlight = tmp_path / "highlight.jpg"
    different = tmp_path / "different.jpg"

    img = Image.new("RGB", (320, 180), color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 150, 150), fill=(80, 80, 80))
    draw.rectangle((170, 20, 300, 150), fill=(90, 90, 90))
    img.save(base, format="JPEG")

    highlighted = img.copy()
    pixels = highlighted.load()
    for y in range(16, 154):
        for x in range(16, 154):
            if x <= 20 or x >= 150 or y <= 20 or y >= 150:
                red, green, blue = pixels[x, y]
                pixels[x, y] = (min(255, red + 50), min(255, green + 40), blue)
    highlighted.save(highlight, format="JPEG")

    other = Image.new("RGB", (320, 180), color=(10, 10, 10))
    odraw = ImageDraw.Draw(other)
    odraw.rectangle((40, 40, 280, 140), fill=(200, 200, 200))
    other.save(different, format="JPEG")

    assert hamming_distance(layout_hash(base), layout_hash(highlight)) <= 8
    assert hamming_distance(layout_hash(base), layout_hash(different)) > 8


def _fake_duration(_recording: Path) -> float:
    return 60.0


def _fake_scene_times(_recording: Path, _threshold: float) -> list[float]:
    return [5.0, 30.0]


def test_extract_writes_frames_and_metadata(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"
    frames_written: list[Path] = []

    def fake_run_cmd(
        cmd: list[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        if "-frames:v" in cmd:
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"jpeg-bytes-" + str(len(frames_written)).encode())
            frames_written.append(out)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    extractor = FfmpegFrameExtractor(
        {
            "frames": {
                "scene_threshold": 0.35,
                "interval_seconds": 25,
                "max_frames": 5,
                "jpeg_quality": 3,
                "max_width": 640,
            },
            "visual_cues": ["tablero"],
        },
        run_cmd_fn=fake_run_cmd,
        probe_duration_fn=_fake_duration,
        scene_times_fn=_fake_scene_times,
        layout_hash_fn=lambda _p: 1,
    )
    transcript = type(
        "T",
        (),
        {"segments": [{"start": 12.0, "end": 14.0, "text": "el tablero"}], "text": "tablero"},
    )()

    paths = extractor.extract(recording, transcript, output_dir)

    assert len(paths) <= 5
    assert (output_dir / "frames.json").is_file()
    meta = json.loads((output_dir / "frames.json").read_text(encoding="utf-8"))
    assert meta
    assert all(entry["trigger"] in ("scene", "interval", "visual_cue") for entry in meta)
    assert frames_written


def test_extract_ffmpeg_scale_and_quality_flags(tmp_path: Path) -> None:
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"fake-video")
    captured: list[list[str]] = []

    def fake_run_cmd(
        cmd: list[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        if "-frames:v" in cmd:
            Path(cmd[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    extractor = FfmpegFrameExtractor(
        {"frames": {"max_frames": 2, "interval_seconds": 30, "jpeg_quality": 3, "max_width": 1280}},
        run_cmd_fn=fake_run_cmd,
        probe_duration_fn=lambda _r: 30.0,
        scene_times_fn=lambda _r, _t: [],
        layout_hash_fn=lambda _p: 1,
    )
    transcript = type("T", (), {"segments": [], "text": ""})()

    extractor.extract(recording, transcript, tmp_path / "out")

    frame_cmds = [c for c in captured if "-frames:v" in c]
    assert frame_cmds
    vf = frame_cmds[0][frame_cmds[0].index("-vf") + 1]
    assert "scale=1280:-2" in vf
    assert frame_cmds[0][frame_cmds[0].index("-q:v") + 1] == "3"
