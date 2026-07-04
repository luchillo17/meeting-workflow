"""Tests for frames CLI command."""

from __future__ import annotations

from pathlib import Path

from workflow.cli import cmd_frames


def _write_test_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "frames:\n  max_frames: 5\noutput:\n  base_dir: output\n",
        encoding="utf-8",
    )


def test_cmd_frames_skips_when_frames_exist(tmp_path: Path, monkeypatch) -> None:
    _write_test_config(tmp_path)
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"video")
    out = tmp_path / "output" / "meeting"
    out.mkdir(parents=True)
    (out / "frames.json").write_text("[]", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RECORDING_PATH", str(recording))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))

    args = type("Args", (), {"file": recording, "output_dir": out, "force": False})()
    assert cmd_frames(args) == 0


def test_cmd_frames_force_calls_extractor(tmp_path: Path, monkeypatch) -> None:
    _write_test_config(tmp_path)
    recording = tmp_path / "meeting.mp4"
    recording.write_bytes(b"video")
    out = tmp_path / "output" / "meeting"
    out.mkdir(parents=True)
    (out / "frames.json").write_text("[]", encoding="utf-8")
    (out / "visual_content.json").write_text("[]", encoding="utf-8")
    calls: list[Path] = []

    class FakeExtractor:
        def __init__(self, _config: dict) -> None:
            pass

        def extract(self, rec: Path, _transcript: object, output_dir: Path) -> list[Path]:
            calls.append(output_dir)
            frame = output_dir / "frames" / "frame_0001.jpg"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"jpeg")
            return [frame]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RECORDING_PATH", str(recording))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr("workflow.cli.FfmpegFrameExtractor", FakeExtractor)

    args = type("Args", (), {"file": recording, "output_dir": out, "force": True})()
    assert cmd_frames(args) == 0
    assert calls == [out]
    assert not (out / "visual_content.json").exists()
