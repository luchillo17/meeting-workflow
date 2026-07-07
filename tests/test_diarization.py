"""Tests for speaker diarization assignment (no GPU / pyannote)."""

from __future__ import annotations

from pathlib import Path

from workflow.diarization import (
    SpeakerTurn,
    assign_speakers_to_segments,
    diarize_segments,
    overlap_duration,
    resolve_hf_token,
)


def test_overlap_duration() -> None:
    assert overlap_duration(0.0, 5.0, 3.0, 8.0) == 2.0
    assert overlap_duration(0.0, 1.0, 5.0, 6.0) == 0.0


def test_assign_speakers_by_max_overlap() -> None:
    segments = [
        {"start": 0.0, "end": 3.0, "text": "A"},
        {"start": 3.0, "end": 6.0, "text": "B"},
    ]
    turns = [
        SpeakerTurn(0.0, 3.5, "SPEAKER_00"),
        SpeakerTurn(3.0, 7.0, "SPEAKER_01"),
    ]

    labeled = assign_speakers_to_segments(segments, turns)

    assert labeled[0]["speaker"] == "Speaker 1"
    assert labeled[1]["speaker"] == "Speaker 2"


def test_assign_speakers_stable_labels_across_turns() -> None:
    segments = [{"start": 0.0, "end": 2.0, "text": "x"}]
    turns = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
        SpeakerTurn(1.0, 2.0, "SPEAKER_00"),
    ]

    labeled = assign_speakers_to_segments(segments, turns)

    assert labeled[0]["speaker"] == "Speaker 1"


def test_diarize_segments_noop_when_disabled(tmp_path: Path) -> None:
    segments = [{"start": 0.0, "end": 1.0, "text": "hola"}]

    result = diarize_segments(
        tmp_path / "audio.wav",
        segments,
        {"diarization": {"enabled": False}},
    )

    assert result == segments


def test_diarize_segments_uses_injected_runner(tmp_path: Path) -> None:
    segments = [{"start": 0.0, "end": 2.0, "text": "hola"}]
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_diarize(_audio: Path, _config: dict) -> list[SpeakerTurn]:
        return [SpeakerTurn(0.0, 2.0, "SPEAKER_00")]

    result = diarize_segments(
        tmp_path / "audio.wav",
        segments,
        {"diarization": {"enabled": True}},
        diarize_fn=fake_diarize,
        output_dir=output_dir,
    )

    assert result[0]["speaker"] == "Speaker 1"
    assert (output_dir / "diarization.json").is_file()


def test_diarize_segments_skips_on_runner_failure(tmp_path: Path) -> None:
    segments = [{"start": 0.0, "end": 1.0, "text": "hola"}]

    def failing_diarize(_audio: Path, _config: dict) -> list[SpeakerTurn]:
        raise RuntimeError("HF_TOKEN unset")

    result = diarize_segments(
        tmp_path / "audio.wav",
        segments,
        {"diarization": {"enabled": True}},
        diarize_fn=failing_diarize,
    )

    assert result == segments


def test_resolve_hf_token_from_config_env_name(monkeypatch) -> None:
    monkeypatch.setenv("MY_HF", "token-123")
    token = resolve_hf_token({"diarization": {"hf_token_env": "MY_HF"}})
    assert token == "token-123"
