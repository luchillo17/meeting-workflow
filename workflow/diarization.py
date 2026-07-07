"""Speaker diarization and assignment to Whisper segments."""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow.speaker_names import infer_speaker_names
from workflow.utils import write_json

logger = logging.getLogger(__name__)

DiarizeFn = Callable[[Path, dict[str, Any]], list["SpeakerTurn"]]


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_speakers_to_segments(
    segments: list[dict],
    turns: list[SpeakerTurn],
) -> list[dict]:
    """Attach a display speaker label to each segment by maximum time overlap."""
    if not segments or not turns:
        return [dict(segment) for segment in segments]

    label_map: dict[str, str] = {}
    labeled: list[dict] = []

    for segment in segments:
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        best_raw: str | None = None
        best_overlap = 0.0
        for turn in turns:
            overlap = overlap_duration(seg_start, seg_end, turn.start, turn.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_raw = turn.speaker

        updated = dict(segment)
        if best_raw and best_overlap > 0:
            if best_raw not in label_map:
                label_map[best_raw] = f"Speaker {len(label_map) + 1}"
            updated["speaker"] = label_map[best_raw]
        labeled.append(updated)

    return labeled


def resolve_hf_token(config: dict[str, Any]) -> str:
    diar_cfg = config.get("diarization", {})
    env_name = str(diar_cfg.get("hf_token_env", "HF_TOKEN"))
    for key in (env_name, "HUGGINGFACE_TOKEN", "HF_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def run_pyannote_diarization(audio_path: Path, config: dict[str, Any]) -> list[SpeakerTurn]:
    """Run pyannote speaker-diarization on mono 16 kHz audio."""
    diar_cfg = config.get("diarization", {})
    model_id = str(diar_cfg.get("model", "pyannote/speaker-diarization-3.1"))
    token = resolve_hf_token(config)
    if not token:
        raise RuntimeError(
            "Diarization requires a Hugging Face token. "
            "Accept the model terms at huggingface.co/pyannote/speaker-diarization-3.1 "
            "and set HF_TOKEN in .env"
        )

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Diarization requires pyannote.audio and torch. Run: uv sync --extra diarization"
        ) from exc

    device_name = str(diar_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
        logger.warning("CUDA unavailable for diarization; falling back to CPU")

    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
    pipeline.to(torch.device(device_name))

    kwargs: dict[str, Any] = {}
    if (min_speakers := diar_cfg.get("min_speakers")) is not None:
        kwargs["min_speakers"] = int(min_speakers)
    if (max_speakers := diar_cfg.get("max_speakers")) is not None:
        kwargs["max_speakers"] = int(max_speakers)

    diarization = pipeline(str(audio_path), **kwargs)
    turns: list[SpeakerTurn] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        )

    del pipeline
    gc.collect()
    if device_name == "cuda":
        torch.cuda.empty_cache()

    return turns


def write_diarization_artifact(
    output_dir: Path,
    turns: list[SpeakerTurn],
    *,
    speaker_map: dict[str, str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "turns": [
            {"start": turn.start, "end": turn.end, "speaker": turn.speaker} for turn in turns
        ],
    }
    if speaker_map:
        payload["speaker_map"] = speaker_map
    write_json(output_dir / "diarization.json", payload)


def diarize_segments(
    audio_path: Path,
    segments: list[dict],
    config: dict[str, Any],
    *,
    diarize_fn: DiarizeFn | None = None,
    output_dir: Path | None = None,
) -> list[dict]:
    """Run diarization and label segments; no-op when disabled or empty turns."""
    diar_cfg = config.get("diarization", {})
    if not bool(diar_cfg.get("enabled", False)):
        return segments

    runner = diarize_fn or run_pyannote_diarization
    try:
        turns = runner(audio_path, config)
    except RuntimeError as exc:
        logger.warning("Diarization skipped for %s: %s", audio_path.name, exc)
        return segments
    if not turns:
        logger.warning("Diarization returned no speaker turns for %s", audio_path.name)
        return segments

    labeled = assign_speakers_to_segments(segments, turns)
    labeled, speaker_map = infer_speaker_names(labeled, config)

    if output_dir is not None:
        write_diarization_artifact(output_dir, turns, speaker_map=speaker_map or None)

    return labeled
