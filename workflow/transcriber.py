"""Real Transcript adapter: ffmpeg audio extraction + faster-whisper."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow.cuda_paths import ensure_cuda_dll_paths
from workflow.transcript_filters import filter_hallucination_segments, join_segment_texts
from workflow.utils import run_cmd, write_json


@dataclass
class WhisperTranscript:
    segments: list[dict]
    text: str


def build_whisper_transcribe_options(config: dict) -> dict[str, Any]:
    """Map config.yaml whisper section to faster-whisper transcribe() kwargs."""
    whisper_cfg = config.get("whisper", {})
    options: dict[str, Any] = {
        "language": whisper_cfg.get("language", "es"),
        "vad_filter": whisper_cfg.get("vad_filter", True),
        "condition_on_previous_text": whisper_cfg.get("condition_on_previous_text", False),
    }

    if vad_parameters := whisper_cfg.get("vad_parameters"):
        options["vad_parameters"] = vad_parameters

    optional_floats = (
        "hallucination_silence_threshold",
        "no_speech_threshold",
        "compression_ratio_threshold",
        "log_prob_threshold",
        "temperature",
    )
    for key in optional_floats:
        if key in whisper_cfg:
            options[key] = whisper_cfg[key]

    if options.get("hallucination_silence_threshold") is not None:
        options["word_timestamps"] = True

    return options


class WhisperTranscriber:
    def __init__(
        self,
        config: dict,
        *,
        run_cmd_fn: Callable[..., object] = run_cmd,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        whisper_cfg = config.get("whisper", {})
        self._config = config
        self._model_name = whisper_cfg.get("model", "large-v3")
        self._device = whisper_cfg.get("device", "cuda")
        self._compute_type = whisper_cfg.get("compute_type", "float16")
        self._language = whisper_cfg.get("language", "es")
        self._filter_hallucinations = whisper_cfg.get("filter_hallucination_phrases", True)
        self._transcribe_options = build_whisper_transcribe_options(config)
        self._run_cmd = run_cmd_fn
        self._model_factory = model_factory
        self._model: object | None = None

    def _effective_compute_type(self) -> str:
        if self._device == "cpu" and self._compute_type in ("float16", "float32"):
            return "int8"
        return self._compute_type

    def transcribe(
        self, recording: Path, output_dir: Path, *, unload_after: bool = True
    ) -> WhisperTranscript:
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / "audio.wav"
        self._extract_audio(recording, audio_path)
        try:
            segments, text = self._run_whisper(audio_path)
            result = WhisperTranscript(segments=segments, text=text)
            write_json(output_dir / "transcript.json", {"segments": result.segments})
            (output_dir / "transcript.txt").write_text(result.text, encoding="utf-8")
            return result
        finally:
            if unload_after:
                self.unload()

    def unload(self) -> None:
        """Release Whisper from VRAM (call once after a batch of transcriptions)."""
        self._unload_model()

    def _extract_audio(self, recording: Path, audio_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(recording),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
        try:
            self._run_cmd(cmd)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Audio extraction failed for {recording.name}. "
                "Ensure ffmpeg is installed and the file is a valid video."
            ) from exc

    def _get_model(self) -> object:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(
                    self._model_name,
                    device=self._device,
                    compute_type=self._effective_compute_type(),
                )
            else:
                ensure_cuda_dll_paths()
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._effective_compute_type(),
                )
        return self._model

    def _run_whisper(self, audio_path: Path) -> tuple[list[dict], str]:
        model = self._get_model()
        raw_segments, _info = model.transcribe(str(audio_path), **self._transcribe_options)
        segments: list[dict] = []
        for segment in raw_segments:
            text = segment.text.strip()
            if not text:
                continue
            segments.append({"start": segment.start, "end": segment.end, "text": text})

        if self._filter_hallucinations:
            segments = filter_hallucination_segments(segments, language=self._language)

        return segments, join_segment_texts(segments)

    def _unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
