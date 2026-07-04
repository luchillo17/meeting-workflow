"""Real Transcript adapter: ffmpeg audio extraction + faster-whisper."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from workflow.cuda_paths import ensure_cuda_dll_paths
from workflow.utils import run_cmd, write_json


@dataclass
class WhisperTranscript:
    segments: list[dict]
    text: str


class WhisperTranscriber:
    def __init__(
        self,
        config: dict,
        *,
        run_cmd_fn: Callable[..., object] = run_cmd,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        whisper_cfg = config.get("whisper", {})
        self._model_name = whisper_cfg.get("model", "large-v3")
        self._device = whisper_cfg.get("device", "cuda")
        self._compute_type = whisper_cfg.get("compute_type", "float16")
        self._language = whisper_cfg.get("language", "es")
        self._run_cmd = run_cmd_fn
        self._model_factory = model_factory
        self._model: object | None = None

    def _effective_compute_type(self) -> str:
        if self._device == "cpu" and self._compute_type in ("float16", "float32"):
            return "int8"
        return self._compute_type

    def transcribe(self, recording: Path, output_dir: Path) -> WhisperTranscript:
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
        raw_segments, _info = model.transcribe(str(audio_path), language=self._language)
        segments: list[dict] = []
        texts: list[str] = []
        for segment in raw_segments:
            text = segment.text.strip()
            segments.append({"start": segment.start, "end": segment.end, "text": text})
            texts.append(text)
        return segments, " ".join(texts)

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
