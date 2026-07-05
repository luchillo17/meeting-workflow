"""Real Visual Capture vision adapter via Ollama."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable
from pathlib import Path

from workflow.ollama_client import OllamaClient, resolve_ollama_settings
from workflow.output_language import language_display_name, resolve_output_language
from workflow.utils import invalidate_downstream_artifacts, write_json

_VISION_PROMPT = (
    "Describe the visual content of this meeting image. "
    "Ignore meeting notetaker bots and UI chrome (e.g. read.ai, Otter, Fireflies labels) — "
    "describe only human-shared content when present. "
    'Respond with JSON only: {{"type":"whiteboard|diagram|slide|other","description":"..."}} '
    "Write the description in {output_language}."
)


def build_vision_prompt(*, output_language: str = "es") -> str:
    return _VISION_PROMPT.format(output_language=language_display_name(output_language))


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_vision_response(content: str) -> tuple[str, str]:
    content = content.strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return (
                str(data.get("type", "other")),
                str(data.get("description", content)),
            )
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return (
                        str(data.get("type", "other")),
                        str(data.get("description", content)),
                    )
            except json.JSONDecodeError:
                pass
    return "other", content


class OllamaVisionAnalyzer:
    def __init__(
        self,
        config: dict,
        *,
        chat_fn: Callable[[dict], dict] | None = None,
        unload_fn: Callable[[str], None] | None = None,
    ) -> None:
        ollama_cfg = config.get("ollama", {})
        self._settings = resolve_ollama_settings(config)
        self._client = OllamaClient(self._settings)
        self._model = ollama_cfg.get("vision_model", "qwen2.5vl:7b")
        self._output_language = resolve_output_language(config)
        self._chat_fn = chat_fn
        self._unload = unload_fn or self._client.unload

    def analyze(self, frame_paths: list[Path], output_dir: Path) -> list[dict]:
        invalidate_downstream_artifacts(output_dir)
        timestamps = self._load_timestamps(output_dir)
        visual: list[dict] = []
        try:
            for frame_path in frame_paths:
                if not frame_path.is_file():
                    continue
                frame_type, description = self._describe_frame(frame_path)
                seconds = timestamps.get(
                    str(frame_path), timestamps.get(frame_path.as_posix(), 0.0)
                )
                visual.append(
                    {
                        "timestamp": format_timestamp(seconds),
                        "type": frame_type,
                        "description": description,
                    }
                )
        finally:
            if self._settings.unload_between_stages:
                self._unload(self._model)
        write_json(output_dir / "visual_content.json", visual)
        return visual

    def _describe_frame(self, frame_path: Path) -> tuple[str, str]:
        encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": build_vision_prompt(output_language=self._output_language),
                    "images": [encoded],
                }
            ],
        }
        response = (
            self._chat_fn(payload) if self._chat_fn is not None else self._client.chat(payload)
        )
        content = response.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError(f"Empty vision response for {frame_path.name}")
        return _parse_vision_response(content)

    @staticmethod
    def _load_timestamps(output_dir: Path) -> dict[str, float]:
        meta_path = output_dir / "frames.json"
        if not meta_path.is_file():
            return {}
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, list):
            return {}
        mapping: dict[str, float] = {}
        for entry in data:
            if isinstance(entry, dict) and "path" in entry:
                mapping[str(entry["path"])] = float(entry.get("timestamp", 0.0))
        return mapping
