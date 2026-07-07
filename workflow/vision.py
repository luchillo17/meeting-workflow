"""Real Visual Capture vision adapter via Ollama."""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from workflow.extraction import is_low_value_visual_description
from workflow.ollama_client import OllamaClient, resolve_ollama_settings
from workflow.output_language import language_display_name, resolve_output_language
from workflow.utils import invalidate_downstream_artifacts, write_json

logger = logging.getLogger(__name__)

_VISION_EMPTY_RETRIES = 2

_VISION_PROMPT = (
    "Describe the visual content of this meeting image. "
    "Ignore meeting notetaker bots and UI chrome (e.g. read.ai, Otter, Fireflies labels). "
    "If the image shows ONLY participant tiles, avatars, or a black waiting screen with no "
    "shared content, respond with: "
    '{{"type":"skip","description":""}}. '
    "Otherwise describe only human-shared content: slides, screen shares, whiteboards, diagrams. "
    'Respond with JSON only: {{"type":"whiteboard|diagram|slide|other|skip","description":"..."}} '
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
        self._model = ollama_cfg.get("vision_model", "qwen3.5:9b")
        self._output_language = resolve_output_language(config)
        self._chat_fn = chat_fn
        self._unload = unload_fn or self._client.unload

    def analyze(
        self, frame_paths: list[Path], output_dir: Path, *, unload_after: bool = True
    ) -> list[dict]:
        invalidate_downstream_artifacts(output_dir)
        timestamps = self._load_timestamps(output_dir)
        visual: list[dict] = []
        try:
            for frame_path in frame_paths:
                if not frame_path.is_file():
                    continue
                frame_type, description = self._describe_frame(frame_path)
                if frame_type == "skip" or not description.strip():
                    continue
                if is_low_value_visual_description(description):
                    continue
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
            if unload_after and self._settings.unload_between_stages:
                self.unload()
        write_json(output_dir / "visual_content.json", visual)
        return visual

    def unload(self) -> None:
        """Release the vision model from VRAM."""
        if self._settings.unload_between_stages:
            self._unload(self._model)

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
        if "qwen3" in self._model.lower():
            payload["messages"][0]["content"] = "/no_think\n" + str(
                payload["messages"][0]["content"]
            )
        for attempt in range(1, _VISION_EMPTY_RETRIES + 1):
            response = (
                self._chat_fn(payload) if self._chat_fn is not None else self._client.chat(payload)
            )
            content = response.get("message", {}).get("content", "")
            if content.strip():
                return _parse_vision_response(content)
            logger.warning(
                "Empty vision response for %s (attempt %d/%d)",
                frame_path.name,
                attempt,
                _VISION_EMPTY_RETRIES,
            )
        logger.warning("Skipping frame after empty vision responses: %s", frame_path.name)
        return "skip", ""

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
