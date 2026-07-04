"""Real Visual Capture vision adapter via Ollama."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable
from pathlib import Path
from urllib import request

from workflow.utils import write_json

_VISION_PROMPT = (
    "Describe el contenido visual de esta imagen de una reunión. "
    'Responde solo con JSON: {"type":"whiteboard|diagram|slide|other","description":"..."} '
    "en español."
)


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
        chat_fn: Callable[[str, dict], dict] | None = None,
    ) -> None:
        ollama_cfg = config.get("ollama", {})
        self._base_url = ollama_cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self._model = ollama_cfg.get("vision_model", "qwen2.5vl:7b")
        self._timeout = int(ollama_cfg.get("timeout_seconds", 300))
        self._chat = chat_fn or self._chat_http

    def analyze(self, frame_paths: list[Path], output_dir: Path) -> list[dict]:
        timestamps = self._load_timestamps(output_dir)
        visual: list[dict] = []
        for frame_path in frame_paths:
            if not frame_path.is_file():
                continue
            frame_type, description = self._describe_frame(frame_path)
            seconds = timestamps.get(str(frame_path), timestamps.get(frame_path.as_posix(), 0.0))
            visual.append(
                {
                    "timestamp": format_timestamp(seconds),
                    "type": frame_type,
                    "description": description,
                }
            )
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
                    "content": _VISION_PROMPT,
                    "images": [encoded],
                }
            ],
        }
        response = self._chat(f"{self._base_url}/api/chat", payload)
        content = response.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError(f"Empty vision response for {frame_path.name}")
        return _parse_vision_response(content)

    def _chat_http(self, url: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except OSError as exc:
            raise RuntimeError(
                f"Ollama vision request failed at {self._base_url}. Is Ollama running?"
            ) from exc

    @staticmethod
    def _load_timestamps(output_dir: Path) -> dict[str, float]:
        meta_path = output_dir / "frames.json"
        if not meta_path.is_file():
            return {}
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return {}
        mapping: dict[str, float] = {}
        for entry in data:
            if isinstance(entry, dict) and "path" in entry:
                mapping[str(entry["path"])] = float(entry.get("timestamp", 0.0))
        return mapping
