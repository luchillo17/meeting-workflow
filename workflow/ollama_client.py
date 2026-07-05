"""Shared Ollama HTTP client with sequential model lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    timeout_seconds: int
    keep_alive: str | int
    unload_between_stages: bool


def resolve_ollama_settings(config: dict) -> OllamaSettings:
    ollama_cfg = config.get("ollama", {})
    return OllamaSettings(
        base_url=ollama_cfg.get("base_url", "http://localhost:11434").rstrip("/"),
        timeout_seconds=int(ollama_cfg.get("timeout_seconds", 300)),
        keep_alive=ollama_cfg.get("keep_alive", "10m"),
        unload_between_stages=bool(ollama_cfg.get("unload_between_stages", True)),
    )


class OllamaClient:
    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    @property
    def timeout_seconds(self) -> int:
        return self._settings.timeout_seconds

    def chat(
        self, payload: dict[str, Any], *, keep_alive: str | int | None = None
    ) -> dict[str, Any]:
        body = dict(payload)
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        elif "keep_alive" not in body:
            body["keep_alive"] = self._settings.keep_alive
        return self._post("/api/chat", body)

    def unload(self, model: str) -> None:
        """Release VRAM for a model before loading the next stage."""
        self._post(
            "/api/generate",
            {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._settings.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._settings.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except OSError as exc:
            raise RuntimeError(
                f"Ollama request failed at {self._settings.base_url}. Is Ollama running?"
            ) from exc
