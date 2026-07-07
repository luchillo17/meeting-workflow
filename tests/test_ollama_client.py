"""Tests for OllamaClient."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from workflow.ollama_client import OllamaClient, OllamaSettings, resolve_ollama_settings


def test_resolve_ollama_settings_defaults() -> None:
    settings = resolve_ollama_settings({})
    assert settings.unload_between_stages is True
    assert settings.keep_alive == "10m"


def test_chat_adds_keep_alive_when_missing() -> None:
    client = OllamaClient(OllamaSettings("http://localhost:11434", 30, "5m", True))
    captured: dict = {}

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        response = MagicMock()
        response.read.return_value = json.dumps({"message": {"content": "ok"}}).encode("utf-8")
        response.__enter__ = lambda *_a: response
        response.__exit__ = lambda *_a: None
        return response

    with patch("workflow.ollama_client.request.urlopen", fake_urlopen):
        client.chat({"model": "qwen2.5:14b", "messages": []})

    assert captured["body"]["keep_alive"] == "5m"
    assert captured["body"]["think"] is False


def test_chat_preserves_explicit_think_flag() -> None:
    client = OllamaClient(OllamaSettings("http://localhost:11434", 30, "5m", True))
    captured: dict = {}

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        response = MagicMock()
        response.read.return_value = json.dumps({"message": {"content": "ok"}}).encode("utf-8")
        response.__enter__ = lambda *_a: response
        response.__exit__ = lambda *_a: None
        return response

    with patch("workflow.ollama_client.request.urlopen", fake_urlopen):
        client.chat({"model": "qwen3.5:9b", "messages": [], "think": True})

    assert captured["body"]["think"] is True


def test_unload_uses_generate_with_zero_keep_alive() -> None:
    client = OllamaClient(OllamaSettings("http://localhost:11434", 30, "5m", True))
    captured: dict = {}

    def fake_urlopen(req, timeout=0):
        captured["path"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__ = lambda *_a: response
        response.__exit__ = lambda *_a: None
        return response

    with patch("workflow.ollama_client.request.urlopen", fake_urlopen):
        client.unload("qwen2.5vl:7b")

    assert captured["path"].endswith("/api/generate")
    assert captured["body"]["model"] == "qwen2.5vl:7b"
    assert captured["body"]["keep_alive"] == 0
