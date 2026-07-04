"""Tests for transcript hallucination filters."""

from __future__ import annotations

from workflow.transcript_filters import (
    filter_hallucination_segments,
    is_hallucination_phrase,
    join_segment_texts,
)


def test_is_hallucination_phrase_spanish() -> None:
    assert is_hallucination_phrase("Gracias.", "es")
    assert not is_hallucination_phrase("Revisemos el tablero.", "es")


def test_filter_drops_short_gracias_segments() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Gracias."},
        {"start": 30.0, "end": 31.0, "text": "Gracias."},
        {
            "start": 390.0,
            "end": 395.0,
            "text": "¿Cuál es el rol de cada uno, la experiencia y la trayectoria?",
        },
    ]
    filtered = filter_hallucination_segments(segments, language="es")
    assert len(filtered) == 1
    assert "rol de cada uno" in filtered[0]["text"]


def test_filter_drops_repeated_short_identical_segments() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hola equipo"},
        {"start": 1.0, "end": 2.0, "text": "hola equipo"},
        {"start": 2.0, "end": 3.0, "text": "hola equipo"},
        {"start": 10.0, "end": 15.0, "text": "Contenido real de la reunión."},
    ]
    filtered = filter_hallucination_segments(segments, language="es", min_repeat_count=3)
    assert len(filtered) == 1
    assert filtered[0]["text"] == "Contenido real de la reunión."


def test_filter_keeps_long_gracias_segment() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Gracias a todos por venir y por el esfuerzo de esta semana.",
        }
    ]
    filtered = filter_hallucination_segments(segments, language="es")
    assert filtered == segments


def test_join_segment_texts() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hola."},
        {"start": 1.0, "end": 2.0, "text": "Mundo."},
    ]
    assert join_segment_texts(segments) == "Hola. Mundo."
