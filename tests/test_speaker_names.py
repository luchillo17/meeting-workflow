"""Tests for heuristic speaker name inference."""

from __future__ import annotations

from workflow.speaker_names import infer_speaker_names


def test_infer_self_introduction() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hola, soy Sergio.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "text": "Buenas noches.", "speaker": "Speaker 2"},
    ]

    labeled, speaker_map = infer_speaker_names(segments, {"diarization": {}})

    assert speaker_map.get("Speaker 1") == "Sergio"
    assert labeled[0]["speaker"] == "Sergio"
    assert labeled[0]["speaker_id"] == "Speaker 1"


def test_infer_vocative_address() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "Sergio, ¿qué opinas del flujo?",
            "speaker": "Speaker 1",
        },
        {"start": 3.0, "end": 6.0, "text": "Yo creo que está claro.", "speaker": "Speaker 2"},
        {"start": 6.0, "end": 9.0, "text": "Lucho, una sugerencia rápida.", "speaker": "Speaker 2"},
        {
            "start": 9.0,
            "end": 12.0,
            "text": "Sí, tiene que partir del maestro.",
            "speaker": "Speaker 3",
        },
    ]

    labeled, speaker_map = infer_speaker_names(segments, {"diarization": {}})

    assert speaker_map.get("Speaker 2") == "Sergio"
    assert speaker_map.get("Speaker 3") == "Lucho"
    assert labeled[1]["speaker"] == "Sergio"


def test_roster_hints_allow_known_attendee_names() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "IPS, arranquemos.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "text": "Soy Luisca.", "speaker": "Speaker 2"},
    ]

    _, speaker_map = infer_speaker_names(
        segments,
        {"diarization": {"roster": ["Luisca"]}},
    )

    assert speaker_map.get("Speaker 2") == "Luisca"


def test_keeps_speaker_label_when_not_inferable() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Arranquemos la reunión.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "text": "De acuerdo.", "speaker": "Speaker 2"},
    ]

    labeled, speaker_map = infer_speaker_names(segments, {"diarization": {}})

    assert speaker_map == {}
    assert labeled[0]["speaker"] == "Speaker 1"
    assert labeled[1]["speaker"] == "Speaker 2"
