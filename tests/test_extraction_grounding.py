"""Tests for deterministic extraction grounding."""

from __future__ import annotations

from workflow.extraction_grounding import (
    apply_deterministic_grounding,
    compute_grounding_ratio,
    dedupe_similar_strings,
    is_deadline_grounded,
    is_grounded_in_source,
)


def test_is_grounded_in_source_accepts_supported_spanish_bullet() -> None:
    transcript = "Acordamos validar el convenio con la IPS antes del viernes."
    bullet = "Validar el convenio con la IPS antes del viernes."

    assert is_grounded_in_source(bullet, transcript)


def test_is_grounded_in_source_rejects_unrelated_bullet() -> None:
    transcript = "Hablamos del portal paciente y diseño responsivo."
    bullet = "Contratar diez ingenieros para el proyecto de blockchain."

    assert not is_grounded_in_source(bullet, transcript)


def test_is_deadline_grounded_accepts_spoken_date_fragment() -> None:
    transcript = "Nos reunimos el día 26 de mayo para revisar el portal."

    assert is_deadline_grounded("26/05/2026", transcript)


def test_apply_deterministic_grounding_drops_unsupported_items() -> None:
    raw = {
        "topic": "Portal",
        "key_decisions": [
            "Priorizar diseño responsivo del portal paciente",
            "Lanzar producto en blockchain el próximo trimestre",
        ],
        "action_items": [
            {"owner": "", "task": "Ajustar menú hamburguesa en móvil", "deadline": "lunes"},
            {"owner": "", "task": "Migrar todo a Kubernetes", "deadline": ""},
        ],
        "blockers_risks": [],
        "status_updates": [],
        "technical_details": ["Pantalla Figma del portal con banner no responsivo"],
        "open_questions": [],
        "next_steps": [],
    }
    transcript = (
        "Priorizamos el diseño responsivo del portal paciente y ajustar el menú hamburguesa "
        "en móvil para el lunes."
    )
    visual = "Pantalla Figma del portal con banner no responsivo"

    filtered = apply_deterministic_grounding(
        raw,
        transcript_text=transcript,
        visual_text=visual,
    )

    assert len(filtered["key_decisions"]) == 1
    assert len(filtered["action_items"]) == 1
    assert filtered["action_items"][0]["deadline"] == "lunes"
    assert len(filtered["technical_details"]) == 1


def test_compute_grounding_ratio() -> None:
    extraction = {
        "key_decisions": ["Validar convenio IPS"],
        "action_items": [{"owner": "", "task": "Inventado totalmente", "deadline": ""}],
        "blockers_risks": [],
        "status_updates": [],
        "technical_details": [],
        "open_questions": [],
        "next_steps": [],
    }
    transcript = "Debemos validar el convenio con la IPS esta semana."

    ratio, grounded, total = compute_grounding_ratio(extraction, transcript_text=transcript)

    assert total == 2
    assert grounded == 1
    assert ratio == 0.5


def test_dedupe_similar_strings() -> None:
    items = [
        "Validar convenio con la IPS",
        "Validar el convenio IPS",
        "Revisar portal paciente",
    ]

    deduped = dedupe_similar_strings(items)

    assert len(deduped) == 2
