"""Tests for preflight checks."""

from __future__ import annotations

from pathlib import Path

from workflow.preflight import Check, has_failures, run_preflight


def test_has_failures_detects_fail_status() -> None:
    assert has_failures([Check("a", "ok", "")]) is False
    assert has_failures([Check("a", "warn", ""), Check("b", "fail", "")]) is True


def test_run_preflight_reports_python_and_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("output:\n  base_dir: output\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("OUTPUT_DIR=output\n", encoding="utf-8")

    checks = {check.name: check for check in run_preflight(tmp_path)}

    assert checks["python"].status == "ok"
    assert checks["config.yaml"].status == "ok"
    assert checks[".env"].status == "warn"
