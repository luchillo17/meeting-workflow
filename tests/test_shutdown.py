"""Tests for graceful batch shutdown."""

from __future__ import annotations

from pathlib import Path

import pytest

from workflow.shutdown import BatchInterrupted, ShutdownCoordinator


def test_check_interrupted_raises_after_request(tmp_path: Path) -> None:
    coordinator = ShutdownCoordinator()
    coordinator.request_shutdown("test")
    with pytest.raises(BatchInterrupted):
        coordinator.check_interrupted()


def test_second_interrupt_force_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = ShutdownCoordinator()
    coordinator.request_shutdown("first")
    with pytest.raises(BatchInterrupted):
        coordinator.check_interrupted()  # cancel watchdog

    exits: list[int] = []
    monkeypatch.setattr("workflow.shutdown.os._exit", lambda code: exits.append(code))

    coordinator.request_shutdown("second")
    assert exits == [130]


def test_batch_lock_acquire_and_release(tmp_path: Path) -> None:
    coordinator = ShutdownCoordinator()
    coordinator.acquire_batch_lock(tmp_path)
    lock_path = tmp_path / ".batch.lock"
    assert lock_path.is_file()
    coordinator.release_batch_lock()
    assert not lock_path.is_file()


def test_batch_lock_rejects_stale_file(tmp_path: Path) -> None:
    lock_path = tmp_path / ".batch.lock"
    lock_path.write_text("99999999", encoding="utf-8")
    coordinator = ShutdownCoordinator()
    coordinator.acquire_batch_lock(tmp_path)
    assert lock_path.read_text(encoding="utf-8") != "99999999"


def test_cleanup_runs_registered_callbacks() -> None:
    coordinator = ShutdownCoordinator()
    seen: list[str] = []

    coordinator.register_cleanup(lambda: seen.append("ran"))
    coordinator.cleanup()
    assert seen == ["ran"]


def test_cleanup_does_not_release_batch_lock(tmp_path: Path) -> None:
    coordinator = ShutdownCoordinator()
    coordinator.acquire_batch_lock(tmp_path)
    lock_path = tmp_path / ".batch.lock"
    coordinator.cleanup()
    assert lock_path.is_file()
    coordinator.release_batch_lock()


def test_cleanup_is_idempotent() -> None:
    coordinator = ShutdownCoordinator()
    seen: list[str] = []
    coordinator.register_cleanup(lambda: seen.append("ran"))
    coordinator.cleanup()
    coordinator.cleanup()
    assert seen == ["ran"]


def test_request_shutdown_defers_adapter_unload() -> None:
    coordinator = ShutdownCoordinator()
    seen: list[str] = []
    coordinator.register_cleanup(lambda: seen.append("ran"))
    coordinator.request_shutdown("test")
    with pytest.raises(BatchInterrupted):
        coordinator.check_interrupted()
    assert seen == []
    coordinator.cleanup()
    assert seen == ["ran"]


def test_prepare_for_batch_resets_interrupt_state() -> None:
    coordinator = ShutdownCoordinator()
    coordinator.request_shutdown("test")
    with pytest.raises(BatchInterrupted):
        coordinator.check_interrupted()
    coordinator.cleanup()
    coordinator.prepare_for_batch()
    coordinator.check_interrupted()


def test_hard_exit_callbacks_run() -> None:
    coordinator = ShutdownCoordinator()
    seen: list[str] = []
    coordinator.register_hard_exit(lambda: seen.append("hard"))
    coordinator._run_hard_exit_callbacks()
    assert seen == ["hard"]
