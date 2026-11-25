"""Graceful batch shutdown: signal handlers, subprocess cleanup, GPU unload."""

from __future__ import annotations

import atexit
import gc
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from filelock import FileLock, Timeout

_HARD_EXIT_SECONDS = 1.5


class BatchInterrupted(Exception):
    """Raised when the user or OS requests batch cancellation."""


class ShutdownCoordinator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._subprocesses: list[subprocess.Popen[str]] = []
        self._interrupt_requested = False
        self._handlers_installed = False
        self._batch_lock_guard = threading.Lock()
        self._batch_lock: FileLock | None = None
        self._graceful_exit_event = threading.Event()
        self._watchdog_started = False
        self._resources_released = False
        self._hard_exit_callbacks: list[Callable[[], None]] = []

    def prepare_for_batch(self) -> None:
        """Reset interrupt state so a new batch can run in the same process."""
        with self._lock:
            self._interrupt_requested = False
            self._resources_released = False
            self._watchdog_started = False
            self._cleanup_callbacks.clear()
            self._hard_exit_callbacks.clear()
        self._graceful_exit_event.set()
        self._graceful_exit_event.clear()

    def register_hard_exit(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._hard_exit_callbacks.append(callback)

    def _run_hard_exit_callbacks(self) -> None:
        with self._lock:
            callbacks = list(self._hard_exit_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    def interrupt_requested(self) -> bool:
        return self._interrupt_requested

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def track_subprocess(self, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            self._subprocesses.append(proc)

    def untrack_subprocess(self, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            if proc in self._subprocesses:
                self._subprocesses.remove(proc)

    def install_handlers(self) -> None:
        with self._lock:
            if self._handlers_installed:
                return
            signal.signal(signal.SIGINT, self._handle_signal)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, self._handle_signal)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, self._handle_signal)
            atexit.register(self.cleanup)
            self._handlers_installed = True

    def _handle_signal(self, signum: int, _frame: object) -> None:
        self.request_shutdown(f"signal {signum}")

    def request_shutdown(self, reason: str = "interrupt") -> None:
        with self._lock:
            if self._interrupt_requested:
                print("\nForce stopping…", file=sys.stderr)
                os._exit(130)
            self._interrupt_requested = True
        print(f"\nStopping batch ({reason})…", file=sys.stderr)
        print(
            "Waiting for current GPU step to finish (max "
            f"{_HARD_EXIT_SECONDS:.0f}s). Press Ctrl+C again to force quit.",
            file=sys.stderr,
        )
        self._terminate_subprocesses()
        self._start_hard_exit_watchdog()

    def _start_hard_exit_watchdog(self) -> None:
        with self._lock:
            if self._watchdog_started:
                return
            self._watchdog_started = True
        self._graceful_exit_event.clear()

        def _watchdog() -> None:
            deadline = time.monotonic() + _HARD_EXIT_SECONDS
            while time.monotonic() < deadline:
                if self._graceful_exit_event.wait(timeout=0.1):
                    return
            if self._graceful_exit_event.is_set():
                return
            self._run_hard_exit_callbacks()
            self.cleanup()
            os._exit(130)

        threading.Thread(target=_watchdog, daemon=True, name="batch-shutdown-watchdog").start()

    def acquire_batch_lock(self, output_dir: Path) -> None:
        with self._batch_lock_guard:
            message = f"Another batch is already running for {output_dir}"
            if self._batch_lock is not None:
                raise RuntimeError(message)
            output_dir.mkdir(parents=True, exist_ok=True)
            # The watchdog may release a lock acquired by the main thread.
            lock = FileLock(str(output_dir / ".batch.lock"), thread_local=False)
            try:
                lock.acquire(timeout=0)
            except Timeout as exc:
                raise RuntimeError(message) from exc
            self._batch_lock = lock

    def release_batch_lock(self) -> None:
        with self._batch_lock_guard:
            if self._batch_lock is not None:
                self._batch_lock.release()
                self._batch_lock = None

    def _terminate_subprocesses(self) -> None:
        with self._lock:
            subprocesses = list(self._subprocesses)
        for proc in subprocesses:
            if proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

    def _run_cleanup_callbacks(self) -> None:
        with self._lock:
            callbacks = list(self._cleanup_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    def _empty_cuda_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

    def cleanup(self) -> None:
        with self._lock:
            if self._resources_released:
                return
            self._resources_released = True
        self._terminate_subprocesses()
        self._run_cleanup_callbacks()
        self._empty_cuda_cache()

    def check_interrupted(self) -> None:
        if self._interrupt_requested:
            self._graceful_exit_event.set()
            raise BatchInterrupted("Batch cancelled")


_coordinator = ShutdownCoordinator()


def get_shutdown_coordinator() -> ShutdownCoordinator:
    return _coordinator
