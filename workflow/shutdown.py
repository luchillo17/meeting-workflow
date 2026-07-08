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
        self._lock_path: Path | None = None
        self._graceful_exit_event = threading.Event()
        self._watchdog_started = False

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
        self.cleanup()
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
            os._exit(130)

        threading.Thread(target=_watchdog, daemon=True, name="batch-shutdown-watchdog").start()

    def acquire_batch_lock(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = output_dir / ".batch.lock"
        if lock_path.is_file():
            try:
                old_pid = int(lock_path.read_text(encoding="utf-8").strip())
            except ValueError:
                old_pid = -1
            if old_pid > 0 and _process_alive(old_pid) and old_pid != os.getpid():
                raise RuntimeError(
                    f"Another batch is already running (PID {old_pid}). "
                    f"Stop it first: Stop-Process -Id {old_pid} -Force"
                )
            lock_path.unlink(missing_ok=True)
        lock_path.write_text(str(os.getpid()), encoding="utf-8")
        self._lock_path = lock_path

    def release_batch_lock(self) -> None:
        if self._lock_path is None:
            return
        try:
            if self._lock_path.is_file():
                current = self._lock_path.read_text(encoding="utf-8").strip()
                if current == str(os.getpid()):
                    self._lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        self._lock_path = None

    def cleanup(self) -> None:
        with self._lock:
            subprocesses = list(self._subprocesses)
            callbacks = list(self._cleanup_callbacks)

        for proc in subprocesses:
            if proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        self.release_batch_lock()

    def check_interrupted(self) -> None:
        if self._interrupt_requested:
            self._graceful_exit_event.set()
            raise BatchInterrupted("Batch cancelled")


_coordinator = ShutdownCoordinator()


def get_shutdown_coordinator() -> ShutdownCoordinator:
    return _coordinator


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
