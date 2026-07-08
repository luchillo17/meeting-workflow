"""Tests for CUDA DLL path registration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from workflow.cuda_paths import ensure_cuda_dll_paths


def test_windows_registers_nvidia_bin_dirs(monkeypatch) -> None:
    added: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        "workflow.cuda_paths._windows_nvidia_bin_dirs",
        lambda: [Path("C:/fake/nvidia/cublas/bin"), Path("C:/fake/nvidia/cudnn/bin")],
    )
    monkeypatch.setattr("workflow.cuda_paths._torch_library_dirs", lambda: [])

    with patch("workflow.cuda_paths.os.add_dll_directory", side_effect=lambda p: added.append(p)):
        with patch.dict("os.environ", {}, clear=False):
            ensure_cuda_dll_paths()

    assert added == ["C:\\fake\\nvidia\\cublas\\bin", "C:\\fake\\nvidia\\cudnn\\bin"]
