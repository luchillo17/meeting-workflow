"""Register pip-installed NVIDIA CUDA libraries for CTranslate2 (all platforms)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _windows_nvidia_bin_dirs() -> list[Path]:
    try:
        import nvidia
    except ImportError:
        return []

    root = Path(next(iter(nvidia.__path__)))
    candidates = (
        root / "cublas" / "bin",
        root / "cudnn" / "bin",
        root / "cuda_nvrtc" / "bin",
    )
    return [path for path in candidates if path.is_dir()]


def _unix_nvidia_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib", "nvidia.cuda_nvrtc.lib"):
        try:
            module = __import__(module_name, fromlist=["__file__"])
            dirs.append(Path(module.__file__).parent)
        except ImportError:
            continue

    if dirs:
        return dirs

    try:
        import nvidia
    except ImportError:
        return []

    root = Path(next(iter(nvidia.__path__)))
    return [path for path in (root / "cublas" / "lib", root / "cudnn" / "lib") if path.is_dir()]


def _torch_library_dirs() -> list[Path]:
    try:
        import torch
    except ImportError:
        return []
    torch_root = Path(torch.__file__).resolve().parent
    return [path for path in (torch_root / "lib", torch_root) if path.is_dir()]


def nvidia_library_dirs() -> list[Path]:
    dirs = _torch_library_dirs()
    if sys.platform == "win32":
        dirs.extend(_windows_nvidia_bin_dirs())
    else:
        dirs.extend(_unix_nvidia_lib_dirs())
    seen: set[str] = set()
    unique: list[Path] = []
    for path in dirs:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _register_lib_dir(lib_dir: Path) -> None:
    path_str = str(lib_dir)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(path_str)
    if sys.platform == "win32":
        if path_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
    else:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [part for part in existing.split(os.pathsep) if part]
        if path_str not in parts:
            os.environ["LD_LIBRARY_PATH"] = path_str + (os.pathsep + existing if existing else "")


def ensure_cuda_dll_paths() -> None:
    """Make pip-installed cuBLAS/cuDNN discoverable before loading faster-whisper."""
    for lib_dir in nvidia_library_dirs():
        _register_lib_dir(lib_dir)
