"""Cross-platform environment checks before a Workflow Run."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.request import urlopen

from workflow.cuda_paths import ensure_cuda_dll_paths
from workflow.settings import Settings

CheckStatus = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    name: str
    status: CheckStatus
    detail: str


def _cuda_library_dirs() -> list[Path]:
    from workflow.cuda_paths import nvidia_library_dirs

    return nvidia_library_dirs()


def _ollama_tags(base_url: str) -> list[str] | None:
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as response:
            import json

            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, ValueError):
        return None
    return [item.get("name", "") for item in payload.get("models", [])]


def _model_available(tags: list[str], model: str) -> bool:
    if not model:
        return False
    return any(tag == model or tag.startswith(f"{model}:") for tag in tags)


def run_preflight(project_root: Path | None = None) -> list[Check]:
    root = project_root or Path.cwd()
    checks: list[Check] = []

    # Runtime guard for `python -m workflow.cli` without uv (requires-python is 3.12+).
    if sys.version_info < (3, 12):  # noqa: UP036
        checks.append(
            Check(
                "python",
                "fail",
                f"{sys.version_info.major}.{sys.version_info.minor} (requires 3.12+)",
            )
        )
    else:
        checks.append(Check("python", "ok", f"{sys.version_info.major}.{sys.version_info.minor}"))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        checks.append(Check("ffmpeg", "ok", ffmpeg))
    else:
        checks.append(
            Check(
                "ffmpeg",
                "fail",
                "not found in PATH (install via package manager or https://ffmpeg.org)",
            )
        )

    config_path = root / "config.yaml"
    if config_path.is_file():
        checks.append(Check("config.yaml", "ok", str(config_path)))
    else:
        checks.append(Check("config.yaml", "fail", f"missing at {config_path}"))

    env_path = root / ".env"
    if env_path.is_file():
        checks.append(Check(".env", "ok", str(env_path)))
    else:
        checks.append(Check(".env", "warn", "missing — copy .env.example to .env"))

    cuda_dirs = _cuda_library_dirs()
    whisper_device = "cuda"
    if config_path.is_file():
        from workflow.utils import load_config

        whisper_device = load_config(config_path).get("whisper", {}).get("device", "cuda")

    if sys.platform == "darwin":
        if whisper_device == "cuda":
            checks.append(
                Check(
                    "gpu",
                    "fail",
                    "macOS has no CUDA — set whisper.device to cpu in config.yaml",
                )
            )
        else:
            checks.append(Check("gpu", "ok", f"whisper.device={whisper_device}"))
    elif cuda_dirs:
        checks.append(Check("cuda libraries", "ok", f"{len(cuda_dirs)} NVIDIA lib dir(s)"))
        try:
            ensure_cuda_dll_paths()
            import ctranslate2

            count = ctranslate2.get_cuda_device_count()
            if whisper_device == "cuda" and count == 0:
                checks.append(
                    Check("cuda device", "fail", "whisper.device=cuda but no GPU detected")
                )
            elif count > 0:
                checks.append(Check("cuda device", "ok", f"{count} device(s)"))
            else:
                checks.append(
                    Check("cuda device", "warn", "no CUDA devices (CPU fallback possible)")
                )
        except ImportError:
            checks.append(Check("cuda device", "warn", "ctranslate2 not importable"))
        except RuntimeError as exc:
            checks.append(Check("cuda device", "fail", str(exc)))
    else:
        checks.append(
            Check(
                "cuda libraries",
                "warn" if whisper_device != "cuda" else "fail",
                "nvidia-cublas-cu12 / nvidia-cudnn-cu12 not found — run uv sync",
            )
        )

    settings: Settings | None = None
    try:
        settings = Settings.load(root)
    except OSError as exc:
        checks.append(Check("settings", "fail", str(exc)))

    if settings is not None:
        tags = _ollama_tags(settings.ollama_base_url)
        if tags is None:
            checks.append(
                Check(
                    "ollama",
                    "warn",
                    (
                        f"not reachable at {settings.ollama_base_url} "
                        "(needed for Visual Capture + Extraction)"
                    ),
                )
            )
        else:
            checks.append(Check("ollama", "ok", settings.ollama_base_url))
            if _model_available(tags, settings.ollama_text_model):
                checks.append(Check("ollama text model", "ok", settings.ollama_text_model))
            else:
                checks.append(
                    Check(
                        "ollama text model",
                        "warn",
                        (
                            f"{settings.ollama_text_model} not pulled "
                            f"(ollama pull {settings.ollama_text_model})"
                        ),
                    )
                )
            if _model_available(tags, settings.ollama_vision_model):
                checks.append(Check("ollama vision model", "ok", settings.ollama_vision_model))
            else:
                checks.append(
                    Check(
                        "ollama vision model",
                        "warn",
                        (
                            f"{settings.ollama_vision_model} not pulled "
                            f"(ollama pull {settings.ollama_vision_model})"
                        ),
                    )
                )

        output_dir = settings.output_dir
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe = output_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            checks.append(Check("output dir", "ok", str(output_dir.resolve())))
        except OSError as exc:
            checks.append(Check("output dir", "fail", f"{output_dir}: {exc}"))

        import os

        raw_recording = os.environ.get("RECORDING_PATH", "").strip()
        if raw_recording:
            recording = Path(raw_recording.strip('"').strip("'"))
            if recording.is_file():
                checks.append(Check("recording", "ok", str(recording)))
            else:
                checks.append(Check("recording", "fail", f"RECORDING_PATH not found: {recording}"))
        else:
            checks.append(
                Check(
                    "recording",
                    "warn",
                    "RECORDING_PATH unset — use process --file or set it in .env",
                )
            )

    if config_path.is_file():
        from workflow.utils import load_config

        config = load_config(config_path)
        diar_cfg = config.get("diarization", {})
        if bool(diar_cfg.get("enabled", False)):
            import os

            from workflow.diarization import resolve_hf_token

            token = resolve_hf_token(config)
            if token:
                checks.append(Check("diarization token", "ok", "HF token set"))
            else:
                checks.append(
                    Check(
                        "diarization token",
                        "fail",
                        "HF_TOKEN unset — accept pyannote model terms and set HF_TOKEN in .env",
                    )
                )
            try:
                import pyannote.audio  # noqa: F401

                checks.append(Check("pyannote.audio", "ok", "installed"))
            except ImportError:
                checks.append(
                    Check(
                        "pyannote.audio",
                        "fail",
                        "missing — run: uv sync --extra diarization",
                    )
                )

    return checks


def has_failures(checks: list[Check]) -> bool:
    return any(check.status == "fail" for check in checks)
