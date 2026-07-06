"""Application settings from environment and config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from workflow.utils import load_config


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    config: dict
    ollama_base_url: str
    ollama_text_model: str
    ollama_vision_model: str

    @classmethod
    def load(cls, project_root: Path | None = None) -> Settings:
        root = project_root or Path.cwd()
        load_dotenv(root / ".env")
        config = load_config(root / "config.yaml")
        output_dir = Path(
            os.environ.get("OUTPUT_DIR", config.get("output", {}).get("base_dir", "output"))
        )
        return cls(
            output_dir=output_dir,
            config=config,
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL",
                config.get("ollama", {}).get("base_url", "http://localhost:11434"),
            ),
            ollama_text_model=os.environ.get(
                "OLLAMA_TEXT_MODEL", config.get("ollama", {}).get("text_model", "qwen2.5:7b")
            ),
            ollama_vision_model=os.environ.get(
                "OLLAMA_VISION_MODEL", config.get("ollama", {}).get("vision_model", "qwen2.5vl:7b")
            ),
        )

    def extraction_dir(self, recording_name: str) -> Path:
        from workflow.utils import slugify

        return self.output_dir / slugify(recording_name)
