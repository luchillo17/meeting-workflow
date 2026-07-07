"""Quick stats for pilot extraction outputs (dev helper)."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("output")
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        extraction_path = folder / "extraction.json"
        if not extraction_path.is_file():
            continue
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        chapters_path = folder / "chapters.json"
        chapters = 0
        if chapters_path.is_file():
            data = json.loads(chapters_path.read_text(encoding="utf-8"))
            chapters = len(data) if isinstance(data, list) else 0
        decisions = len(extraction.get("key_decisions") or [])
        actions = len(extraction.get("action_items") or [])
        topic = (extraction.get("topic") or "")[:55]
        print(f"{folder.name[:42]:42} ch={chapters:2} dec={decisions} act={actions} | {topic}")


if __name__ == "__main__":
    main()
