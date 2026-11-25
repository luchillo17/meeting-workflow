# Hybrid vision in v1, not transcript-only

Spanish product meetings rely on whiteboards, screen-shared parameters, slides,
and Miro diagrams. Speech-only transcripts miss field names, layout, and
visual agreements that never get spoken clearly.

**Decision:** v1 ships the full chain — **Transcript + Visual Capture +
Structured Extraction** — not a transcript-only shortcut.

Visual capture is **targeted**, not dense:

- ffmpeg scene detection + interval gap-fill (`frames` config)
- Spanish **visual-cue** keyword boost from transcript timestamps
- Cap per meeting (`max_frames`) for pilot-friendly cost

Vision describes **shared content** (slides, screen share, whiteboards), not
participant tiles. Tile-only frames are filtered before extraction
([ADR 0005](0005-no-speaker-attribution-from-video.md)); publish applies the
same filter when copying frame images ([ADR 0010](0010-batch-run-statistics.md)
quality signals include publishable vs raw vision counts).

**Consequences**

- Higher wall time and Ollama cost than transcript-only, but better extraction
  recall on UI/field discussions (convenios, portal, HC laboral)
- `visual_content.json` + curated `frames/` in published agent bundles
- Re-run `--vision-only` after prompt/filter changes without re-transcribing
