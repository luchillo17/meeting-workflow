## Parent

#1

## Build

Real **Visual Capture**: ffmpeg frames (scene-change + interval fallback + transcript visual-cue boost), cap, resize, near-duplicate dedup per `config.yaml`. Write `frames/`, `frames.json`. Vision adapter: mocked in tests, real Ollama in prod. Output `visual_content.json`.

## Acceptance

- [x] Frames when scene changes or visual-cue segments in Transcript
- [x] Frame count respects `max_frames`; resize per config
- [x] `visual_content.json`: timestamp, type, description per frame
- [x] WorkflowRunner tests pass with fake vision

## Blocked by

#3
