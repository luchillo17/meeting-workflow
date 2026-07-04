## Parent

#1

## What to build

Add real **Visual Capture**: ffmpeg frame extraction (scene-change + interval fallback + transcript visual-cue boost), frame cap, resize, and near-duplicate dedup per `config.yaml`. Write `frames/` and `frames.json`. Run vision adapter (mocked in tests, real Ollama in production) to produce `visual_content.json`.

## Acceptance criteria

- [ ] Frames extracted when a recording has scene changes or visual-cue segments in the Transcript
- [ ] Frame count respects `max_frames`; images resized per config
- [ ] `visual_content.json` lists timestamp, type, and description per frame
- [ ] WorkflowRunner tests pass with fake vision adapter

## Blocked by

#3 (real Transcript stage)
