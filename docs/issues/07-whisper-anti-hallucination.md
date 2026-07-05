## Parent

#1

## What to build

Reduce Whisper silence hallucinations in transcripts (e.g. repeated "Gracias." on Teams intro silence) by enabling faster-whisper quality options and post-filtering known phrase loops.

## Acceptance criteria

- [x] `config.yaml` documents VAD, hallucination threshold, and phrase filter toggles
- [x] Transcriber passes faster-whisper options from config (VAD, `condition_on_previous_text`, thresholds)
- [x] Post-filter removes short known hallucination phrases and identical short repeats
- [x] Unit tests cover filter logic and transcribe option wiring (no GPU in CI)
- [x] Pilot recording re-run shows far fewer silence hallucinations in `transcript.txt` (43 → 0 `"Gracias."` segments)

## Blocked by

None (merged via PR #10).
