## Parent

#1

## Build

Cut Whisper silence hallucinations (e.g. repeated "Gracias." on intro silence). faster-whisper quality options + post-filter known phrase loops.

## Acceptance

- [x] `config.yaml` documents VAD, hallucination threshold, phrase filter
- [x] Transcriber passes faster-whisper options from config
- [x] Post-filter removes known hallucination phrases + identical short repeats
- [x] Unit tests: filter logic + transcribe wiring (no GPU)
- [x] Pilot re-run: 43 → 0 `"Gracias."` segments in `transcript.txt`

## Blocked by

None (PR #10)
