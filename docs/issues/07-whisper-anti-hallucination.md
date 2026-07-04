## Parent

#1

## What to build

Reduce Whisper silence hallucinations in transcripts (e.g. repeated "Gracias." on Teams intro silence) by enabling faster-whisper quality options and post-filtering known phrase loops.

## Acceptance criteria

- [ ] `config.yaml` documents VAD, hallucination threshold, and phrase filter toggles
- [ ] Transcriber passes faster-whisper options from config (VAD, `condition_on_previous_text`, thresholds)
- [ ] Post-filter removes short known hallucination phrases and identical short repeats
- [ ] Unit tests cover filter logic and transcribe option wiring (no GPU in CI)
- [ ] Pilot recording re-run shows far fewer silence hallucinations in `transcript.txt`

## Blocked by

None (can merge independently of #5)
