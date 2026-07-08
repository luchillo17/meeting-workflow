# Audio speaker diarization for transcripts

Extraction quality suffers when a flat transcript mixes proposals, questions, and
agreements from different people. Example: one speaker's "se me ocurre" idea was
classified as a key decision because the pipeline could not see turn boundaries.

**Decision:** After Whisper transcription, run optional **audio diarization**
(pyannote `speaker-diarization-3.1`) on the extracted mono 16 kHz WAV. Assign each
Whisper segment a display label `Speaker 1`, `Speaker 2`, … (stable per meeting).

**Name inference (always on with diarization):** A heuristic pass always tries to
replace `Speaker N` with spoken names from self-introductions (`soy Sergio`) and
direct address (`Lucho, una sugerencia`). When evidence is weak or ambiguous, the
label stays `Speaker N`. Optional `diarization.roster` hints help match known
attendees. Segments keep `speaker_id` when a display name is inferred.

Artifacts:

- `transcript.json` segments gain optional `speaker` (name or `Speaker N`)
- `transcript.txt` uses `[Sergio]` or `[Speaker 1]` lines (merged consecutive same-speaker text)
- `diarization.json` stores raw pyannote turns and any `speaker_map` for debugging

Whisper is unloaded before diarization to free VRAM. Diarization is gated by
`diarization.enabled` in `config.yaml` and requires `uv sync --extra diarization`
plus `HF_TOKEN` (Hugging Face model terms accepted).

**Not in scope:** calendar roster lookup, Teams tile attribution
(see [ADR 0005](0005-no-speaker-attribution-from-video.md)), or re-running
diarization on `--extract-only`.

**Extraction:** Prompts treat `[Speaker N]` as turn boundaries — proposals from
one speaker stay out of `key_decisions` unless another speaker explicitly agrees.
