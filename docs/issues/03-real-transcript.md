## Parent

#1

## What to build

Replace the mocked transcriber adapter with a real implementation: extract mono audio via ffmpeg, transcribe Spanish speech with faster-whisper per `config.yaml`, and write `transcript.json` + `transcript.txt` into the Extraction. Other stages may remain mocked so the Workflow Run still completes end-to-end in tests.

Unload the whisper model before later stages (sequential VRAM use).

## Acceptance criteria

- [x] Real Meeting Recording produces readable Spanish `transcript.txt` and segmented `transcript.json`
- [x] Transcriber adapter is injectable; WorkflowRunner tests still pass with fakes
- [x] Whisper model is not held in memory after transcription stage completes
- [x] Audio extraction failures surface a clear error

## Blocked by

#2 (WorkflowRunner + mocked adapters)
