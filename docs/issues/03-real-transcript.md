## Parent

#1

## Build

Replace mocked transcriber with real: ffmpeg mono audio, faster-whisper Spanish per `config.yaml`, write `transcript.json` + `transcript.txt`. Other stages may stay mocked for end-to-end tests.

Unload whisper model before later stages.

## Acceptance

- [x] Real `.mp4` produces Spanish `transcript.txt` + segmented `transcript.json`
- [x] Transcriber injectable; WorkflowRunner tests pass with fakes
- [x] Whisper not in memory after transcription
- [x] Audio extraction failures = clear error

## Blocked by

#2
