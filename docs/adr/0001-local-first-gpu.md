# Local-first GPU processing on Windows

Cloud transcription and vision APIs are viable but add per-meeting cost, data
egress, and opaque model behavior. Vital Link pilot runs happen on a developer
Windows workstation with an RTX 4080 (16 GB VRAM).

**Decision:** Process Meeting Recordings locally on Windows:

- **Whisper** via faster-whisper on CUDA (`whisper.device: cuda`)
- **Vision + extraction** via Ollama (`qwen3.5:9b` text + vision)
- **Diarization** via pyannote on CUDA when the `diarization` extra is installed

Models load **sequentially** — never Whisper + vision in VRAM at once.
`ollama.unload_between_stages` and explicit adapter `unload()` free GPU memory
between transcript, vision, and extraction.

**Consequences**

- ~$0 marginal API cost per meeting; HF token required for pyannote model download
- Pilot throughput is bounded by single-machine GPU and sequential stages
- CUDA DLL paths and torch+cu124 pins are part of setup (`docs/SETUP.md`)
- Batch statistics stay on local disk ([ADR 0010](0010-batch-run-statistics.md));
  no cloud metrics in v1
