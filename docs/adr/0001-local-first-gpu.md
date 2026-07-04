# Local-first GPU processing on Windows

Teams Meeting Recordings are processed entirely on the developer's Windows PC using faster-whisper (CUDA) and Ollama (Qwen2.5 + Qwen2.5-VL). API cost is ~$0. Models load sequentially — never whisper and vision simultaneously — to stay within 16 GB VRAM on an RTX 4080.
