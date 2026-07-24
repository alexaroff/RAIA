"""
RAIA Tool — Transcription module (fully local, Apple Silicon optimized)

Phase 1 stack:
  - Preprocess: FFmpeg → 16 kHz mono WAV
  - VAD (optional): Silero + segment-wise transcription
  - STT: mlx-whisper (default) / GigaAM-v3 (skeleton)
  - Auto model selection by RAM
  - config.toml + RAIA_* env vars

Usage:
    from tool.transcription import LocalTranscriber, transcribe_file

    result = transcribe_file("interview.webm", model="auto")
    print(result.text)
"""

from .pipeline import LocalTranscriber, TranscriptionResult, transcribe_file
from .backends import (
    TranscriptionBackend,
    MLXWhisperBackend,
    GigaAMBackend,
    create_backend,
)
from .config import TranscribeConfig, load_config

__all__ = [
    "LocalTranscriber",
    "TranscriptionResult",
    "transcribe_file",
    "TranscriptionBackend",
    "MLXWhisperBackend",
    "GigaAMBackend",
    "create_backend",
    "TranscribeConfig",
    "load_config",
]
