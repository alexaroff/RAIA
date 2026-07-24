"""
RAIA Tool — Transcription module (fully local, Apple Silicon optimized)

Phase 1 stack:
  - Preprocess: FFmpeg → 16 kHz mono WAV
  - VAD (optional): Silero
  - STT: mlx-whisper (large-v3-turbo by default)
  - Backend abstraction ready for GigaAM-v3

Usage:
    from tool.transcription import LocalTranscriber, transcribe_file

    result = transcribe_file("interview.webm", model="turbo", language="ru")
    print(result.text)
"""

from .pipeline import LocalTranscriber, TranscriptionResult, transcribe_file
from .backends import TranscriptionBackend, MLXWhisperBackend, GigaAMBackend

__all__ = [
    "LocalTranscriber",
    "TranscriptionResult",
    "transcribe_file",
    "TranscriptionBackend",
    "MLXWhisperBackend",
    "GigaAMBackend",
]
