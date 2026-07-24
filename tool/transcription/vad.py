"""
Lightweight Voice Activity Detection for RAIA Tool.
Primary backend: Silero VAD (via official package or ONNX fallback).
Designed for low memory on Apple Silicon 8 GB machines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_speech_timestamps(
    audio_path: str | Path,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    sampling_rate: int = 16000,
) -> List[Dict[str, float]]:
    """
    Return list of speech segments: [{"start": float, "end": float}, ...]
    Times in seconds.

    Falls back to full-audio if Silero is unavailable (graceful degradation).
    """
    audio_path = Path(audio_path)

    try:
        return _silero_vad(
            audio_path,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            sampling_rate=sampling_rate,
        )
    except Exception as e:
        logger.warning(
            "Silero VAD failed (%s). Falling back to full audio as one segment. "
            "Install with: pip install silero-vad torch torchaudio",
            e,
        )
        # Fallback: treat entire file as speech
        from .preprocess import get_audio_duration
        duration = get_audio_duration(audio_path)
        if duration <= 0:
            return []
        return [{"start": 0.0, "end": duration}]


def _silero_vad(
    audio_path: Path,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    sampling_rate: int,
) -> List[Dict[str, float]]:
    """Use official silero-vad package (pulls torch, but model is tiny ~2MB)."""
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps as silero_get

    # Limit threads to reduce memory pressure / overheating on M2 8GB
    import torch
    torch.set_num_threads(2)

    model = load_silero_vad(onnx=False)  # or True if onnxruntime preferred
    wav = read_audio(str(audio_path), sampling_rate=sampling_rate)

    timestamps = silero_get(
        wav,
        model,
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True,
    )
    return timestamps


def merge_close_segments(
    segments: List[Dict[str, float]],
    max_gap_s: float = 0.8,
) -> List[Dict[str, float]]:
    """Merge segments that are close together to reduce fragmentation."""
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["start"] - last["end"] <= max_gap_s:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg.copy())
    return merged
