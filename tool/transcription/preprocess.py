"""
Audio preprocessing for RAIA Tool transcription pipeline.
Converts any supported audio to 16 kHz mono WAV suitable for STT models.
Uses FFmpeg for reliability and low memory footprint.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def preprocess_audio(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Convert audio to 16 kHz mono WAV using FFmpeg.

    Returns path to the processed file.
    If output_path is None, creates a temporary file (caller must clean up if needed).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(tmp.name)
        tmp.close()
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        "-hide_banner",
        "-loglevel", "error",
        str(output_path),
    ]

    logger.info("Preprocessing %s → %s (16kHz mono)", input_path.name, output_path.name)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"FFmpeg failed: {stderr}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Install with: brew install ffmpeg (macOS)"
        ) from None

    return output_path


def get_audio_duration(path: str | Path) -> float:
    """Return duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning("Could not get duration: %s", e)
        return 0.0
