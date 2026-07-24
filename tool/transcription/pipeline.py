"""
RAIA Tool — Local Transcription Pipeline (Phase 1)
Optimized for Apple Silicon M2/M3/M4 with 8–32 GB unified memory.

Design goals:
- Low peak memory (model unload after use, limited threads)
- Good Russian quality
- Reproducible
- Easy to swap backends (GigaAM later)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .backends import MLXWhisperBackend, TranscriptionBackend, BackendResult
from .preprocess import preprocess_audio, get_audio_duration
from .vad import get_speech_timestamps, merge_close_segments

logger = logging.getLogger(__name__)

# Limit OpenMP / BLAS threads early — critical on M2 8 GB
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: List[Dict[str, Any]]
    duration_s: float
    processing_time_s: float
    model: str
    backend: str = "mlx-whisper"
    vad_segments: Optional[List[Dict[str, float]]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_txt(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.text.strip() + "\n", encoding="utf-8")

    def save_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_srt(self, path: Union[str, Path]) -> None:
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = seg.get("start", 0.0)
            end = seg.get("end", start + 1.0)
            text = seg.get("text", "").strip()
            if not text:
                continue
            lines.append(str(i))
            lines.append(f"{_format_ts(start)} --> {_format_ts(end)}")
            lines.append(text)
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")


def _format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class LocalTranscriber:
    """
    Main entry point for local transcription.

    Example:
        t = LocalTranscriber(model="turbo")
        result = t.transcribe("interview.webm", language="ru")
        result.save_txt("out.txt")
    """

    def __init__(
        self,
        model: str = "turbo",
        language: Optional[str] = "ru",
        use_vad: bool = True,
        word_timestamps: bool = True,
        backend: Optional[TranscriptionBackend] = None,
        verbose: bool = False,
    ):
        """
        model: short name ("turbo"/"medium"/"small"/"large") or full repo id
        language: "ru" recommended; None = auto-detect
        use_vad: run Silero VAD (useful for long files, currently informational)
        backend: optional custom backend (default = MLXWhisperBackend)
        """
        self.language = language
        self.use_vad = use_vad
        self.word_timestamps = word_timestamps
        self.verbose = verbose

        if backend is not None:
            self.backend = backend
        else:
            self.backend = MLXWhisperBackend(model=model)

        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        keep_preprocessed: bool = False,
    ) -> TranscriptionResult:
        """
        Full pipeline:
          1. FFmpeg preprocess → 16 kHz mono
          2. Optional Silero VAD (logs speech regions)
          3. STT via selected backend
          4. Optional auto-save txt/json/srt
        """
        start_time = time.perf_counter()
        audio_path = Path(audio_path)
        lang = language or self.language

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 1. Preprocess
        preprocessed = preprocess_audio(audio_path)
        duration = get_audio_duration(preprocessed)
        logger.info("Audio duration: %.1f s | backend=%s", duration, self.backend.name)

        vad_segments: Optional[List[Dict[str, float]]] = None

        try:
            # 2. VAD (currently informational + stored in result)
            #    Full segment-wise transcription will be added after first real tests.
            if self.use_vad and duration > 20:
                try:
                    vad_segments = get_speech_timestamps(preprocessed)
                    vad_segments = merge_close_segments(vad_segments)
                    speech_ratio = (
                        sum(s["end"] - s["start"] for s in vad_segments) / duration
                        if duration > 0 else 0
                    )
                    logger.info(
                        "VAD: %d segments, speech ratio ≈ %.0f%%",
                        len(vad_segments),
                        speech_ratio * 100,
                    )
                except Exception as e:
                    logger.warning("VAD skipped: %s", e)
                    vad_segments = None

            # 3. STT
            backend_result: BackendResult = self.backend.transcribe(
                preprocessed,
                language=lang,
                word_timestamps=self.word_timestamps,
                verbose=self.verbose,
            )

            processing_time = time.perf_counter() - start_time

            result = TranscriptionResult(
                text=backend_result.text,
                language=backend_result.language,
                segments=backend_result.segments,
                duration_s=duration,
                processing_time_s=processing_time,
                model=getattr(self.backend, "model_id", self.backend.name),
                backend=self.backend.name,
                vad_segments=vad_segments,
            )

            # 4. Auto-save
            if output_dir is not None:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                stem = audio_path.stem
                result.save_txt(out / f"{stem}.txt")
                result.save_json(out / f"{stem}.json")
                result.save_srt(out / f"{stem}.srt")
                logger.info("Saved → %s/{%s.txt,.json,.srt}", out, stem)

            rtf = duration / processing_time if processing_time > 0 else 0
            logger.info(
                "Done in %.1fs (≈%.1fx realtime) | model=%s",
                processing_time,
                rtf,
                result.model,
            )
            return result

        finally:
            if not keep_preprocessed and preprocessed.exists():
                try:
                    preprocessed.unlink()
                except OSError:
                    pass
            self.backend.unload()

    def unload(self) -> None:
        self.backend.unload()
        import gc
        gc.collect()


def transcribe_file(
    audio_path: str | Path,
    model: str = "turbo",
    language: str = "ru",
    use_vad: bool = True,
    output_dir: Optional[str | Path] = None,
    verbose: bool = False,
) -> TranscriptionResult:
    """One-shot helper."""
    t = LocalTranscriber(
        model=model,
        language=language,
        use_vad=use_vad,
        verbose=verbose,
    )
    try:
        return t.transcribe(audio_path, output_dir=output_dir)
    finally:
        t.unload()
