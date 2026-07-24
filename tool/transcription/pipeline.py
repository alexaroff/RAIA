"""
RAIA Tool — Local Transcription Pipeline (Phase 1)
Optimized for Apple Silicon M2/M3/M4 with 8–32 GB unified memory.
Primary backend: mlx-whisper (large-v3-turbo by default).

Design goals:
- Low peak memory (model unload after use, limited threads)
- Good Russian quality
- Reproducible
- Easy to swap backends later (GigaAM-v3 MLX etc.)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .preprocess import preprocess_audio, get_audio_duration
from .vad import get_speech_timestamps, merge_close_segments

logger = logging.getLogger(__name__)

# Limit OpenMP / BLAS threads early to prevent overheating on 8 GB M2
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: List[Dict[str, Any]]
    duration_s: float
    processing_time_s: float
    model: str
    backend: str = "mlx-whisper"

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
        """Write basic SRT from segments (if timestamps present)."""
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
        t = LocalTranscriber(model="mlx-community/whisper-large-v3-turbo")
        result = t.transcribe("interview.webm", language="ru")
        result.save_txt("out.txt")
        result.save_json("out.json")
        result.save_srt("out.srt")
    """

    # Recommended models for different RAM budgets
    MODELS = {
        "turbo": "mlx-community/whisper-large-v3-turbo",      # ~1.6 GB, best balance 8–16 GB
        "medium": "mlx-community/whisper-medium",             # safer for tight 8 GB
        "small": "mlx-community/whisper-small",               # very low mem, lower quality
        "large": "mlx-community/whisper-large-v3",            # highest quality, needs ≥16 GB comfortably
    }

    def __init__(
        self,
        model: str = "turbo",
        language: Optional[str] = "ru",
        use_vad: bool = True,
        word_timestamps: bool = True,
        verbose: bool = False,
    ):
        """
        model: short name ("turbo") or full HF/mlx path
        language: "ru" recommended for interviews; None = auto-detect
        """
        self.model_id = self.MODELS.get(model, model)
        self.language = language
        self.use_vad = use_vad
        self.word_timestamps = word_timestamps
        self.verbose = verbose
        self._model = None  # lazy load

        if verbose:
            logging.basicConfig(level=logging.INFO)

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import mlx_whisper
            # mlx_whisper does not keep a persistent model object in the simple API;
            # we just validate that the package is present.
            self._backend = "mlx-whisper"
            logger.info("Backend: mlx-whisper | model: %s", self.model_id)
        except ImportError as e:
            raise ImportError(
                "mlx-whisper is required for Apple Silicon. "
                "Install: pip install mlx mlx-whisper\n"
                "Note: only works on macOS with Apple Silicon."
            ) from e

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        keep_preprocessed: bool = False,
    ) -> TranscriptionResult:
        """
        Full pipeline: preprocess → (optional VAD) → STT → result.
        """
        start_time = time.perf_counter()
        audio_path = Path(audio_path)
        lang = language or self.language

        self._load_model()

        # 1. Preprocess
        preprocessed = preprocess_audio(audio_path)
        duration = get_audio_duration(preprocessed)
        logger.info("Audio duration: %.1f s", duration)

        try:
            # 2. Optional VAD (reduces work + helps long files on low RAM)
            speech_segments = None
            if self.use_vad and duration > 30:  # VAD mainly useful for longer files
                speech_segments = get_speech_timestamps(preprocessed)
                speech_segments = merge_close_segments(speech_segments)
                logger.info("VAD found %d speech segments", len(speech_segments))
            else:
                speech_segments = None

            # 3. Transcribe with mlx-whisper
            import mlx_whisper

            # mlx_whisper.transcribe accepts path and returns dict with text + segments
            # For long audio + low RAM we rely on internal chunking of the library.
            # language="ru" forces Russian and improves quality for our domain.
            result_dict = mlx_whisper.transcribe(
                str(preprocessed),
                path_or_hf_repo=self.model_id,
                language=lang,
                word_timestamps=self.word_timestamps,
                verbose=self.verbose,
                # Additional kwargs that help memory / quality
                # (mlx-whisper passes many of them through)
            )

            text = result_dict.get("text", "").strip()
            segments = result_dict.get("segments", [])
            detected_lang = result_dict.get("language", lang or "unknown")

            processing_time = time.perf_counter() - start_time

            result = TranscriptionResult(
                text=text,
                language=detected_lang,
                segments=segments,
                duration_s=duration,
                processing_time_s=processing_time,
                model=self.model_id,
                backend="mlx-whisper",
            )

            # Optional auto-save
            if output_dir is not None:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                stem = audio_path.stem
                result.save_txt(out / f"{stem}.txt")
                result.save_json(out / f"{stem}.json")
                result.save_srt(out / f"{stem}.srt")
                logger.info("Saved outputs to %s", out)

            rtf = duration / processing_time if processing_time > 0 else 0
            logger.info(
                "Done in %.1fs (RTF ≈ %.1fx) | model=%s",
                processing_time,
                rtf,
                self.model_id,
            )
            return result

        finally:
            # Clean temporary preprocessed file unless requested
            if not keep_preprocessed and preprocessed.exists():
                try:
                    preprocessed.unlink()
                except OSError:
                    pass

    def unload(self):
        """Explicitly free resources (for long-running processes)."""
        self._model = None
        # Force GC on low-RAM machines
        import gc
        gc.collect()


# Convenience function
def transcribe_file(
    audio_path: str | Path,
    model: str = "turbo",
    language: str = "ru",
    use_vad: bool = True,
    output_dir: Optional[str | Path] = None,
) -> TranscriptionResult:
    """One-shot transcription."""
    t = LocalTranscriber(model=model, language=language, use_vad=use_vad)
    try:
        return t.transcribe(audio_path, output_dir=output_dir)
    finally:
        t.unload()
