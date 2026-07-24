"""
RAIA Tool — Local Transcription Pipeline (Phase 1)
Optimized for Apple Silicon M2/M3/M4 with 8–32 GB unified memory.

Features:
- Backend abstraction (mlx-whisper / GigaAM)
- Segment-wise transcription via VAD (memory-friendly for long files)
- Auto model selection by available RAM
- config.toml + env overrides
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .backends import (
    BackendResult,
    TranscriptionBackend,
    create_backend,
)
from .config import TranscribeConfig, load_config
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
    segment_wise: bool = False

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


def _extract_segment(
    source_wav: Path,
    start_s: float,
    end_s: float,
    out_path: Path,
) -> Path:
    """Cut a segment with FFmpeg (stream copy when possible)."""
    import subprocess

    duration = max(0.05, end_s - start_s)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(source_wav),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-hide_banner", "-loglevel", "error",
        str(out_path),
    ]
    # Some containers don't allow stream-copy with -ss after -i cleanly;
    # fall back to re-encode if needed.
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.3f}",
            "-i", str(source_wav),
            "-t", f"{duration:.3f}",
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le",
            "-hide_banner", "-loglevel", "error",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def _split_long_segments(
    segments: List[Dict[str, float]],
    max_s: float,
) -> List[Dict[str, float]]:
    """Further split VAD segments longer than max_s."""
    out: List[Dict[str, float]] = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        length = end - start
        if length <= max_s:
            out.append(seg)
            continue
        t = start
        while t < end:
            chunk_end = min(t + max_s, end)
            out.append({"start": t, "end": chunk_end})
            t = chunk_end
    return out


class LocalTranscriber:
    """
    Main entry point for local transcription.

    Example:
        t = LocalTranscriber(model="auto")          # picks by RAM
        result = t.transcribe("interview.webm")
        result.save_txt("out.txt")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        language: Optional[str] = None,
        use_vad: Optional[bool] = None,
        word_timestamps: Optional[bool] = None,
        backend: Optional[str | TranscriptionBackend] = None,
        config: Optional[TranscribeConfig] = None,
        verbose: bool = False,
    ):
        # Merge config
        overrides = {
            "model": model,
            "language": language,
            "use_vad": use_vad,
            "word_timestamps": word_timestamps,
        }
        if isinstance(backend, str):
            overrides["backend"] = backend

        self.cfg = config or load_config(overrides=overrides)

        self.verbose = verbose
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )

        # Resolve model (auto → concrete)
        resolved = self.cfg.resolve_model()

        # Build backend
        if isinstance(backend, TranscriptionBackend):
            self.backend = backend
        else:
            backend_name = self.cfg.backend
            self.backend = create_backend(backend_name, model=resolved)

        self.language = self.cfg.language
        self.use_vad = self.cfg.use_vad
        self.word_timestamps = self.cfg.word_timestamps

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        keep_preprocessed: bool = False,
        force_segment_wise: Optional[bool] = None,
    ) -> TranscriptionResult:
        """
        Full pipeline:
          1. FFmpeg preprocess → 16 kHz mono
          2. Optional Silero VAD
          3. Full-file OR segment-wise STT (memory-friendly)
          4. Optional auto-save
        """
        start_time = time.perf_counter()
        audio_path = Path(audio_path)
        lang = language or self.language

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 1. Preprocess
        preprocessed = preprocess_audio(audio_path)
        duration = get_audio_duration(preprocessed)
        logger.info(
            "Audio duration: %.1f s | backend=%s | model=%s",
            duration,
            self.backend.name,
            getattr(self.backend, "model_id", getattr(self.backend, "model_name", "?")),
        )

        vad_segments: Optional[List[Dict[str, float]]] = None
        segment_wise = False

        try:
            # 2. VAD
            if self.use_vad and duration >= self.cfg.vad_min_duration_s:
                try:
                    vad_segments = get_speech_timestamps(preprocessed)
                    vad_segments = merge_close_segments(vad_segments)
                    vad_segments = _split_long_segments(
                        vad_segments, self.cfg.max_segment_s
                    )
                    speech_dur = sum(s["end"] - s["start"] for s in vad_segments)
                    speech_ratio = speech_dur / duration if duration > 0 else 0
                    logger.info(
                        "VAD: %d segments, speech ≈ %.0f%% (%.0fs)",
                        len(vad_segments),
                        speech_ratio * 100,
                        speech_dur,
                    )
                except Exception as e:
                    logger.warning("VAD skipped: %s", e)
                    vad_segments = None

            # Decide segment-wise
            do_segment_wise = False
            if force_segment_wise is not None:
                do_segment_wise = force_segment_wise
            elif (
                vad_segments
                and duration >= self.cfg.segment_wise_threshold_s
                and len(vad_segments) > 1
            ):
                do_segment_wise = True

            # 3. STT
            if do_segment_wise and vad_segments:
                logger.info("Using segment-wise transcription (%d chunks)", len(vad_segments))
                backend_result = self._transcribe_segments(
                    preprocessed, vad_segments, lang
                )
                segment_wise = True
            else:
                backend_result = self.backend.transcribe(
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
                model=getattr(
                    self.backend, "model_id",
                    getattr(self.backend, "model_name", self.backend.name),
                ),
                backend=self.backend.name,
                vad_segments=vad_segments,
                segment_wise=segment_wise,
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
                "Done in %.1fs (≈%.1fx realtime) | segment_wise=%s",
                processing_time, rtf, segment_wise,
            )
            return result

        finally:
            if not keep_preprocessed and preprocessed.exists():
                try:
                    preprocessed.unlink()
                except OSError:
                    pass
            self.backend.unload()

    def _transcribe_segments(
        self,
        full_wav: Path,
        vad_segments: List[Dict[str, float]],
        language: Optional[str],
    ) -> BackendResult:
        """
        Transcribe each VAD segment separately and stitch results.
        Keeps peak memory low on long interviews.
        """
        all_text_parts: List[str] = []
        all_segments: List[Dict[str, Any]] = []
        detected_lang = language or "ru"

        with tempfile.TemporaryDirectory(prefix="raia_seg_") as tmp:
            tmp_dir = Path(tmp)
            for idx, seg in enumerate(vad_segments):
                start_s, end_s = seg["start"], seg["end"]
                if end_s - start_s < 0.15:
                    continue

                seg_path = tmp_dir / f"seg_{idx:04d}.wav"
                try:
                    _extract_segment(full_wav, start_s, end_s, seg_path)
                except Exception as e:
                    logger.warning("Failed to cut segment %d (%.1f–%.1f): %s", idx, start_s, end_s, e)
                    continue

                try:
                    part = self.backend.transcribe(
                        seg_path,
                        language=language,
                        word_timestamps=self.word_timestamps,
                        verbose=False,
                    )
                except Exception as e:
                    logger.warning("Segment %d transcription failed: %s", idx, e)
                    continue

                if part.language and part.language != "unknown":
                    detected_lang = part.language

                text = part.text.strip()
                if text:
                    all_text_parts.append(text)

                # Shift timestamps
                for s in part.segments:
                    shifted = dict(s)
                    shifted["start"] = s.get("start", 0.0) + start_s
                    shifted["end"] = s.get("end", 0.0) + start_s
                    all_segments.append(shifted)

                # Free segment file early
                try:
                    seg_path.unlink(missing_ok=True)
                except OSError:
                    pass

        full_text = " ".join(all_text_parts).strip()
        return BackendResult(
            text=full_text,
            language=detected_lang,
            segments=all_segments,
            raw={"segment_count": len(vad_segments)},
        )

    def unload(self) -> None:
        self.backend.unload()
        import gc
        gc.collect()


def transcribe_file(
    audio_path: str | Path,
    model: str = "auto",
    language: str = "ru",
    use_vad: bool = True,
    backend: str = "mlx-whisper",
    output_dir: Optional[str | Path] = None,
    verbose: bool = False,
    force_segment_wise: Optional[bool] = None,
) -> TranscriptionResult:
    """One-shot helper."""
    t = LocalTranscriber(
        model=model,
        language=language,
        use_vad=use_vad,
        backend=backend,
        verbose=verbose,
    )
    try:
        return t.transcribe(
            audio_path,
            output_dir=output_dir,
            force_segment_wise=force_segment_wise,
        )
    finally:
        t.unload()
