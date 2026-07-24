"""
Backend abstraction for local STT engines.
Supports mlx-whisper (default) and GigaAM-v3 (community MLX port).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BackendResult:
    text: str
    language: str
    segments: List[Dict[str, Any]]
    raw: Optional[Dict[str, Any]] = None


class TranscriptionBackend(ABC):
    """Protocol every STT backend must implement."""

    name: str = "base"

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = "ru",
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> BackendResult:
        ...

    def unload(self) -> None:
        """Optional: free GPU/ANE memory."""
        pass


class MLXWhisperBackend(TranscriptionBackend):
    """Default backend — mlx-whisper optimized for Apple Silicon."""

    name = "mlx-whisper"

    MODEL_MAP = {
        "turbo": "mlx-community/whisper-large-v3-turbo",
        "medium": "mlx-community/whisper-medium",
        "small": "mlx-community/whisper-small",
        "large": "mlx-community/whisper-large-v3",
    }

    def __init__(self, model: str = "turbo"):
        self.model_id = self.MODEL_MAP.get(model, model)
        self._checked = False

    def _ensure_installed(self) -> None:
        if self._checked:
            return
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "mlx-whisper is required.\n"
                "Install: pip install mlx mlx-whisper\n"
                "Works only on macOS Apple Silicon."
            ) from e
        self._checked = True

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = "ru",
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> BackendResult:
        self._ensure_installed()
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_id,
            language=language,
            word_timestamps=word_timestamps,
            verbose=verbose,
        )

        return BackendResult(
            text=result.get("text", "").strip(),
            language=result.get("language", language or "unknown"),
            segments=result.get("segments", []),
            raw=result,
        )


class GigaAMBackend(TranscriptionBackend):
    """
    GigaAM-v3 backend (SOTA for Russian).

    Uses community MLX port:
        https://huggingface.co/al-bo/gigaam-v3-rnnt-mlx

    Expected layout after download:
        model.safetensors (~423 MB fp16)
        config.json

    Installation (one-time):
        pip install mlx safetensors huggingface_hub
        # model downloads automatically on first use
    """

    name = "gigaam"
    HF_REPO = "al-bo/gigaam-v3-rnnt-mlx"

    def __init__(self, model: str = "v3-rnnt", cache_dir: Optional[str | Path] = None):
        self.model_name = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._model_dir: Optional[Path] = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            import mlx.core as mx  # noqa: F401
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "GigaAM MLX backend requires: pip install mlx safetensors huggingface_hub\n"
                f"Original error: {e}"
            ) from e

        # Download (or reuse cache)
        logger.info("Loading GigaAM-v3 MLX from %s ...", self.HF_REPO)
        local_dir = snapshot_download(
            repo_id=self.HF_REPO,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        self._model_dir = Path(local_dir)

        # The community package exposes load_model / load_audio
        # We try several common entry points for robustness.
        self._model = self._load_gigaam_mlx(self._model_dir)
        logger.info("GigaAM model ready (%s)", self._model_dir)

    def _load_gigaam_mlx(self, model_dir: Path):
        """
        Try to load the community MLX conversion.
        Falls back to a clear error if the expected API is missing.
        """
        # Attempt 1: dedicated helper package (if user installed it)
        try:
            from gigaam_mlx import load_model  # type: ignore
            return load_model(str(model_dir))
        except ImportError:
            pass

        # Attempt 2: inline minimal loader (weights + config)
        # This is a skeleton — full RNNT decoding is non-trivial.
        # For now we raise a helpful message so the user knows what is missing.
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.safetensors"

        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"GigaAM MLX files not found in {model_dir}. "
                "Expected config.json + model.safetensors."
            )

        # Real decoding requires the Conformer + RNNT joint/decoder ports.
        # Until a stable pure-MLX inference script is published, we keep this
        # as a clear "not yet runnable" path with all download logic ready.
        raise NotImplementedError(
            "GigaAM-v3 MLX weights are downloaded, but a complete pure-MLX "
            "inference loop (Conformer encoder + RNNT decoder) is not yet "
            "bundled in RAIA.\n\n"
            "Current options:\n"
            "  1. Use backend='mlx-whisper' (stable, recommended for now)\n"
            "  2. Wait for official / community pure-MLX inference script\n"
            "  3. Contribute a gigaam_mlx inference helper\n\n"
            f"Model dir ready at: {model_dir}"
        )

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = "ru",
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> BackendResult:
        self._ensure_model()
        # If we reach here, a real model object exists
        assert self._model is not None

        # Expected community API (adjust when real helper appears)
        if hasattr(self._model, "transcribe"):
            text = self._model.transcribe(str(audio_path))
            if isinstance(text, dict):
                return BackendResult(
                    text=text.get("text", "").strip(),
                    language=language or "ru",
                    segments=text.get("segments", []),
                    raw=text,
                )
            return BackendResult(
                text=str(text).strip(),
                language=language or "ru",
                segments=[],
                raw={"text": text},
            )

        raise RuntimeError("Loaded GigaAM model has no .transcribe() method")

    def unload(self) -> None:
        self._model = None
        import gc
        gc.collect()


def create_backend(
    backend_name: str = "mlx-whisper",
    model: str = "turbo",
) -> TranscriptionBackend:
    """Factory."""
    name = backend_name.lower().strip()
    if name in ("mlx-whisper", "whisper", "mlx"):
        return MLXWhisperBackend(model=model)
    if name in ("gigaam", "gigaam-v3", "giga"):
        return GigaAMBackend(model=model)
    raise ValueError(f"Unknown backend: {backend_name}. Use 'mlx-whisper' or 'gigaam'.")
