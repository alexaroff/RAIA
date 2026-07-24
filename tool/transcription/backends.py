"""
Backend abstraction for local STT engines.
Makes it easy to add GigaAM, whisper.cpp, Parakeet etc. later
without touching the public API of LocalTranscriber.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    # Short name → full HuggingFace / mlx-community repo
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


# Placeholder for future GigaAM backend
class GigaAMBackend(TranscriptionBackend):
    """
    Future backend for GigaAM-v3 (SOTA Russian).
    Will use community MLX port or official ONNX/PyTorch.
    """

    name = "gigaam"

    def __init__(self, model: str = "v3-rnnt"):
        self.model = model

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = "ru",
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> BackendResult:
        raise NotImplementedError(
            "GigaAM backend is planned for the next iteration. "
            "See RESEARCH.md Experiment №7."
        )
