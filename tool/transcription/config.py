"""
Configuration for RAIA transcription pipeline.
Supports:
  - tool/config.toml (or RAIA_CONFIG env)
  - Environment variables (RAIA_*)
  - Auto model selection by available RAM (psutil)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Defaults
DEFAULTS = {
    "model": "auto",          # "auto" | "turbo" | "medium" | "small" | "large"
    "backend": "mlx-whisper", # "mlx-whisper" | "gigaam"
    "language": "ru",
    "use_vad": True,
    "word_timestamps": True,
    "vad_min_duration_s": 20.0,   # below this → full-file transcription
    "segment_wise_threshold_s": 90.0,  # above this + VAD → segment-wise
    "max_segment_s": 30.0,        # split very long VAD segments further
}


@dataclass
class TranscribeConfig:
    model: str = "auto"
    backend: str = "mlx-whisper"
    language: str = "ru"
    use_vad: bool = True
    word_timestamps: bool = True
    vad_min_duration_s: float = 20.0
    segment_wise_threshold_s: float = 90.0
    max_segment_s: float = 30.0

    # Resolved after auto-selection
    resolved_model: str = field(default="turbo", init=False)

    def resolve_model(self) -> str:
        """Pick concrete model name. 'auto' uses available RAM."""
        if self.model != "auto":
            self.resolved_model = self.model
            return self.resolved_model

        try:
            import psutil
            # Available RAM in GB (prefer available over total on unified memory)
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            # On Apple Silicon unified memory is shared; be conservative
            usable = min(available_gb, total_gb * 0.55)

            if usable >= 10:
                choice = "large"
            elif usable >= 5.5:
                choice = "turbo"
            elif usable >= 3.5:
                choice = "medium"
            else:
                choice = "small"

            logger.info(
                "Auto model: available≈%.1f GB → %s (total=%.1f GB)",
                available_gb, choice, total_gb,
            )
            self.resolved_model = choice
            return choice
        except ImportError:
            logger.warning("psutil not installed → defaulting to 'turbo'")
            self.resolved_model = "turbo"
            return "turbo"
        except Exception as e:
            logger.warning("RAM probe failed (%s) → 'turbo'", e)
            self.resolved_model = "turbo"
            return "turbo"


def _load_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return {}

    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    # Support [transcription] section or flat
    return data.get("transcription", data)


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TranscribeConfig:
    """
    Load config with priority:
      1. Explicit overrides (CLI / function kwargs)
      2. Environment variables RAIA_*
      3. config.toml
      4. Hardcoded defaults
    """
    cfg = dict(DEFAULTS)

    # 3. TOML
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    env_path = os.environ.get("RAIA_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    # Look next to this package and in cwd
    pkg_root = Path(__file__).resolve().parent.parent  # tool/
    candidates.extend([
        pkg_root / "config.toml",
        Path.cwd() / "config.toml",
        Path.cwd() / "tool" / "config.toml",
    ])

    for p in candidates:
        data = _load_toml(p)
        if data:
            logger.debug("Loaded config from %s", p)
            cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
            break

    # 2. Environment
    env_map = {
        "RAIA_MODEL": "model",
        "RAIA_BACKEND": "backend",
        "RAIA_LANGUAGE": "language",
        "RAIA_USE_VAD": "use_vad",
        "RAIA_WORD_TIMESTAMPS": "word_timestamps",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        if cfg_key in ("use_vad", "word_timestamps"):
            cfg[cfg_key] = val.lower() in ("1", "true", "yes", "on")
        else:
            cfg[cfg_key] = val

    # 1. Overrides
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None and k in DEFAULTS})

    return TranscribeConfig(**{k: cfg[k] for k in DEFAULTS})
