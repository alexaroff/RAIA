#!/usr/bin/env python3
"""
CLI for RAIA local transcription.

Examples:
  python -m tool.transcription.cli interview.webm
  python -m tool.transcription.cli interview.webm --model auto --backend mlx-whisper -o out/
  python -m tool.transcription.cli interview.webm --force-segment-wise
  RAIA_MODEL=medium python -m tool.transcription.cli interview.webm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import LocalTranscriber


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAIA Tool — fully local transcription (Apple Silicon)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument(
        "--model", "-m",
        default=None,
        choices=["auto", "turbo", "medium", "small", "large"],
        help="Model size. 'auto' picks by available RAM",
    )
    parser.add_argument(
        "--backend", "-b",
        default=None,
        choices=["mlx-whisper", "gigaam"],
        help="STT backend",
    )
    parser.add_argument("--lang", "-l", default=None, help="Language code")
    parser.add_argument("--no-vad", action="store_true", help="Disable Silero VAD")
    parser.add_argument("--no-word-ts", action="store_true", help="Disable word timestamps")
    parser.add_argument(
        "--force-segment-wise",
        action="store_true",
        help="Force VAD segment-wise transcription even for short files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save .txt / .json / .srt",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--keep-wav", action="store_true", help="Keep preprocessed WAV")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.toml",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    audio = Path(args.audio)
    if not audio.exists():
        print(f"✗ File not found: {audio}", file=sys.stderr)
        return 1

    from .config import load_config

    overrides = {
        "model": args.model,
        "backend": args.backend,
        "language": args.lang,
        "use_vad": False if args.no_vad else None,
        "word_timestamps": False if args.no_word_ts else None,
    }
    cfg = load_config(config_path=args.config, overrides=overrides)

    print(f"→ Transcribing: {audio.name}")
    print(
        f"  model={cfg.model}  backend={cfg.backend}  "
        f"lang={cfg.language}  vad={cfg.use_vad}"
    )

    t = LocalTranscriber(config=cfg, verbose=args.verbose)

    try:
        result = t.transcribe(
            audio,
            output_dir=args.output_dir,
            keep_preprocessed=args.keep_wav,
            force_segment_wise=True if args.force_segment_wise else None,
        )

        print("\n=== Transcription ===\n")
        print(result.text)
        print("\n---")
        print(f"Language      : {result.language}")
        print(f"Duration      : {result.duration_s:.1f}s")
        print(
            f"Processing    : {result.processing_time_s:.1f}s  "
            f"(≈{result.duration_s / max(result.processing_time_s, 0.01):.1f}x realtime)"
        )
        print(f"Model         : {result.model}")
        print(f"Backend       : {result.backend}")
        print(f"Segment-wise  : {result.segment_wise}")
        if result.vad_segments is not None:
            print(f"VAD segments  : {len(result.vad_segments)}")
        if args.output_dir:
            print(f"Saved to      : {args.output_dir}/")
        return 0

    except Exception as e:
        logging.exception("Transcription failed")
        print(f"✗ Error: {e}", file=sys.stderr)
        return 2
    finally:
        t.unload()


if __name__ == "__main__":
    sys.exit(main())
