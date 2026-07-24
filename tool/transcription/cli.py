#!/usr/bin/env python3
"""
CLI for RAIA local transcription.

Example:
  python -m tool.transcription.cli interview.webm --model turbo --lang ru -o output/
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
    parser.add_argument("audio", type=str, help="Path to audio file (webm/mp3/wav/flac/m4a...)")
    parser.add_argument(
        "--model",
        "-m",
        default="turbo",
        choices=["turbo", "medium", "small", "large"],
        help="Model size / quality trade-off",
    )
    parser.add_argument("--lang", "-l", default="ru", help="Language code (ru recommended)")
    parser.add_argument("--no-vad", action="store_true", help="Disable Silero VAD")
    parser.add_argument("--no-word-ts", action="store_true", help="Disable word timestamps")
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save .txt / .json / .srt",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--keep-wav", action="store_true", help="Keep preprocessed WAV")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    audio = Path(args.audio)
    if not audio.exists():
        print(f"File not found: {audio}", file=sys.stderr)
        return 1

    t = LocalTranscriber(
        model=args.model,
        language=args.lang if args.lang != "auto" else None,
        use_vad=not args.no_vad,
        word_timestamps=not args.no_word_ts,
        verbose=args.verbose,
    )

    try:
        result = t.transcribe(
            audio,
            output_dir=args.output_dir,
            keep_preprocessed=args.keep_wav,
        )
        print("\n=== Transcription ===\n")
        print(result.text)
        print(f"\n---")
        print(f"Language: {result.language}")
        print(f"Duration: {result.duration_s:.1f}s | Processing: {result.processing_time_s:.1f}s")
        print(f"Model: {result.model}")
        if args.output_dir:
            print(f"Saved to: {args.output_dir}")
        return 0
    except Exception as e:
        logging.exception("Transcription failed")
        print(f"Error: {e}", file=sys.stderr)
        return 2
    finally:
        t.unload()


if __name__ == "__main__":
    sys.exit(main())
