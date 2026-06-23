#!/usr/bin/env python3
"""Convert a video, including HEVC MOV files, to a compatible MP4."""

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a video to an H.264/AAC MP4 using ffmpeg."
    )
    parser.add_argument("input", type=Path, help="Input video, for example video.mov")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output MP4 path (default: INPUT_converted.mp4)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg is required. On macOS, install it with: brew install ffmpeg"
        )
    if not args.input.is_file():
        raise SystemExit(f"Input video does not exist: {args.input}")

    output = args.output or args.input.with_name(f"{args.input.stem}_converted.mp4")
    if output.suffix.lower() != ".mp4":
        raise SystemExit(f"Output path must end with .mp4: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y" if args.overwrite else "-n",
        "-i",
        str(args.input),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"ffmpeg conversion failed with exit code {error.returncode}")

    print(f"Converted video saved to: {output}")


if __name__ == "__main__":
    main()
