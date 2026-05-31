"""
Convert a directory of MP4 videos into PHOENIX-format PNG frame sequences.

Sign2GPT's dataloader expects per-clip folders of PNG frames, not raw MP4s.
This script does the conversion:

    raw_videos/
    ├── clip_0001.mp4
    ├── clip_0002.mp4
    └── ...

becomes:

    frames/
    ├── clip_0001/
    │   ├── images0001.png
    │   ├── images0002.png
    │   └── ... (one PNG per sampled frame at target fps)
    ├── clip_0002/
    └── ...

Usage:
    python scripts/isl/mp4_to_frames.py \\
        --input_dir /workspace/data/isl/raw_videos \\
        --output_dir /workspace/data/isl/frames \\
        --target_fps 25 \\
        --resize 256

Defaults match PHOENIX (25 fps, 256x256 frames) so the existing LMDB creator
and training pipeline work without modification.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def convert_one_video(video_path: Path, output_dir: Path, target_fps: int, resize: int):
    """Read an MP4 and save its frames as PNGs in output_dir."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"error:could_not_open:{video_path.name}"

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Read all frames first
    raw_frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        raw_frames.append(bgr)
    cap.release()

    if not raw_frames:
        return f"error:empty_video:{video_path.name}"

    # Subsample to target fps
    if src_fps > target_fps + 1:
        step = src_fps / target_fps
        indices = np.arange(0, len(raw_frames), step).astype(int)
        raw_frames = [raw_frames[i] for i in indices if i < len(raw_frames)]

    # Write each frame as PNG (PHOENIX naming: images0001.png, images0002.png, ...)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, bgr in enumerate(raw_frames, start=1):
        # Resize (square crop NOT done — preserves aspect)
        h, w = bgr.shape[:2]
        if (w, h) != (resize, resize):
            bgr = cv2.resize(bgr, (resize, resize), interpolation=cv2.INTER_LINEAR)
        png_path = output_dir / f"images{i:04d}.png"
        cv2.imwrite(str(png_path), bgr)

    return f"ok:{video_path.name}:{len(raw_frames)}_frames"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing MP4 (or MOV/AVI/...) files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory; one subdirectory per video will be created here")
    parser.add_argument("--target_fps", type=int, default=25,
                        help="Resample input to this fps (default: 25, matches PHOENIX)")
    parser.add_argument("--resize", type=int, default=256,
                        help="Output frame resolution (default: 256, matches PHOENIX LMDB)")
    parser.add_argument("--extensions", nargs="+", default=[".mp4", ".mov", ".avi", ".mkv"],
                        help="Video file extensions to process")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.is_dir():
        print(f"[err] Input dir does not exist: {in_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all videos
    video_files = []
    for ext in args.extensions:
        video_files.extend(in_dir.glob(f"*{ext}"))
        video_files.extend(in_dir.glob(f"*{ext.upper()}"))
    video_files = sorted(set(video_files))

    if not video_files:
        print(f"[err] No videos found in {in_dir} with extensions {args.extensions}", file=sys.stderr)
        sys.exit(1)

    print(f"[mp4_to_frames] Found {len(video_files)} videos in {in_dir}")
    print(f"[mp4_to_frames] Output: {out_dir}")
    print(f"[mp4_to_frames] target_fps={args.target_fps}, resize={args.resize}x{args.resize}")
    print()

    n_ok = n_skip = n_err = 0
    for video_path in tqdm(video_files, desc="Converting"):
        clip_id = video_path.stem  # filename without extension
        clip_out_dir = out_dir / clip_id

        # Skip if already done
        if clip_out_dir.is_dir() and any(clip_out_dir.glob("images*.png")):
            n_skip += 1
            continue

        result = convert_one_video(video_path, clip_out_dir, args.target_fps, args.resize)
        if result.startswith("ok"):
            n_ok += 1
        else:
            n_err += 1
            print(f"\n[warn] {result}", file=sys.stderr)

    print()
    print(f"[mp4_to_frames] Done: {n_ok} converted, {n_skip} skipped (already done), {n_err} errors")
    print(f"[mp4_to_frames] Output: {out_dir} ({n_ok + n_skip} clips total)")


if __name__ == "__main__":
    main()
