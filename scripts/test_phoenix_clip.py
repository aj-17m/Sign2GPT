"""
Test the trained model on a specific PHOENIX clip (test/dev split).

This script:
  1. Picks a clip from a PHOENIX split (test/dev) - either specific or random
  2. Builds an MP4 from the clip's PNG frames using ffmpeg
  3. Runs infer_video.py on the MP4
  4. Prints prediction + ground truth side by side

Usage:
    # Random test clip
    python scripts/test_phoenix_clip.py --split test --random

    # Specific test clip
    python scripts/test_phoenix_clip.py --split test --clip 01April_2010_Thursday_heute-6694

    # Multiple random clips at once
    python scripts/test_phoenix_clip.py --split test --random --num_samples 5

    # List available clips in a split
    python scripts/test_phoenix_clip.py --split test --list

    # Use dev split instead of test
    python scripts/test_phoenix_clip.py --split dev --random
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = "/workspace/Sign2GPT"
CSV_DIR = f"{REPO_ROOT}/data/phoenix2014t"
FRAMES_ROOT = "/workspace/data/phoenix2014t/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
DEMO_DIR = "/workspace/demos"
INFER_SCRIPT = f"{REPO_ROOT}/scripts/infer_video.py"


def get_clip_groundtruth(clip_name: str, split: str) -> str:
    """Read the ground-truth translation from the corpus CSV."""
    csv_path = f"{CSV_DIR}/PHOENIX-2014-T.{split}.corpus.csv"
    df = pd.read_csv(csv_path, sep="|")
    row = df[df["name"] == clip_name]
    if len(row) == 0:
        return "(clip not found in CSV)"
    return row.iloc[0]["translation"]


def list_clips(split: str):
    """List all clips in a split that have PNG frames on disk."""
    frames_dir = Path(f"{FRAMES_ROOT}/{split}")
    if not frames_dir.is_dir():
        print(f"[err] Frames dir not found: {frames_dir}", file=sys.stderr)
        sys.exit(1)
    clips = sorted([p.name for p in frames_dir.iterdir() if p.is_dir()])
    return clips


def png_dir_to_mp4(clip_name: str, split: str, output_mp4: Path) -> bool:
    """Convert a clip's PNG sequence to an MP4 file using ffmpeg."""
    png_dir = Path(f"{FRAMES_ROOT}/{split}/{clip_name}")
    if not png_dir.is_dir():
        print(f"[err] PNG dir not found: {png_dir}", file=sys.stderr)
        return False

    # Check it has PNGs
    pngs = sorted(png_dir.glob("images*.png"))
    if not pngs:
        print(f"[err] No images*.png found in {png_dir}", file=sys.stderr)
        return False

    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Run ffmpeg quietly
    cmd = [
        "ffmpeg",
        "-y",                                    # overwrite if exists
        "-framerate", "25",                      # PHOENIX is 25 fps
        "-pattern_type", "glob",
        "-i", str(png_dir / "images*.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",                    # quiet
        str(output_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[err] ffmpeg failed:\n{result.stderr}", file=sys.stderr)
        return False

    print(f"[demo] Built MP4: {output_mp4} ({len(pngs)} frames)")
    return True


def run_inference(mp4_path: Path) -> str:
    """Run scripts/infer_video.py on the MP4 and capture its translation."""
    cmd = ["python", INFER_SCRIPT, "--video", str(mp4_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[err] infer_video.py failed:\n{result.stderr}", file=sys.stderr)
        return ""

    # Parse the "TRANSLATION:" line from the output
    pred = ""
    for line in result.stdout.splitlines():
        if line.startswith("TRANSLATION:"):
            pred = line.replace("TRANSLATION:", "").strip()
            break

    return pred


def test_one_clip(clip_name: str, split: str):
    """Build MP4 + run inference + print results for one clip."""
    print()
    print("=" * 70)
    print(f"[demo] Testing clip: {clip_name} ({split} split)")
    print("=" * 70)

    mp4_path = Path(DEMO_DIR) / f"{clip_name}.mp4"

    # Build MP4 if not exists
    if not mp4_path.exists():
        if not png_dir_to_mp4(clip_name, split, mp4_path):
            return
    else:
        print(f"[demo] MP4 already exists: {mp4_path}")

    # Get ground truth
    gt = get_clip_groundtruth(clip_name, split)

    # Run inference (model loads ~10 sec, inference ~5 sec)
    print(f"[demo] Running inference (this takes ~15-20 sec)...")
    pred = run_inference(mp4_path)

    # Print side by side
    print()
    print("┌" + "─" * 68 + "┐")
    print(f"│ Clip:         {clip_name:<53} │")
    print(f"│ Split:        {split:<53} │")
    print("├" + "─" * 68 + "┤")
    print(f"│ GROUND TRUTH: {gt[:53]:<53} │")
    if len(gt) > 53:
        print(f"│               {gt[53:106]:<53} │")
    print("├" + "─" * 68 + "┤")
    print(f"│ MODEL OUTPUT: {pred[:53]:<53} │")
    if len(pred) > 53:
        print(f"│               {pred[53:106]:<53} │")
    print("└" + "─" * 68 + "┘")
    print(f"[demo] MP4 saved at: {mp4_path}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"],
                        help="Which PHOENIX split to test (default: test - never seen by model)")
    parser.add_argument("--clip", help="Specific clip name to test")
    parser.add_argument("--random", action="store_true", help="Pick random clip(s) from the split")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of random samples (default: 1)")
    parser.add_argument("--list", action="store_true", help="Just list available clips and exit")
    args = parser.parse_args()

    # List mode
    if args.list:
        clips = list_clips(args.split)
        print(f"[demo] {len(clips)} clips available in {args.split} split:")
        for c in clips[:30]:
            print(f"  {c}")
        if len(clips) > 30:
            print(f"  ... and {len(clips) - 30} more")
        return

    # Check ffmpeg is available
    if not shutil.which("ffmpeg"):
        print("[err] ffmpeg not found. Install with: apt-get install -y ffmpeg", file=sys.stderr)
        sys.exit(1)

    # Resolve which clip(s) to test
    if args.clip:
        clips_to_test = [args.clip]
    elif args.random:
        all_clips = list_clips(args.split)
        clips_to_test = random.sample(all_clips, min(args.num_samples, len(all_clips)))
        print(f"[demo] Randomly selected {len(clips_to_test)} clip(s) from {args.split} split")
    else:
        parser.error("Must pass --clip <name>, --random, or --list")

    # Test each clip
    for clip_name in clips_to_test:
        test_one_clip(clip_name, args.split)


if __name__ == "__main__":
    main()
