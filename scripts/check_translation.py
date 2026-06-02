"""
Check whether a model's German prediction matches the ground truth -
in English, so non-German speakers can judge.

Takes a clip name (or random), runs inference if needed, then:
  1. Shows German ground truth + German prediction (side by side)
  2. Translates BOTH to English using Google Translate (free, no API key)
  3. Computes word-overlap score in the original German
  4. Gives a verdict: ✓ Good / ⚠️ Partial / ❌ Wrong

Usage:
    # Random test clip with English translation
    python scripts/check_translation.py --split test --random

    # Specific clip
    python scripts/check_translation.py --split test --clip 01April_2010_Thursday_heute-6694

    # Multiple random
    python scripts/check_translation.py --split test --random --num_samples 5

REQUIRES (one-time install):
    pip install deep-translator
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


def translate_to_english(german_text: str) -> str:
    """Translate German -> English using deep_translator (Google free tier)."""
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="de", target="en").translate(german_text)
        return result or "(translation failed)"
    except ImportError:
        return "(install deep-translator: pip install deep-translator)"
    except Exception as e:
        return f"(translation error: {e})"


def get_clip_groundtruth(clip_name: str, split: str) -> str:
    csv_path = f"{CSV_DIR}/PHOENIX-2014-T.{split}.corpus.csv"
    df = pd.read_csv(csv_path, sep="|")
    row = df[df["name"] == clip_name]
    return row.iloc[0]["translation"] if len(row) > 0 else "(not found)"


def list_clips(split: str):
    frames_dir = Path(f"{FRAMES_ROOT}/{split}")
    return sorted([p.name for p in frames_dir.iterdir() if p.is_dir()])


def build_mp4(clip_name: str, split: str, output_mp4: Path) -> bool:
    if output_mp4.exists():
        return True
    png_dir = Path(f"{FRAMES_ROOT}/{split}/{clip_name}")
    if not png_dir.is_dir():
        return False
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", "25", "-pattern_type", "glob",
           "-i", str(png_dir / "images*.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error",
           str(output_mp4)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def run_inference(mp4_path: Path) -> str:
    result = subprocess.run(
        ["python", INFER_SCRIPT, "--video", str(mp4_path)],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("TRANSLATION:"):
            return line.replace("TRANSLATION:", "").strip()
    return "(inference failed)"


def word_overlap(pred: str, gt: str) -> float:
    """Simple word-overlap score: fraction of GT words appearing in PRED."""
    pred_words = set(pred.lower().split())
    gt_words = set(gt.lower().split())
    if not gt_words:
        return 0.0
    return len(pred_words & gt_words) / len(gt_words)


def matching_words(pred: str, gt: str):
    """Find words that match between PRED and GT."""
    pred_set = set(pred.lower().split())
    gt_set = set(gt.lower().split())
    return sorted(pred_set & gt_set)


def verdict(overlap: float) -> str:
    if overlap >= 0.6:
        return "✓ GOOD — most key words match"
    elif overlap >= 0.3:
        return "⚠ PARTIAL — some words match, content differs"
    elif overlap >= 0.1:
        return "△ WEAK — few words match, mostly wrong"
    else:
        return "✗ POOR — no real overlap"


def test_one(clip_name: str, split: str):
    print()
    print("═" * 78)
    print(f"  CLIP: {clip_name}  ({split} split)")
    print("═" * 78)

    mp4_path = Path(DEMO_DIR) / f"{clip_name}.mp4"
    if not mp4_path.exists():
        print("[demo] Building MP4 from PNG frames...")
        if not build_mp4(clip_name, split, mp4_path):
            print("[err] Could not build MP4")
            return

    # Get ground truth German
    gt_de = get_clip_groundtruth(clip_name, split)
    if gt_de == "(not found)":
        print("[err] Clip not in CSV")
        return

    # Run inference
    print("[demo] Running inference (~15-20 sec)...")
    pred_de = run_inference(mp4_path)
    if pred_de == "(inference failed)":
        print("[err] Inference failed")
        return

    # Translate both to English
    print("[demo] Translating to English (Google Translate)...")
    gt_en = translate_to_english(gt_de)
    pred_en = translate_to_english(pred_de)

    # Word overlap analysis
    overlap = word_overlap(pred_de, gt_de)
    matched = matching_words(pred_de, gt_de)

    # Print results
    print()
    print("┌─ GERMAN (original)" + "─" * 58 + "┐")
    print(f"│ Ground truth: {gt_de}")
    print(f"│ Prediction:   {pred_de}")
    print("├─ ENGLISH (auto-translated)" + "─" * 50 + "┤")
    print(f"│ Ground truth: {gt_en}")
    print(f"│ Prediction:   {pred_en}")
    print("├─ ANALYSIS" + "─" * 67 + "┤")
    print(f"│ Word overlap: {overlap*100:.1f}% ({len(matched)} words match)")
    print(f"│ Matching:     {', '.join(matched) if matched else '(none)'}")
    print(f"│ Verdict:      {verdict(overlap)}")
    print("└" + "─" * 77 + "┘")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--clip", help="Specific clip name")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        print("[err] ffmpeg not installed. Run: apt-get install -y ffmpeg", file=sys.stderr)
        sys.exit(1)

    # Check deep-translator
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("[err] deep-translator not installed.", file=sys.stderr)
        print("       Install with: pip install deep-translator", file=sys.stderr)
        sys.exit(1)

    # Pick clips
    if args.clip:
        clips = [args.clip]
    elif args.random:
        all_clips = list_clips(args.split)
        clips = random.sample(all_clips, min(args.num_samples, len(all_clips)))
        print(f"[demo] Randomly selected {len(clips)} clip(s) from {args.split} split")
    else:
        parser.error("Use --clip <name> or --random")

    # Test each
    summary = []
    for clip_name in clips:
        test_one(clip_name, args.split)

    # Aggregate summary if multiple clips
    if len(clips) > 1:
        print("[demo] Done. Inspect each clip above and pick the best for your demo.")


if __name__ == "__main__":
    main()
