"""
Convert your ISL annotations.csv into 3 PHOENIX-format split CSVs.

Sign2GPT's dataloader expects CSVs with these columns (pipe-separated):
    name | speaker | orth | translation | start | end

Your annotations.csv only needs:
    clip_id, split, english_text, signer_id

This script:
1. Reads your annotations CSV
2. Splits rows by split column (train/dev/test)
3. Maps your columns to the expected format:
     name = clip_id
     speaker = signer_id
     orth = english_text (we use the English text as both "gloss" and "translation")
     translation = english_text
     start = 0 (placeholder)
     end = -1 (placeholder, dataloader doesn't use this for ISL)
4. Writes ISL.{train,dev,test}.corpus.csv

Usage:
    python scripts/isl/build_isl_csvs.py \\
        --annotations /workspace/data/isl/annotations.csv \\
        --output_dir /workspace/Sign2GPT/data/isl

Output:
    /workspace/Sign2GPT/data/isl/ISL.train.corpus.csv
    /workspace/Sign2GPT/data/isl/ISL.dev.corpus.csv
    /workspace/Sign2GPT/data/isl/ISL.test.corpus.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["clip_id", "split", "english_text", "signer_id"]
EXPECTED_SPLITS = {"train", "dev", "test"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Path to your annotations.csv")
    parser.add_argument("--output_dir", required=True, help="Where to write the 3 split CSVs")
    args = parser.parse_args()

    annot_path = Path(args.annotations)
    out_dir = Path(args.output_dir)

    if not annot_path.is_file():
        print(f"[err] Annotations file not found: {annot_path}", file=sys.stderr)
        sys.exit(1)

    # Read input
    df = pd.read_csv(annot_path)
    print(f"[build_csvs] Read {len(df)} rows from {annot_path}")

    # Validate columns
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        print(f"[err] Missing required columns in annotations.csv: {missing}", file=sys.stderr)
        print(f"      Required: {REQUIRED_COLS}", file=sys.stderr)
        print(f"      Found:    {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Validate splits
    bad_splits = set(df["split"].unique()) - EXPECTED_SPLITS
    if bad_splits:
        print(f"[err] Unknown split values: {bad_splits}", file=sys.stderr)
        print(f"      Expected one of: {EXPECTED_SPLITS}", file=sys.stderr)
        sys.exit(1)

    # Clean text — lowercase, strip whitespace, remove most punctuation
    def clean_text(s):
        s = str(s).lower().strip()
        # Remove punctuation that confuses tokenization
        for ch in ".,!?;:\"'":
            s = s.replace(ch, "")
        # Collapse multiple spaces
        s = " ".join(s.split())
        return s

    df["english_text"] = df["english_text"].apply(clean_text)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write one CSV per split
    for split in ["train", "dev", "test"]:
        split_df = df[df["split"] == split].copy()
        if len(split_df) == 0:
            print(f"[warn] No rows for split={split}", file=sys.stderr)
            continue

        # Build PHOENIX-style columns
        out_df = pd.DataFrame({
            "name": split_df["clip_id"],
            "speaker": split_df["signer_id"],
            "orth": split_df["english_text"],          # use English text as gloss
            "translation": split_df["english_text"],
            "start": 0,
            "end": -1,
        })

        out_path = out_dir / f"ISL.{split}.corpus.csv"
        out_df.to_csv(out_path, sep="|", index=False)
        print(f"[build_csvs] Wrote {len(out_df)} rows -> {out_path}")

        # Also symlink to PHOENIX naming so the LMDB creator and existing
        # dataloader code work without modification
        phx_path = out_dir / f"PHOENIX-2014-T.{split}.corpus.csv"
        try:
            if phx_path.exists() or phx_path.is_symlink():
                phx_path.unlink()
            phx_path.symlink_to(out_path.name)
            print(f"[build_csvs] Symlink: {phx_path.name} -> {out_path.name}")
        except OSError as e:
            # Windows or filesystem doesn't support symlinks - copy instead
            import shutil
            shutil.copy(out_path, phx_path)
            print(f"[build_csvs] Copied (no symlinks): {phx_path.name}")

    # Summary stats
    print()
    print("[build_csvs] Summary:")
    print(df["split"].value_counts().to_string())
    avg_len = df["english_text"].str.split().str.len().mean()
    print(f"\n[build_csvs] Average sentence length: {avg_len:.1f} words")
    n_unique = len(set(" ".join(df["english_text"]).split()))
    print(f"[build_csvs] Unique words across dataset: {n_unique}")


if __name__ == "__main__":
    main()
