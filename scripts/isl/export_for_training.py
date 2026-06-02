"""
Export approved ISL submissions from the collection database into the
exact format the ISL training pipeline expects.

Reads:
    /workspace/_state/isl_data.db                (SQLite database)
    /workspace/submissions/upload_*.mp4          (raw uploaded videos)

Writes:
    /workspace/data/isl/raw_videos/clip_NNNN.mp4 (renamed sequentially)
    /workspace/data/isl/annotations.csv          (training-ready CSV)

After running this, you can directly follow ISL_TRAINING_GUIDE.md Phase 4:
    python scripts/isl/mp4_to_frames.py \\
        --input_dir /workspace/data/isl/raw_videos \\
        --output_dir /workspace/data/isl/frames
    python scripts/isl/build_isl_csvs.py \\
        --annotations /workspace/data/isl/annotations.csv \\
        --output_dir /workspace/Sign2GPT/data/isl
    python scripts/pseudo_gloss_en.py \\
        --csv_dir /workspace/Sign2GPT/data/isl \\
        --output_pkl /workspace/Sign2GPT/data/isl/processed_words.isl_pkl
    python scripts/phoenix2014t/image_lmdb_creator.py \\
        --frames_root /workspace/data/isl/frames \\
        --lmdb_root /workspace/lmdb/isl/lmdb_videos \\
        --csv_dir /workspace/Sign2GPT/data/isl \\
        --all_splits

Usage:
    # Default - reads approved submissions, writes training-ready data
    python scripts/isl/export_for_training.py

    # Custom paths
    python scripts/isl/export_for_training.py \\
        --db /workspace/_state/isl_data.db \\
        --videos_in /workspace/submissions \\
        --output_dir /workspace/data/isl

    # Dry run - show what would be exported without copying anything
    python scripts/isl/export_for_training.py --dry_run

    # Include pending too (not just approved) - useful for early development
    python scripts/isl/export_for_training.py --status pending --status approved
"""

import argparse
import csv
import shutil
import sqlite3
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/workspace/_state/isl_data.db",
                        help="SQLite database path (default: /workspace/_state/isl_data.db)")
    parser.add_argument("--videos_in", default="/workspace",
                        help="Root containing the submissions/ folder (default: /workspace)")
    parser.add_argument("--output_dir", default="/workspace/data/isl",
                        help="Where to write raw_videos/ and annotations.csv (default: /workspace/data/isl)")
    parser.add_argument("--status", action="append", default=None,
                        help="Which statuses to include (default: approved only). Can be passed multiple times.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be done without actually copying files")
    args = parser.parse_args()

    statuses = args.status or ["approved"]

    db_path = Path(args.db)
    videos_in = Path(args.videos_in)
    output_dir = Path(args.output_dir)

    if not db_path.is_file():
        print(f"[err] Database not found: {db_path}", file=sys.stderr)
        print(f"      Has anyone uploaded a video yet?", file=sys.stderr)
        sys.exit(1)

    # Read approved submissions from the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in statuses)
    cur.execute(f"""
        SELECT id, s3_key, english_text, signer_name, email, status, submitted_at
        FROM submissions
        WHERE status IN ({placeholders})
        ORDER BY submitted_at ASC
    """, statuses)
    submissions = [dict(r) for r in cur.fetchall()]
    conn.close()

    print(f"[export] Found {len(submissions)} submissions with status in {statuses}")

    if not submissions:
        print(f"[export] Nothing to export. Approve some submissions in /admin first.")
        sys.exit(0)

    # Set up output paths
    raw_videos_dir = output_dir / "raw_videos"
    annotations_csv = output_dir / "annotations.csv"

    if args.dry_run:
        print(f"[export] DRY RUN — would create:")
        print(f"  {raw_videos_dir}/")
        print(f"  {annotations_csv}")
    else:
        raw_videos_dir.mkdir(parents=True, exist_ok=True)

    # Map each submission to clip_NNNN.mp4 and a split
    rows = []
    n_ok = 0
    n_missing = 0
    for i, sub in enumerate(submissions, start=1):
        clip_id = f"clip_{i:04d}"

        # Split rule (matches the admin export logic):
        #   NN0 -> test, NN1 -> dev, else train
        last_digit = i % 10
        if last_digit == 0:
            split = "test"
        elif last_digit == 1:
            split = "dev"
        else:
            split = "train"

        # Source file (s3_key is "submissions/upload_*.mp4")
        src_path = videos_in / sub["s3_key"]
        dst_path = raw_videos_dir / f"{clip_id}.mp4"

        if not src_path.is_file():
            print(f"[warn] Missing video file: {src_path} (submission #{sub['id']})", file=sys.stderr)
            n_missing += 1
            continue

        # Clean signer name (remove commas/special chars for CSV safety)
        signer = (sub["signer_name"] or "anon")
        signer = "".join(c for c in signer if c.isalnum() or c in "_-") or "anon"

        # Clean text (lowercase, no commas)
        text = sub["english_text"].lower().replace(",", " ").strip()

        rows.append({
            "clip_id": clip_id,
            "split": split,
            "english_text": text,
            "signer_id": signer,
        })

        if args.dry_run:
            print(f"  [dry] {src_path.name} -> {dst_path.name} ({split}) - '{text}'")
        else:
            shutil.copy2(src_path, dst_path)
            n_ok += 1

    # Write the annotations.csv
    if args.dry_run:
        print(f"[export] Would write {len(rows)} rows to {annotations_csv}")
    else:
        annotations_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(annotations_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["clip_id", "split", "english_text", "signer_id"])
            writer.writeheader()
            writer.writerows(rows)

    # Summary
    print()
    print("=" * 60)
    print(f"[export] Done:")
    print(f"  Submissions processed: {len(submissions)}")
    print(f"  Videos copied:         {n_ok}")
    print(f"  Missing video files:   {n_missing}")
    print(f"  Output:                {output_dir}")
    print("=" * 60)

    # Split breakdown
    train = sum(1 for r in rows if r["split"] == "train")
    dev = sum(1 for r in rows if r["split"] == "dev")
    test = sum(1 for r in rows if r["split"] == "test")
    print(f"\n[export] Split breakdown:")
    print(f"  train: {train} ({train/max(len(rows),1)*100:.1f}%)")
    print(f"  dev:   {dev} ({dev/max(len(rows),1)*100:.1f}%)")
    print(f"  test:  {test} ({test/max(len(rows),1)*100:.1f}%)")

    if not args.dry_run:
        print(f"\n[export] Ready for next step:")
        print(f"  python scripts/isl/mp4_to_frames.py \\")
        print(f"      --input_dir {raw_videos_dir} \\")
        print(f"      --output_dir {output_dir}/frames")


if __name__ == "__main__":
    main()
