"""
One-command ISL training pipeline.

Reads collected videos from the network volume's submission database,
runs every step (export -> frames -> vocab -> LMDB -> train), and produces
a trained ISL -> English translation model.

Designed to be idempotent — re-running it skips steps already done.

Usage:
    # Train on all approved submissions (default)
    python run_isl_training.py

    # Include pending submissions too (useful for early dev)
    python run_isl_training.py --include-pending

    # Just prepare data, don't actually train
    python run_isl_training.py --no-train

    # Custom epoch counts
    python run_isl_training.py --stage1-epochs 30 --stage2-epochs 30

    # Skip stage 1 (useful if it already finished)
    python run_isl_training.py --skip-stage1

    # Skip stage 2
    python run_isl_training.py --skip-stage2

This script wraps:
    scripts/isl/export_for_training.py   (DB -> raw_videos/ + annotations.csv)
    scripts/isl/mp4_to_frames.py         (MP4 -> PNG sequences)
    scripts/isl/build_isl_csvs.py        (annotations.csv -> 3 split CSVs)
    scripts/pseudo_gloss_en.py           (build English vocab pkl)
    scripts/phoenix2014t/image_lmdb_creator.py  (PNG -> LMDB)
    main.py + configs/isl/isl_stage1_config.py   (stage 1 training)
    main.py + configs/isl/isl_stage2_config.py   (stage 2 training)
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


# ----- Default paths (all on the RunPod network volume) -----
DEFAULT_DB = "/workspace/_state/isl_data.db"
DEFAULT_SUBMISSIONS_DIR = "/workspace"  # parent of submissions/
DEFAULT_DATA_DIR = "/workspace/data/isl"
DEFAULT_LMDB_DIR = "/workspace/lmdb/isl/lmdb_videos"
DEFAULT_FRAMES_DIR = "/workspace/data/isl/frames"
DEFAULT_RESULTS_DIR = "/workspace/results"
REPO_ROOT = Path(__file__).resolve().parent


def banner(text):
    """Print a visually distinct section header."""
    bar = "═" * 70
    print()
    print(bar)
    print(f"  {text}")
    print(bar)


def run_cmd(cmd, check=True, env=None):
    """Run a shell command, stream output, raise on failure."""
    print(f"[run] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env)
    if check and result.returncode != 0:
        print(f"[err] Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def count_db_submissions(db_path: Path, statuses=("approved",)) -> int:
    """Count submissions in the DB with given statuses."""
    if not db_path.is_file():
        return 0
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in statuses)
    cur.execute(
        f"SELECT COUNT(*) FROM submissions WHERE status IN ({placeholders})",
        statuses,
    )
    n = cur.fetchone()[0]
    conn.close()
    return n


def step_export(args):
    """Export approved submissions from DB to /workspace/data/isl/"""
    banner("Step 1/7 — Export DB to training format")

    statuses = ["pending", "approved"] if args.include_pending else ["approved"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "isl" / "export_for_training.py"),
        "--db", args.db,
        "--videos_in", args.submissions_dir,
        "--output_dir", args.data_dir,
    ]
    for s in statuses:
        cmd.extend(["--status", s])
    run_cmd(cmd)


def step_mp4_to_frames(args):
    """Convert MP4 videos to PNG sequence directories."""
    banner("Step 2/7 — Convert MP4 to PNG frame sequences")

    raw_videos = Path(args.data_dir) / "raw_videos"
    out_frames = Path(args.frames_dir)

    # Quick skip check — if every clip already has frames, skip
    if out_frames.is_dir():
        existing = list(out_frames.iterdir())
        all_clips = list(raw_videos.iterdir())
        if existing and len(existing) >= len(all_clips):
            print(f"[skip] {len(existing)} clip directories already in {out_frames}")
            return

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "isl" / "mp4_to_frames.py"),
        "--input_dir", str(raw_videos),
        "--output_dir", str(out_frames),
        "--target_fps", "25",
        "--resize", "256",
    ]
    run_cmd(cmd)


def step_build_csvs(args):
    """Convert annotations.csv to 3 PHOENIX-format split CSVs."""
    banner("Step 3/7 — Build train/dev/test split CSVs")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "isl" / "build_isl_csvs.py"),
        "--annotations", f"{args.data_dir}/annotations.csv",
        "--output_dir", str(REPO_ROOT / "data" / "isl"),
    ]
    run_cmd(cmd)


def step_install_spacy_en(args):
    """Ensure the English spaCy model is installed."""
    banner("Step 4a/7 — Ensure English spaCy model installed")
    try:
        import spacy
        spacy.load("en_core_web_lg")
        print("[ok] en_core_web_lg already installed")
        return
    except Exception:
        pass

    wheel_url = ("https://github.com/explosion/spacy-models/releases/download/"
                 "en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl")
    run_cmd([sys.executable, "-m", "pip", "install", wheel_url])


def step_pseudo_gloss(args):
    """Build English pseudo-gloss vocabulary pkl."""
    banner("Step 4b/7 — Build English pseudo-gloss vocab")

    out_pkl = Path(REPO_ROOT) / "data" / "isl" / "processed_words.isl_pkl"
    if out_pkl.is_file() and not args.force:
        print(f"[skip] {out_pkl} already exists (use --force to rebuild)")
        return

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "pseudo_gloss_en.py"),
        "--csv_dir", str(REPO_ROOT / "data" / "isl"),
        "--output_pkl", str(out_pkl),
    ]
    run_cmd(cmd)


def step_lmdb(args):
    """Convert PNG frames to LMDB."""
    banner("Step 5/7 — Convert PNG frames to LMDB")

    lmdb_out = Path(args.lmdb_dir)
    if lmdb_out.is_dir() and any(lmdb_out.iterdir()) and not args.force:
        # Quick check — count clip subdirs vs expected
        n_lmdb = len(list(lmdb_out.iterdir()))
        n_csvs = sum(1 for _ in Path(args.data_dir).glob("annotations.csv"))
        if n_lmdb > 0 and n_csvs > 0:
            print(f"[skip] {n_lmdb} LMDB clip directories already exist (use --force to rebuild)")
            return

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "phoenix2014t" / "image_lmdb_creator.py"),
        "--frames_root", args.frames_dir,
        "--lmdb_root", args.lmdb_dir,
        "--csv_dir", str(REPO_ROOT / "data" / "isl"),
        "--all_splits",
    ]
    run_cmd(cmd)


def step_stage1(args):
    """Run ISL stage 1 training."""
    banner(f"Step 6/7 — Stage 1 training ({args.stage1_epochs} epochs)")

    if args.skip_stage1:
        print("[skip] --skip-stage1 passed")
        return

    # Override max_epochs if user specified
    config_path = REPO_ROOT / "configs" / "isl" / "isl_stage1_config.py"
    if args.stage1_epochs is not None:
        _patch_max_epochs(config_path, args.stage1_epochs)

    log_path = Path(args.results_dir) / "isl_stage1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Run main.py with the ISL stage 1 config
    cmd = [
        "bash", "-c",
        f"cd {REPO_ROOT} && {sys.executable} main.py "
        f"--config=configs/isl/isl_stage1_config.py 2>&1 | tee {log_path}"
    ]
    run_cmd(cmd)


def step_stage2(args):
    """Run ISL stage 2 training."""
    banner(f"Step 7/7 — Stage 2 training ({args.stage2_epochs} epochs)")

    if args.skip_stage2:
        print("[skip] --skip-stage2 passed")
        return

    config_path = REPO_ROOT / "configs" / "isl" / "isl_stage2_config.py"
    if args.stage2_epochs is not None:
        _patch_max_epochs(config_path, args.stage2_epochs)

    log_path = Path(args.results_dir) / "isl_stage2.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash", "-c",
        f"cd {REPO_ROOT} && {sys.executable} main.py "
        f"--config=configs/isl/isl_stage2_config.py 2>&1 | tee {log_path}"
    ]
    run_cmd(cmd)


def _patch_max_epochs(config_path: Path, new_value: int):
    """In-place edit of cfg.max_epochs in a config file."""
    text = config_path.read_text()
    # Find the line "cfg.max_epochs = N" and replace
    import re
    new_text = re.sub(
        r"cfg\.max_epochs\s*=\s*\d+",
        f"cfg.max_epochs = {new_value}",
        text,
    )
    if new_text != text:
        config_path.write_text(new_text)
        print(f"[patch] Set cfg.max_epochs = {new_value} in {config_path.name}")


def print_data_size_advice(n):
    """Warn user if the dataset is too small for meaningful results."""
    if n < 10:
        print(f"⚠️  WARNING: Only {n} clips. Training will fail or produce garbage.")
        print("    Recommend at least 50 clips for any useful signal,")
        print("    200+ for a working demo, 500+ for production-ish quality.")
    elif n < 100:
        print(f"⚠️  WARNING: Only {n} clips. Expect very poor BLEU (< 2).")
        print("    Useful only to verify the pipeline trains without crashing.")
    elif n < 300:
        print(f"⚠️  {n} clips. Expect BLEU 2-4. Demo quality, not production.")
    elif n < 1000:
        print(f"✓ {n} clips. Expect BLEU 4-8. Decent for a first demo.")
    else:
        print(f"✓ {n} clips. Expect BLEU 8-15. Real working model.")


def main():
    parser = argparse.ArgumentParser(
        description="One-command ISL training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--submissions_dir", default=DEFAULT_SUBMISSIONS_DIR,
                        help=f"Parent of submissions/ folder (default: {DEFAULT_SUBMISSIONS_DIR})")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help=f"Where to put raw_videos/ and annotations.csv (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--frames_dir", default=DEFAULT_FRAMES_DIR,
                        help=f"Where to put PNG sequences (default: {DEFAULT_FRAMES_DIR})")
    parser.add_argument("--lmdb_dir", default=DEFAULT_LMDB_DIR,
                        help=f"Where to put LMDB files (default: {DEFAULT_LMDB_DIR})")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                        help=f"Where to put training logs (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--include-pending", action="store_true",
                        help="Include pending submissions (default: approved only)")
    parser.add_argument("--no-train", action="store_true",
                        help="Prepare data but don't actually train")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip stage 1 training (assumes already done)")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Skip stage 2 training")
    parser.add_argument("--stage1-epochs", type=int, default=None,
                        help="Override stage 1 max_epochs (default: 60)")
    parser.add_argument("--stage2-epochs", type=int, default=None,
                        help="Override stage 2 max_epochs (default: 60)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run of cached steps")
    args = parser.parse_args()

    db_path = Path(args.db)

    # Step 0 — sanity check
    banner("Step 0/7 — Sanity check")
    if not db_path.is_file():
        print(f"[err] No database found at {db_path}")
        print("       Has anyone uploaded a video to your collector yet?")
        print(f"       Database is created by the web app on first submission.")
        sys.exit(1)

    statuses = ("pending", "approved") if args.include_pending else ("approved",)
    n_submissions = count_db_submissions(db_path, statuses)
    print(f"[ok] Database has {n_submissions} submissions with status in {statuses}")
    print_data_size_advice(n_submissions)

    if n_submissions == 0:
        print("[err] No submissions to train on. Get some videos uploaded first.")
        sys.exit(1)

    # Run all steps
    step_export(args)
    step_mp4_to_frames(args)
    step_build_csvs(args)
    step_install_spacy_en(args)
    step_pseudo_gloss(args)
    step_lmdb(args)

    if args.no_train:
        banner("Done — data prepared, training skipped (--no-train)")
        print(f"\nYou can now train manually with:")
        print(f"  python main.py --config=configs/isl/isl_stage1_config.py")
        print(f"  python main.py --config=configs/isl/isl_stage2_config.py")
        return

    step_stage1(args)
    step_stage2(args)

    # Summary
    banner("🎉 ISL training pipeline complete")
    print(f"\nStage 1 log: {args.results_dir}/isl_stage1.log")
    print(f"Stage 2 log: {args.results_dir}/isl_stage2.log")
    print(f"\nFinal BLEU (from stage 2 log):")
    print(f"  grep 'valid/ableu_bleu4\\|test/ableu_bleu4' {args.results_dir}/isl_stage2.log | tail -5")


if __name__ == "__main__":
    main()
