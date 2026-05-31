"""
Bulk-rename ISL videos to systematic clip_NNNN.mp4 naming.

Run this after dumping videos from your phone into raw_videos/. It scans
the directory, sorts by file creation time (or filename), and renames
sequentially starting from the next available number.

USAGE:

    # Dry-run (preview only, don't actually rename)
    python scripts/isl/rename_videos.py \\
        --videos_dir D:/Sign2GPT/dataset/isl/raw_videos \\
        --dry_run

    # Actually rename (creates clip_NNNN.mp4 in numerical sequence)
    python scripts/isl/rename_videos.py \\
        --videos_dir D:/Sign2GPT/dataset/isl/raw_videos

    # Start numbering from a specific point (e.g., if you have clip_0001-0050 already)
    python scripts/isl/rename_videos.py \\
        --videos_dir D:/Sign2GPT/dataset/isl/raw_videos \\
        --start_num 51

The script PRESERVES already-correctly-named clips (clip_NNNN.mp4 pattern)
and only renames the rest. So you can safely run it multiple times.
"""

import argparse
import re
import sys
from pathlib import Path


CLIP_PATTERN = re.compile(r"^clip_(\d{4})\.mp4$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", required=True,
                        help="Directory containing your raw MP4 files")
    parser.add_argument("--start_num", type=int, default=None,
                        help="Starting number (default: next available based on existing clip_NNNN.mp4 files)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be renamed without actually renaming")
    parser.add_argument("--sort_by", choices=["mtime", "name"], default="mtime",
                        help="Sort order for unnamed videos (default: mtime = creation time)")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        print(f"[err] Directory not found: {videos_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all MP4 files
    all_mp4s = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.MP4"))

    # Split into already-named (clip_NNNN.mp4) and needs-renaming
    already_named = []
    needs_renaming = []
    for path in all_mp4s:
        if CLIP_PATTERN.match(path.name):
            already_named.append(path)
        else:
            needs_renaming.append(path)

    print(f"[rename] Found {len(all_mp4s)} MP4 files in {videos_dir}")
    print(f"[rename]   Already named: {len(already_named)}")
    print(f"[rename]   Need renaming: {len(needs_renaming)}")

    if not needs_renaming:
        print("[rename] Nothing to do — all files already named correctly.")
        return

    # Determine starting number
    if args.start_num is not None:
        start_num = args.start_num
    else:
        # Next number after highest existing
        if already_named:
            highest = max(int(CLIP_PATTERN.match(p.name).group(1)) for p in already_named)
            start_num = highest + 1
        else:
            start_num = 1

    print(f"[rename] Starting number: {start_num}")
    print()

    # Sort needs_renaming
    if args.sort_by == "mtime":
        needs_renaming.sort(key=lambda p: p.stat().st_mtime)
    else:
        needs_renaming.sort(key=lambda p: p.name)

    # Plan the renames
    planned = []
    current_num = start_num
    for old_path in needs_renaming:
        new_name = f"clip_{current_num:04d}.mp4"
        new_path = videos_dir / new_name
        # Make sure new_name doesn't collide with an existing already-named file
        while new_path.exists():
            current_num += 1
            new_name = f"clip_{current_num:04d}.mp4"
            new_path = videos_dir / new_name
        planned.append((old_path, new_path))
        current_num += 1

    # Show plan
    print("[rename] Plan:")
    for old_path, new_path in planned[:20]:
        print(f"    {old_path.name}  ->  {new_path.name}")
    if len(planned) > 20:
        print(f"    ... and {len(planned) - 20} more")
    print()

    if args.dry_run:
        print("[rename] DRY RUN — no files were renamed. Remove --dry_run to apply.")
        return

    # Confirm before executing (only if more than 50 renames)
    if len(planned) > 50:
        response = input(f"About to rename {len(planned)} files. Continue? (y/N): ")
        if response.lower() != "y":
            print("[rename] Cancelled.")
            return

    # Execute
    n_ok = n_err = 0
    for old_path, new_path in planned:
        try:
            old_path.rename(new_path)
            n_ok += 1
        except Exception as e:
            print(f"[err] Could not rename {old_path.name}: {e}", file=sys.stderr)
            n_err += 1

    print(f"\n[rename] Done: {n_ok} renamed, {n_err} errors")
    print(f"[rename] Highest clip number is now: {start_num + n_ok - 1}")
    print(f"[rename] Next clip you record will be: clip_{start_num + n_ok:04d}.mp4")


if __name__ == "__main__":
    main()
