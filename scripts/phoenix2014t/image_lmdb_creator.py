"""
PHOENIX-2014-T frames -> LMDB converter.

The upstream repo only ships an LMDB creator for CSL-Daily (which expects
one .mp4 per clip). PHOENIX-2014-T stores each clip as a directory of
PNG frames, so we need a slightly different script that:

  1. Walks each clip directory.
  2. Reads frames in sorted order (lexicographic on filename).
  3. Resizes to 256x256 (matching CSL-Daily preprocessing).
  4. Writes them into one LMDB per clip, keyed by frame index.
  5. Adds a "details" key with {"num_frames": N, "id": clip_name}.

The dataloader (`dataloaders/phoenix_video_dataset.py`) opens each LMDB
via `LMDBUtility(f"{lmdb_dir}/{file_name}")`.

Usage (single clip - useful for debugging):
    python scripts/phoenix2014t/image_lmdb_creator.py \\
        --frames_root /workspace/data/phoenix2014t/PHOENIX-2014-T/features/fullFrame-210x260px \\
        --lmdb_root  /workspace/lmdb/phoenix2014t/lmdb_videos \\
        --split train \\
        --name 01April_2010_Thursday_heute_default-0

Usage (whole dataset - what you actually run):
    python scripts/phoenix2014t/image_lmdb_creator.py \\
        --frames_root /workspace/data/phoenix2014t/PHOENIX-2014-T/features/fullFrame-210x260px \\
        --lmdb_root  /workspace/lmdb/phoenix2014t/lmdb_videos \\
        --csv_dir    /workspace/Sign2GPT/data/phoenix2014t \\
        --all_splits
"""

from argparse import ArgumentParser
from pathlib import Path
import shutil
import io
import pickle
import sys
import time
import tempfile

import pandas as pd
from PIL import Image
import lmdb
from tqdm import tqdm


N_BYTES = 2**38  # 256 GB max-map (LMDB only allocates what it needs).
RESIZE = (256, 256)
COMMIT_EVERY = 100  # Commit a write txn every N frames to keep memory flat.


def list_clip_frames(clip_dir: Path):
    """Return frames as a sorted list of file paths.

    PHOENIX-2014-T frames are named like ``images0001.png``; lexicographic
    sort happens to be numerically correct because of the zero-padding,
    but we also handle the case where the filenames are bare integers.
    """
    files = [p for p in clip_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    files.sort(key=lambda p: p.name)
    return files


def convert_clip(clip_dir: Path, lmdb_out_dir: Path, clip_id: str, overwrite: bool = False):
    """Convert one PHOENIX clip directory into a single LMDB."""

    if lmdb_out_dir.exists() and lmdb_out_dir.is_dir() and not overwrite:
        # Idempotent: skip if already converted. This means re-running the
        # whole-dataset command is safe and cheap after a crash.
        return "skipped"

    if lmdb_out_dir.exists():
        shutil.rmtree(lmdb_out_dir)

    frames = list_clip_frames(clip_dir)
    if not frames:
        return f"empty:{clip_id}"

    # Write to a tmp dir then atomically move into place, so a crashed run
    # never leaves a half-written LMDB behind.
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"lmdb_{clip_id}_"))
    env = lmdb.open(path=str(tmp_dir), map_size=N_BYTES)
    txn = env.begin(write=True)

    ind = 0
    for frame_path in frames:
        try:
            img = Image.open(frame_path).convert("RGB").resize(RESIZE)
        except Exception as e:
            env.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return f"error:{clip_id}:{e}"

        buf = io.BytesIO()
        img.save(buf, format="jpeg", quality=90)
        buf.seek(0)
        txn.put(key=f"{ind}".encode("ascii"), value=buf.read(), dupdata=False)
        ind += 1

        if ind % COMMIT_EVERY == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.put(
        key=b"details",
        value=pickle.dumps({"num_frames": ind, "id": clip_id}, protocol=4),
        dupdata=False,
    )
    txn.commit()
    env.close()

    lmdb_out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_dir), str(lmdb_out_dir))
    return f"ok:{ind}"


def convert_split(frames_root: Path, lmdb_root: Path, split: str, csv_dir: Path, overwrite: bool):
    """Convert every clip listed in the PHOENIX corpus CSV for one split."""
    csv_path = csv_dir / f"PHOENIX-2014-T.{split}.corpus.csv"
    if not csv_path.exists():
        print(f"[!] CSV not found: {csv_path}", file=sys.stderr)
        return

    df = pd.read_csv(csv_path, sep="|")
    print(f"[{split}] {len(df)} clips listed in {csv_path.name}")

    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"LMDB[{split}]"):
        clip_id = row["name"]
        clip_dir = frames_root / split / clip_id
        if not clip_dir.is_dir():
            # PHOENIX sometimes stores a "1/" subdirectory inside the clip
            # folder (camera-1 frames). Handle that fallback.
            alt = frames_root / split / clip_id / "1"
            if alt.is_dir():
                clip_dir = alt
            else:
                print(f"[!] missing frames for {clip_id} (looked in {clip_dir})", file=sys.stderr)
                n_err += 1
                continue

        out = lmdb_root / clip_id
        result = convert_clip(clip_dir, out, clip_id, overwrite=overwrite)
        if result.startswith("ok"):
            n_ok += 1
        elif result == "skipped":
            n_skip += 1
        else:
            n_err += 1
            print(f"[!] {result}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"[{split}] done: ok={n_ok} skipped={n_skip} errors={n_err} in {elapsed/60:.1f} min")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--frames_root",
        required=True,
        help="Path to PHOENIX-2014-T/features/fullFrame-210x260px (parent of train/dev/test)",
    )
    parser.add_argument(
        "--lmdb_root",
        required=True,
        help="Output LMDB directory; one sub-LMDB per clip will be created here",
    )
    parser.add_argument(
        "--csv_dir",
        default=None,
        help="Directory containing PHOENIX-2014-T.{train,dev,test}.corpus.csv. "
        "Required unless --name is given.",
    )
    parser.add_argument("--split", default=None, choices=["train", "dev", "test"])
    parser.add_argument("--name", default=None, help="Single clip name (debug mode)")
    parser.add_argument("--all_splits", action="store_true", help="Process train + dev + test")
    parser.add_argument("--overwrite", action="store_true", help="Re-create LMDBs that already exist")
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    lmdb_root = Path(args.lmdb_root)
    lmdb_root.mkdir(parents=True, exist_ok=True)

    if args.name:
        assert args.split, "--split required when using --name"
        clip_dir = frames_root / args.split / args.name
        result = convert_clip(clip_dir, lmdb_root / args.name, args.name, overwrite=args.overwrite)
        print(f"single-clip result: {result}")
        return

    assert args.csv_dir is not None, "--csv_dir required when not using --name"
    csv_dir = Path(args.csv_dir)

    splits = ["train", "dev", "test"] if args.all_splits else [args.split]
    assert all(s for s in splits), "either --all_splits or --split required"
    for split in splits:
        convert_split(frames_root, lmdb_root, split, csv_dir, args.overwrite)


if __name__ == "__main__":
    main()
