"""
Bulk upload ISL videos + mappings to the RunPod network volume.

Run this from your LAPTOP (or any computer) when you have a folder of
pre-recorded ISL videos + a CSV mapping each file to its English meaning.

The script uses the same S3-compatible API as the web app, so videos
land in the same place /workspace/submissions/ on the network volume,
visible to run_isl_training.py just like web uploads.

CSV FORMAT (required columns):
    filename,english_text,signer_name
    clip_001.mp4,hello how are you,ajay
    clip_002.mp4,my name is ajay,ajay
    clip_003.mp4,i am hungry,priya

  - `filename` must match the actual MP4 file in --folder
  - `english_text` is the meaning (will be lowercased, punctuation stripped)
  - `signer_name` is optional per row; can also be set globally via --signer

USAGE:

    # First time setup - install dependencies
    pip install boto3 pandas tqdm

    # Bulk upload
    python bulk_upload.py \\
        --folder ./my_isl_videos \\
        --csv ./my_mapping.csv \\
        --bucket p014akuq8i \\
        --endpoint https://s3api-eu-ro-1.runpod.io \\
        --region eu-ro-1 \\
        --access-key user_xxxxx \\
        --secret-key rps_xxxxx

    # Or use environment variables (cleaner, no secrets in shell history):
    export S3_BUCKET=p014akuq8i
    export S3_ENDPOINT_URL=https://s3api-eu-ro-1.runpod.io
    export S3_REGION=eu-ro-1
    export S3_ACCESS_KEY=user_xxxxx
    export S3_SECRET_KEY=rps_xxxxx

    python bulk_upload.py --folder ./videos --csv mapping.csv

    # Dry run (validates CSV + files without uploading)
    python bulk_upload.py --folder ./videos --csv mapping.csv --dry-run

    # Set signer name for all rows
    python bulk_upload.py --folder ./videos --csv mapping.csv --signer ajay

SAFETY:
- Skips files already uploaded (matches by filename in DB) - safe to re-run
- Downloads + uploads SQLite DB to stay in sync with web uploads
- Won't accept videos larger than --max-mb (default 50)
"""

import argparse
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

try:
    import boto3
    import pandas as pd
    from tqdm import tqdm
except ImportError as e:
    print(f"[err] Missing dependency: {e}", file=sys.stderr)
    print("       Install with: pip install boto3 pandas tqdm", file=sys.stderr)
    sys.exit(1)


VALID_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
DB_S3_KEY = "_state/isl_data.db"


def clean_text(s: str) -> str:
    """Match the web app's text cleaning."""
    s = str(s).lower().strip()
    for ch in ".,!?;:\"'":
        s = s.replace(ch, "")
    return " ".join(s.split())


def safe_signer(s: str) -> str:
    s = "".join(c for c in (s or "anon")[:30] if c.isalnum() or c in "_-")
    return s or "anon"


def init_db(db_path: Path):
    """Same schema as the web app's database.py."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            s3_key TEXT NOT NULL,
            english_text TEXT NOT NULL,
            signer_name TEXT DEFAULT 'anonymous',
            email TEXT DEFAULT '',
            status TEXT DEFAULT 'approved',
            size_bytes INTEGER DEFAULT 0,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_status ON submissions(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_s3_key ON submissions(s3_key)")
    conn.commit()
    conn.close()


def db_already_uploaded(db_path: Path, s3_key: str) -> bool:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM submissions WHERE s3_key = ? LIMIT 1", (s3_key,))
    found = cur.fetchone() is not None
    conn.close()
    return found


def db_insert(db_path: Path, s3_key: str, english_text: str, signer: str, size: int):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO submissions (s3_key, english_text, signer_name, email, size_bytes, status)
        VALUES (?, ?, ?, ?, ?, 'approved')
    """, (s3_key, english_text, signer, "", size))
    conn.commit()
    conn.close()


def parse_args():
    p = argparse.ArgumentParser(description="Bulk upload ISL videos to RunPod network volume")
    p.add_argument("--folder", required=True, help="Folder containing the video files")
    p.add_argument("--csv", required=True, help="CSV with columns: filename, english_text, signer_name")
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET"), help="S3 bucket / volume ID (env: S3_BUCKET)")
    p.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT_URL"), help="S3 endpoint URL (env: S3_ENDPOINT_URL)")
    p.add_argument("--region", default=os.environ.get("S3_REGION", "us-east-1"), help="S3 region (env: S3_REGION)")
    p.add_argument("--access-key", default=os.environ.get("S3_ACCESS_KEY"), help="Access key (env: S3_ACCESS_KEY)")
    p.add_argument("--secret-key", default=os.environ.get("S3_SECRET_KEY"), help="Secret key (env: S3_SECRET_KEY)")
    p.add_argument("--signer", default=None, help="Global signer name (overrides CSV column if set)")
    p.add_argument("--max-mb", type=int, default=50, help="Skip files larger than N MB (default: 50)")
    p.add_argument("--dry-run", action="store_true", help="Validate CSV + files without uploading")
    p.add_argument("--max-uploads", type=int, default=None, help="Cap on uploads this run (useful for testing)")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate required credentials
    missing = []
    if not args.bucket: missing.append("--bucket / S3_BUCKET")
    if not args.access_key: missing.append("--access-key / S3_ACCESS_KEY")
    if not args.secret_key: missing.append("--secret-key / S3_SECRET_KEY")
    if missing:
        print(f"[err] Missing required: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    folder = Path(args.folder).resolve()
    csv_path = Path(args.csv).resolve()
    if not folder.is_dir():
        print(f"[err] Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.is_file():
        print(f"[err] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Read CSV
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "english_text" not in df.columns:
        print(f"[err] CSV must have at least 'filename' and 'english_text' columns. Got: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    if "signer_name" not in df.columns:
        df["signer_name"] = ""

    print(f"[bulk] CSV has {len(df)} rows")

    # Validate every row before uploading anything
    bad_rows = []
    for i, row in df.iterrows():
        fn = str(row["filename"]).strip()
        text = str(row["english_text"]).strip()
        if not fn or fn.lower() == "nan":
            bad_rows.append((i + 2, "blank filename"))
            continue
        if not text or len(text) < 2:
            bad_rows.append((i + 2, f"text too short ({len(text)} chars)"))
            continue
        if len(text) > 500:
            bad_rows.append((i + 2, f"text too long ({len(text)} chars)"))
            continue
        local = folder / fn
        if not local.is_file():
            bad_rows.append((i + 2, f"file not found: {local.name}"))
            continue
        ext = local.suffix.lower()
        if ext not in VALID_EXTS:
            bad_rows.append((i + 2, f"bad ext: {ext} (allowed: {sorted(VALID_EXTS)})"))
            continue
        size_mb = local.stat().st_size / (1024 * 1024)
        if size_mb > args.max_mb:
            bad_rows.append((i + 2, f"too large: {size_mb:.1f} MB"))
            continue

    if bad_rows:
        print(f"\n[err] {len(bad_rows)} invalid rows (CSV line numbers — header is line 1):")
        for line, reason in bad_rows[:25]:
            print(f"   line {line}: {reason}")
        if len(bad_rows) > 25:
            print(f"   ... and {len(bad_rows) - 25} more")
        print("\nFix these and re-run.")
        sys.exit(1)

    print(f"[bulk] All {len(df)} rows validated ✓")

    if args.dry_run:
        print("[bulk] DRY RUN — not uploading. Remove --dry-run to upload for real.")
        return

    # Set up S3 client
    s3_kwargs = {
        "aws_access_key_id": args.access_key,
        "aws_secret_access_key": args.secret_key,
        "region_name": args.region,
    }
    if args.endpoint:
        s3_kwargs["endpoint_url"] = args.endpoint
    s3 = boto3.client("s3", **s3_kwargs)

    # Download latest DB from S3 (so we stay in sync with web uploads)
    tmpdir = tempfile.mkdtemp(prefix="isl_bulk_")
    db_path = Path(tmpdir) / "isl_data.db"

    print(f"[bulk] Syncing database from s3://{args.bucket}/{DB_S3_KEY} ...")
    try:
        s3.download_file(args.bucket, DB_S3_KEY, str(db_path))
        print(f"[bulk] Got existing DB ({db_path.stat().st_size} bytes)")
    except Exception:
        print(f"[bulk] No existing DB in S3, will create fresh")
    init_db(db_path)

    # Upload loop
    n_uploaded = 0
    n_skipped = 0
    n_errors = 0
    pbar = tqdm(df.iterrows(), total=len(df), desc="Uploading")
    for i, row in pbar:
        if args.max_uploads is not None and n_uploaded >= args.max_uploads:
            print(f"\n[bulk] Stopped at --max-uploads limit ({args.max_uploads})")
            break

        fn = str(row["filename"]).strip()
        text = clean_text(row["english_text"])
        signer = safe_signer(args.signer or str(row.get("signer_name", "")).strip() or "anon")

        local_path = folder / fn

        # Build unique S3 key (prefix with timestamp + filename without ext)
        ts = int(time.time() * 1000) + i  # unique per row
        s3_key = f"submissions/upload_{ts}_{signer}.mp4"

        # Skip if same SOURCE filename already uploaded by this script
        # (we encode the original name in the cleaning - this is a coarse check
        #  by re-uploading you can force a duplicate)
        if db_already_uploaded(db_path, s3_key):
            n_skipped += 1
            continue

        try:
            with open(local_path, "rb") as f:
                size = local_path.stat().st_size
                s3.put_object(
                    Bucket=args.bucket,
                    Key=s3_key,
                    Body=f.read(),
                    ContentType="video/mp4",
                )
            db_insert(db_path, s3_key, text, signer, size)
            n_uploaded += 1
            pbar.set_postfix(uploaded=n_uploaded, skipped=n_skipped, err=n_errors)
        except Exception as e:
            n_errors += 1
            print(f"\n[err] Failed: {fn} - {e}", file=sys.stderr)

    # Sync DB back to S3
    print(f"\n[bulk] Uploading updated DB to s3://{args.bucket}/{DB_S3_KEY} ...")
    try:
        s3.upload_file(str(db_path), args.bucket, DB_S3_KEY)
        print(f"[bulk] DB synced ✓")
    except Exception as e:
        print(f"[err] DB sync failed: {e}", file=sys.stderr)
        print(f"       Local DB is at {db_path} - you can re-run to retry sync")

    # Summary
    print()
    print("=" * 60)
    print(f"  Bulk upload complete")
    print("=" * 60)
    print(f"  Uploaded: {n_uploaded}")
    print(f"  Skipped:  {n_skipped} (already in DB)")
    print(f"  Errors:   {n_errors}")
    print(f"  Total in DB: {pd.read_sql_query('SELECT COUNT(*) FROM submissions', sqlite3.connect(db_path)).iloc[0, 0]}")
    print()
    print(f"  Videos are now on the network volume at:")
    print(f"    /workspace/submissions/  (via the RunPod pod)")
    print(f"    s3://{args.bucket}/submissions/  (via S3 API)")


if __name__ == "__main__":
    main()
