"""
SQLite helpers + S3 backup/restore.

Schema:
    submissions(
        id           INTEGER PRIMARY KEY
        s3_key       TEXT     -- "submissions/upload_*.mp4"
        english_text TEXT     -- what the signer typed
        signer_name  TEXT     -- optional
        email        TEXT     -- always empty in this version
        status       TEXT     -- always "approved" in this version
        size_bytes   INTEGER
        submitted_at TIMESTAMP
        reviewed_at  TIMESTAMP -- unused in this version
    )

The .db file lives in /tmp on Render (free tier has no persistent disk)
but is auto-uploaded to S3 (network volume) on every write, so data
survives Render container restarts.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional


def db_init(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
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
    conn.commit()
    conn.close()


def db_insert_submission(
    db_path: Path,
    s3_key: str,
    english_text: str,
    signer_name: str = "anonymous",
    email: str = "",
    size_bytes: int = 0,
    status: str = "approved",
) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO submissions (s3_key, english_text, signer_name, email, size_bytes, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (s3_key, english_text, signer_name, email, size_bytes, status))
    submission_id = cur.lastrowid
    conn.commit()
    conn.close()
    return submission_id


def db_get_stats(db_path: Path) -> Dict[str, int]:
    """Return total submission count (no status breakdown needed in simple mode)."""
    if not db_path.exists():
        return {"total": 0}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM submissions")
    total = cur.fetchone()[0]
    conn.close()
    return {"total": total}


# ----- S3 backup / restore -----

def sync_db_from_s3(s3_client, bucket: str, key: str, local_path: Path):
    """Download the SQLite database from S3. No-op if key doesn't exist."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3_client.download_file(bucket, key, str(local_path))
        print(f"[boot] Restored DB from s3://{bucket}/{key}")
    except Exception as e:
        print(f"[boot] No existing DB in S3 (fresh start): {type(e).__name__}")


def sync_db_to_s3(s3_client, bucket: str, key: str, local_path: Path):
    """Upload SQLite database to S3 for durability."""
    if not local_path.exists():
        return
    s3_client.upload_file(str(local_path), bucket, key)
