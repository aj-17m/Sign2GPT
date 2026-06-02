"""
SQLite database helpers + S3 backup/restore.

Database file lives locally (in /tmp on Render). On every write, we push
the .db file to S3 so it survives Render container restarts. On boot, we
pull the latest .db file from S3 (if any).

Schema (one table):
    submissions(
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        s3_key          TEXT     (e.g., "submissions/upload_1735000000_ajay.mp4")
        english_text    TEXT     (what the signer typed)
        signer_name     TEXT     (optional, user-provided)
        email           TEXT     (optional, user-provided)
        status          TEXT     "pending" | "approved" | "rejected"
        size_bytes      INTEGER
        submitted_at    TIMESTAMP
        reviewed_at     TIMESTAMP (NULL until reviewed)
    )
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional


def db_init(db_path: Path):
    """Create the submissions table if it doesn't exist."""
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
            status TEXT DEFAULT 'pending',
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
) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO submissions (s3_key, english_text, signer_name, email, size_bytes)
        VALUES (?, ?, ?, ?, ?)
    """, (s3_key, english_text, signer_name, email, size_bytes))
    submission_id = cur.lastrowid
    conn.commit()
    conn.close()
    return submission_id


def db_list_submissions(db_path: Path, limit: int = 500) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT id, s3_key, english_text, signer_name, email, status, size_bytes, submitted_at, reviewed_at
        FROM submissions
        ORDER BY submitted_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def db_get_submission(db_path: Path, submission_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def db_update_status(db_path: Path, submission_id: int, new_status: str):
    if new_status not in ("pending", "approved", "rejected"):
        raise ValueError(f"Invalid status: {new_status}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        UPDATE submissions
        SET status = ?, reviewed_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (new_status, submission_id))
    conn.commit()
    conn.close()


def db_get_stats(db_path: Path) -> Dict[str, int]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT status, COUNT(*) as count
        FROM submissions
        GROUP BY status
    """)
    by_status = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return {
        "total": sum(by_status.values()),
        "pending": by_status.get("pending", 0),
        "approved": by_status.get("approved", 0),
        "rejected": by_status.get("rejected", 0),
    }


# ----- S3 backup / restore helpers -----

def sync_db_from_s3(s3_client, bucket: str, key: str, local_path: Path):
    """Download the SQLite database from S3 to local_path. No-op if key doesn't exist."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3_client.download_file(bucket, key, str(local_path))
        print(f"[boot] Restored database from s3://{bucket}/{key}")
    except Exception as e:
        # Key probably doesn't exist on first run - that's fine
        print(f"[boot] No existing database in S3 (will create fresh): {type(e).__name__}")


def sync_db_to_s3(s3_client, bucket: str, key: str, local_path: Path):
    """Upload the local SQLite database to S3 for persistence."""
    if not local_path.exists():
        return
    s3_client.upload_file(str(local_path), bucket, key)
