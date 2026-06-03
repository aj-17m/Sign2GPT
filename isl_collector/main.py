"""
ISL Data Collection Pipeline.

Two upload modes via the same web app:
  - Single Video: one video + one English meaning
  - Bulk Upload: a CSV mapping + many video files

Both modes require email (so we can contact contributors if needed).
All uploads are auto-approved and saved to /workspace/submissions/
on the RunPod network volume via the S3-compatible API.

Required env vars:
    S3_BUCKET, S3_REGION, S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY
    MAX_VIDEO_MB (default: 50)
"""

import csv as csvmod
import io
import os
import re
import time
from pathlib import Path
from typing import List

import boto3
from botocore.client import Config as BotoConfig
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import (
    db_init,
    db_insert_submission,
    db_get_stats,
    sync_db_from_s3,
    sync_db_to_s3,
)


# ----- Configuration -----
S3_BUCKET = os.environ.get("S3_BUCKET", "isl-videos")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", None)
MAX_VIDEO_MB = int(os.environ.get("MAX_VIDEO_MB", "50"))
MAX_VIDEO_BYTES = MAX_VIDEO_MB * 1024 * 1024

LOCAL_DB_PATH = Path(os.environ.get("LOCAL_DB_PATH", "/tmp/isl_data.db"))
S3_DB_KEY = "_state/isl_data.db"

VALID_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ----- S3 client -----
s3_kwargs = {
    "aws_access_key_id": S3_ACCESS_KEY,
    "aws_secret_access_key": S3_SECRET_KEY,
    "region_name": S3_REGION,
    "config": BotoConfig(signature_version="s3v4"),
}
if S3_ENDPOINT_URL:
    s3_kwargs["endpoint_url"] = S3_ENDPOINT_URL

s3 = boto3.client("s3", **s3_kwargs)

# Restore DB from S3 if available, then ensure schema
print(f"[boot] Restoring database from s3://{S3_BUCKET}/{S3_DB_KEY} if exists...")
sync_db_from_s3(s3, S3_BUCKET, S3_DB_KEY, LOCAL_DB_PATH)
db_init(LOCAL_DB_PATH)


# ----- FastAPI app -----
app = FastAPI(title="ISL Video Collection")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _backup_db():
    try:
        sync_db_to_s3(s3, S3_BUCKET, S3_DB_KEY, LOCAL_DB_PATH)
    except Exception as e:
        print(f"[WARN] DB backup failed: {e}")


def _clean_text(s: str) -> str:
    s = (s or "").strip().lower()
    for ch in ".,!?;:\"'":
        s = s.replace(ch, "")
    return " ".join(s.split())


def _safe_signer(s: str) -> str:
    s = "".join(c for c in (s or "anon")[:30] if c.isalnum() or c in "_-")
    return s or "anon"


def _validate_email(email: str):
    if not email or not EMAIL_RE.match(email):
        raise HTTPException(400, "A valid email address is required")


def _upload_one_video(
    body: bytes,
    text: str,
    signer_name: str,
    email: str,
    size: int,
) -> dict:
    """Upload a single video to S3 and record in DB. Returns submission dict."""
    ts = int(time.time() * 1000)
    safe = _safe_signer(signer_name)
    s3_key = f"submissions/upload_{ts}_{safe}.mp4"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=body,
        ContentType="video/mp4",
    )

    sub_id = db_insert_submission(
        LOCAL_DB_PATH,
        s3_key=s3_key,
        english_text=text,
        signer_name=signer_name or "anonymous",
        email=email,
        size_bytes=size,
        status="approved",
    )
    return {"id": sub_id, "s3_key": s3_key}


# ============================================================================
# Public routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    stats = db_get_stats(LOCAL_DB_PATH)
    return templates.TemplateResponse("index.html", {"request": request, "stats": stats})


# ----- Single video upload -----

@app.post("/submit", response_class=HTMLResponse)
async def submit_single(
    request: Request,
    video: UploadFile = File(...),
    english_text: str = Form(...),
    signer_name: str = Form(...),     # mandatory
    email: str = Form(...),           # mandatory
):
    if not video.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(video.filename).suffix.lower()
    if ext not in VALID_VIDEO_EXTS:
        raise HTTPException(400, f"File type {ext} not supported. Use MP4/MOV/WEBM/AVI/MKV.")

    _validate_email(email)

    if not signer_name.strip():
        raise HTTPException(400, "Name is required")

    text = _clean_text(english_text)
    if len(text) < 2:
        raise HTTPException(400, "Please type what the sign means in English")
    if len(text) > 500:
        raise HTTPException(400, "Text too long (max 500 chars)")

    # Stream + size check
    chunks, size = [], 0
    while True:
        chunk = await video.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_VIDEO_BYTES:
            raise HTTPException(413, f"Video too large (max {MAX_VIDEO_MB} MB)")
        chunks.append(chunk)
    body = b"".join(chunks)
    if size < 1024:
        raise HTTPException(400, "Video file too small or corrupted")

    try:
        result = _upload_one_video(body, text, signer_name.strip(), email.strip(), size)
    except Exception as e:
        raise HTTPException(500, f"Failed to save video: {e}")

    _backup_db()

    return templates.TemplateResponse("thanks.html", {
        "request": request,
        "submission_id": result["id"],
        "english_text": text,
        "filesize_mb": round(size / (1024 * 1024), 2),
    })


# ----- Bulk upload (CSV + multiple videos) -----

@app.post("/submit-bulk", response_class=HTMLResponse)
async def submit_bulk(
    request: Request,
    csv_file: UploadFile = File(...),
    videos: List[UploadFile] = File(...),
    signer_name: str = Form(...),
    email: str = Form(...),
):
    _validate_email(email)
    if not signer_name.strip():
        raise HTTPException(400, "Name is required")

    # Parse CSV
    try:
        csv_bytes = await csv_file.read()
        csv_text = csv_bytes.decode("utf-8-sig").splitlines()
        reader = csvmod.DictReader(csv_text)
        rows = list(reader)
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if not rows:
        raise HTTPException(400, "CSV is empty")

    required_cols = {"filename", "english_text"}
    csv_cols = set(reader.fieldnames or [])
    missing_cols = required_cols - csv_cols
    if missing_cols:
        raise HTTPException(400,
            f"CSV missing required columns: {sorted(missing_cols)}. "
            f"Found: {sorted(csv_cols)}. Expected: filename, english_text")

    # Build map: filename -> english_text from CSV
    csv_map = {}
    for row in rows:
        fn = (row.get("filename") or "").strip()
        text = _clean_text(row.get("english_text") or "")
        if fn:
            csv_map[fn] = text

    # Process each video
    n_ok = 0
    n_fail = 0
    failures = []

    for vf in videos:
        fn = vf.filename or ""
        text = csv_map.get(fn)

        if text is None:
            failures.append({"filename": fn, "reason": "filename not in CSV"})
            n_fail += 1
            continue

        if len(text) < 2:
            failures.append({"filename": fn, "reason": "text too short"})
            n_fail += 1
            continue

        if len(text) > 500:
            failures.append({"filename": fn, "reason": "text too long (max 500 chars)"})
            n_fail += 1
            continue

        ext = Path(fn).suffix.lower()
        if ext not in VALID_VIDEO_EXTS:
            failures.append({"filename": fn, "reason": f"bad file type ({ext})"})
            n_fail += 1
            continue

        # Read video + size check
        chunks, size = [], 0
        oversize = False
        while True:
            chunk = await vf.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                oversize = True
                break
            chunks.append(chunk)

        if oversize:
            failures.append({"filename": fn, "reason": f"too large (max {MAX_VIDEO_MB} MB)"})
            n_fail += 1
            continue

        if size < 1024:
            failures.append({"filename": fn, "reason": "file too small or corrupted"})
            n_fail += 1
            continue

        body = b"".join(chunks)

        try:
            _upload_one_video(body, text, signer_name.strip(), email.strip(), size)
            n_ok += 1
        except Exception as e:
            failures.append({"filename": fn, "reason": f"upload error: {e}"})
            n_fail += 1

    _backup_db()

    return templates.TemplateResponse("bulk_thanks.html", {
        "request": request,
        "n_total": len(videos),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "failures": failures,
    })


@app.get("/health")
def health_check():
    return {"status": "ok", "bucket": S3_BUCKET}
