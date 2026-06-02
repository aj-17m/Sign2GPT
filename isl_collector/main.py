"""
ISL Data Collection Pipeline — Simple Edition.

Single-page web app:
  - Anyone uploads a sign video + types the English meaning
  - Upload goes straight to the network volume (via S3-compatible API)
  - Metadata recorded in SQLite for the training script to read later
  - NO admin login, NO approval queue, NO moderation
  - All submissions are auto-accepted ("approved" status)

Why this design:
  - Lower friction = more contributors
  - Training script (run_isl_training.py) reads directly from the database
  - User can SSH to the pod later to inspect/delete bad uploads if needed

Required env vars (set in Render):
    S3_BUCKET          (your RunPod network volume bucket ID)
    S3_REGION          (e.g., eu-ro-1)
    S3_ENDPOINT_URL    (e.g., https://s3api-eu-ro-1.runpod.io)
    S3_ACCESS_KEY      (RunPod API access key)
    S3_SECRET_KEY      (RunPod API secret)
    MAX_VIDEO_MB       (default: 50)

Start with: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import io
import os
import time
from pathlib import Path

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


# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    stats = db_get_stats(LOCAL_DB_PATH)
    return templates.TemplateResponse("index.html", {"request": request, "stats": stats})


@app.post("/submit", response_class=HTMLResponse)
async def submit_video(
    request: Request,
    video: UploadFile = File(...),
    english_text: str = Form(...),
    signer_name: str = Form(""),
):
    """Receive upload, save to S3, record in DB, auto-approve."""

    if not video.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(video.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".webm", ".avi", ".mkv"]:
        raise HTTPException(400, f"File type {ext} not supported. Use MP4/MOV/WEBM/AVI/MKV.")

    # Clean text
    text = english_text.strip().lower()
    for ch in ".,!?;:\"'":
        text = text.replace(ch, "")
    text = " ".join(text.split())
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

    # Build S3 key + upload
    ts = int(time.time() * 1000)
    safe_signer = "".join(c for c in (signer_name or "anon")[:30] if c.isalnum() or c in "_-") or "anon"
    s3_key = f"submissions/upload_{ts}_{safe_signer}.mp4"

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=body, ContentType="video/mp4")
    except Exception as e:
        raise HTTPException(500, f"Failed to save video: {e}")

    # Auto-approve (no admin review needed)
    submission_id = db_insert_submission(
        LOCAL_DB_PATH,
        s3_key=s3_key,
        english_text=text,
        signer_name=signer_name.strip() or "anonymous",
        email="",
        size_bytes=size,
        status="approved",  # KEY: auto-approve every submission
    )
    _backup_db()

    return templates.TemplateResponse("thanks.html", {
        "request": request,
        "submission_id": submission_id,
        "english_text": text,
        "filesize_mb": round(size / (1024 * 1024), 2),
    })


@app.get("/health")
def health_check():
    return {"status": "ok", "bucket": S3_BUCKET}
