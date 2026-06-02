"""
ISL Data Collection Pipeline — FastAPI app (AWS S3 + Render version).

Public web app where anyone can upload an ISL sign video + English text.
Videos are stored in AWS S3 (or any S3-compatible service like R2).
The SQLite database is auto-backed-up to S3 on every change, so data
survives Render restarts even on the free tier.

Designed to be deployed on Render free tier:
  - Web app runs free on Render
  - Videos live in S3 (5 GB free for 12 months)
  - $0/month during collection phase
  - When ready to train, run scripts/sync_s3_to_runpod.sh on a RunPod pod
    to download all approved videos to /workspace

Environment variables (set in Render dashboard):
    S3_BUCKET           AWS S3 bucket name (e.g., "isl-videos")
    S3_REGION           AWS region (e.g., "us-east-1")
    S3_ACCESS_KEY       AWS access key ID
    S3_SECRET_KEY       AWS secret access key
    S3_ENDPOINT_URL     (optional) for R2 / non-AWS S3-compatible services
    ADMIN_PASSWORD      Password for /admin (CHANGE THIS!)
    MAX_VIDEO_MB        Max upload size in MB (default: 50)

Start the server (Render does this automatically):
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import io
import os
import secrets
import time
import zipfile
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config as BotoConfig
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from database import (
    db_init,
    db_insert_submission,
    db_list_submissions,
    db_update_status,
    db_get_stats,
    db_get_submission,
    sync_db_from_s3,
    sync_db_to_s3,
)

# ----- Configuration from environment variables -----
S3_BUCKET = os.environ.get("S3_BUCKET", "isl-videos")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", None)  # set for R2 / non-AWS
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")
MAX_VIDEO_MB = int(os.environ.get("MAX_VIDEO_MB", "50"))
MAX_VIDEO_BYTES = MAX_VIDEO_MB * 1024 * 1024

# Local SQLite path (gets synced to S3 on every write)
LOCAL_DB_PATH = Path(os.environ.get("LOCAL_DB_PATH", "/tmp/isl_data.db"))
S3_DB_KEY = "_state/isl_data.db"  # where the .db file lives in S3

# ----- S3 client setup -----
if not S3_ACCESS_KEY or not S3_SECRET_KEY:
    print("[WARN] S3 credentials not set. Set S3_ACCESS_KEY and S3_SECRET_KEY env vars.")

s3_client_kwargs = {
    "aws_access_key_id": S3_ACCESS_KEY,
    "aws_secret_access_key": S3_SECRET_KEY,
    "region_name": S3_REGION,
    "config": BotoConfig(signature_version="s3v4"),
}
if S3_ENDPOINT_URL:
    s3_client_kwargs["endpoint_url"] = S3_ENDPOINT_URL

s3 = boto3.client("s3", **s3_client_kwargs)


# ----- App initialization -----
# Restore the SQLite database from S3 if it exists (survives Render restarts)
print(f"[boot] Restoring database from s3://{S3_BUCKET}/{S3_DB_KEY} if exists...")
sync_db_from_s3(s3, S3_BUCKET, S3_DB_KEY, LOCAL_DB_PATH)

# Initialize / ensure schema exists
db_init(LOCAL_DB_PATH)


# ----- FastAPI app -----
app = FastAPI(title="ISL Data Collection Pipeline")

# Static files (CSS) + HTML templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Admin auth (HTTP Basic)
security = HTTPBasic()


def admin_required(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, "admin")
    correct_pass = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def _backup_db():
    """Push the latest SQLite file to S3 so data survives Render restarts."""
    try:
        sync_db_to_s3(s3, S3_BUCKET, S3_DB_KEY, LOCAL_DB_PATH)
    except Exception as e:
        # Don't fail user requests if S3 backup fails - just log
        print(f"[WARN] Failed to backup DB to S3: {e}")


# ============================================================================
# Public routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Homepage with upload form."""
    stats = db_get_stats(LOCAL_DB_PATH)
    return templates.TemplateResponse("index.html", {"request": request, "stats": stats})


@app.post("/submit")
async def submit_video(
    request: Request,
    video: UploadFile = File(...),
    english_text: str = Form(...),
    signer_name: str = Form(""),
    email: str = Form(""),
):
    """Receive upload, save video to S3, record in DB."""

    # Validate filename
    if not video.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(video.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".webm", ".avi", ".mkv"]:
        raise HTTPException(400, f"File type {ext} not supported. Use MP4, MOV, WEBM, AVI, or MKV.")

    # Validate text
    english_text = english_text.strip().lower()
    for ch in ".,!?;:\"'":
        english_text = english_text.replace(ch, "")
    english_text = " ".join(english_text.split())
    if not english_text or len(english_text) < 2:
        raise HTTPException(400, "Please type what the sign means (in English)")
    if len(english_text) > 500:
        raise HTTPException(400, "Text too long (max 500 chars)")

    # Read upload to memory with size check (limit prevents OOM on Render's small RAM)
    chunks = []
    size = 0
    while True:
        chunk = await video.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_VIDEO_BYTES:
            raise HTTPException(413, f"File too large (max {MAX_VIDEO_MB} MB)")
        chunks.append(chunk)
    video_bytes = b"".join(chunks)

    if size < 1024:  # less than 1 KB - empty or invalid
        raise HTTPException(400, "Video file too small or corrupted")

    # Build unique S3 key
    timestamp = int(time.time() * 1000)
    safe_signer = "".join(c for c in (signer_name or "anon")[:30] if c.isalnum() or c in "_-") or "anon"
    s3_key = f"submissions/upload_{timestamp}_{safe_signer}.mp4"

    # Upload to S3
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=video_bytes,
            ContentType="video/mp4",
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to save video: {e}")

    # Record in database
    submission_id = db_insert_submission(
        LOCAL_DB_PATH,
        s3_key=s3_key,
        english_text=english_text,
        signer_name=signer_name.strip() or "anonymous",
        email=email.strip(),
        size_bytes=size,
    )

    # Backup DB to S3 so the entry survives a Render restart
    _backup_db()

    return templates.TemplateResponse("thanks.html", {
        "request": request,
        "submission_id": submission_id,
        "english_text": english_text,
        "filesize_mb": round(size / (1024 * 1024), 2),
    })


@app.get("/health")
def health_check():
    return {"status": "ok", "bucket": S3_BUCKET}


# ============================================================================
# Admin routes (password protected)
# ============================================================================

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request, _admin: str = Depends(admin_required)):
    submissions = db_list_submissions(LOCAL_DB_PATH)
    stats = db_get_stats(LOCAL_DB_PATH)
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "submissions": submissions,
        "stats": stats,
    })


@app.post("/admin/approve/{submission_id}")
def admin_approve(submission_id: int, _admin: str = Depends(admin_required)):
    db_update_status(LOCAL_DB_PATH, submission_id, "approved")
    _backup_db()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/reject/{submission_id}")
def admin_reject(submission_id: int, _admin: str = Depends(admin_required)):
    db_update_status(LOCAL_DB_PATH, submission_id, "rejected")
    _backup_db()
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin/video/{submission_id}")
def admin_serve_video(submission_id: int, _admin: str = Depends(admin_required)):
    """Generate a temporary signed URL to view the S3 video, then redirect."""
    sub = db_get_submission(LOCAL_DB_PATH, submission_id)
    if not sub:
        raise HTTPException(404, "Submission not found")

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": sub["s3_key"]},
            ExpiresIn=3600,  # 1 hour
        )
    except Exception as e:
        raise HTTPException(500, f"Could not generate video URL: {e}")

    return RedirectResponse(url=url)


@app.get("/admin/export")
def admin_export(_admin: str = Depends(admin_required)):
    """Build a ZIP with all approved videos + annotations.csv. Streams from S3."""
    approved = [s for s in db_list_submissions(LOCAL_DB_PATH) if s["status"] == "approved"]
    if not approved:
        raise HTTPException(400, "No approved submissions to export yet")

    # Build CSV
    csv_rows = ["clip_id,split,english_text,signer_id"]
    for i, s in enumerate(approved, start=1):
        clip_id = f"clip_{i:04d}"
        last_digit = i % 10
        split = "test" if last_digit == 0 else ("dev" if last_digit == 1 else "train")
        text = s["english_text"].replace(",", " ")
        signer = (s["signer_name"] or "anon").replace(",", "_")
        csv_rows.append(f"{clip_id},{split},{text},{signer}")
    csv_content = "\n".join(csv_rows)

    # Build ZIP in memory (warning: this loads all videos into RAM - OK for ~100s of videos)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.csv", csv_content)
        for i, s in enumerate(approved, start=1):
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=s["s3_key"])
                zf.writestr(f"raw_videos/clip_{i:04d}.mp4", obj["Body"].read())
            except Exception as e:
                print(f"[WARN] Could not fetch {s['s3_key']}: {e}")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=isl_export_{int(time.time())}.zip"},
    )


# ============================================================================
# Run with: uvicorn main:app --host 0.0.0.0 --port $PORT
# ============================================================================
