"""
ISL Data Collection Pipeline — FastAPI app.

Public web app where anyone can upload an ISL sign video + English text.
Admin can review and approve submissions. Approved data accumulates on
/workspace network volume, ready for ISL model training.

Runs on a cheap RunPod CPU pod (always-on) with /workspace mounted.

Start the server (on RunPod or locally for testing):
    uvicorn main:app --host 0.0.0.0 --port 8000

Local testing without RunPod:
    Set ISL_DATA_ROOT=./test_data via environment variable.

Environment variables (set on the RunPod pod):
    ISL_DATA_ROOT       Where to store data (default: /workspace/isl_data)
    ADMIN_PASSWORD      Password for /admin (default: "changeme" - change this!)
    MAX_VIDEO_MB        Max upload size in MB (default: 50)
"""

import os
import secrets
import shutil
import time
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from database import db_init, db_insert_submission, db_list_submissions, db_update_status, db_get_stats

# ----- Configuration via environment variables -----
ISL_DATA_ROOT = Path(os.environ.get("ISL_DATA_ROOT", "/workspace/isl_data"))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")
MAX_VIDEO_MB = int(os.environ.get("MAX_VIDEO_MB", "50"))
MAX_VIDEO_BYTES = MAX_VIDEO_MB * 1024 * 1024

VIDEOS_DIR = ISL_DATA_ROOT / "videos"
DB_PATH = ISL_DATA_ROOT / "data.db"
EXPORTS_DIR = ISL_DATA_ROOT / "exports"

# Create dirs on startup
ISL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize database
db_init(DB_PATH)

# ----- FastAPI app -----
app = FastAPI(title="ISL Data Collection Pipeline")

# Static files (CSS) + HTML templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Admin auth (HTTP Basic — simple, no signup needed)
security = HTTPBasic()


def admin_required(credentials: HTTPBasicCredentials = Depends(security)):
    """Check admin password. Returns 401 if wrong."""
    correct_user = secrets.compare_digest(credentials.username, "admin")
    correct_pass = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ============================================================================
# Public routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Homepage with upload form."""
    stats = db_get_stats(DB_PATH)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
    })


@app.post("/submit")
async def submit_video(
    request: Request,
    video: UploadFile = File(...),
    english_text: str = Form(...),
    signer_name: str = Form(""),
    email: str = Form(""),
):
    """Receive upload, save video, record in DB."""

    # Validate file
    if not video.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(video.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".webm", ".avi", ".mkv"]:
        raise HTTPException(400, f"File type {ext} not supported. Use MP4, MOV, WEBM, AVI, or MKV.")

    # Validate text
    english_text = english_text.strip().lower()
    # Remove punctuation
    for ch in ".,!?;:\"'":
        english_text = english_text.replace(ch, "")
    english_text = " ".join(english_text.split())
    if not english_text or len(english_text) < 2:
        raise HTTPException(400, "Please type what the sign means (in English)")
    if len(english_text) > 500:
        raise HTTPException(400, "Text too long (max 500 chars)")

    # Save to a temporary file first to enforce size limit
    # Use a unique filename based on timestamp
    timestamp = int(time.time() * 1000)
    safe_signer = "".join(c for c in (signer_name or "anon")[:30] if c.isalnum() or c in "_-") or "anon"
    target_filename = f"upload_{timestamp}_{safe_signer}.mp4"
    target_path = VIDEOS_DIR / target_filename

    # Stream-save with size limit
    size = 0
    with open(target_path, "wb") as out_f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                out_f.close()
                target_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large (max {MAX_VIDEO_MB} MB)")
            out_f.write(chunk)

    # Record in database
    submission_id = db_insert_submission(
        DB_PATH,
        filename=target_filename,
        english_text=english_text,
        signer_name=signer_name.strip() or "anonymous",
        email=email.strip(),
        size_bytes=size,
    )

    # Redirect to thanks page
    return templates.TemplateResponse("thanks.html", {
        "request": request,
        "submission_id": submission_id,
        "english_text": english_text,
        "filesize_mb": round(size / (1024 * 1024), 2),
    })


@app.get("/health")
def health_check():
    """Simple health check for monitoring."""
    return {"status": "ok", "data_root": str(ISL_DATA_ROOT)}


# ============================================================================
# Admin routes (password protected)
# ============================================================================

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request, _admin: str = Depends(admin_required)):
    """Admin dashboard — list all submissions, allow approve/reject."""
    submissions = db_list_submissions(DB_PATH)
    stats = db_get_stats(DB_PATH)
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "submissions": submissions,
        "stats": stats,
    })


@app.post("/admin/approve/{submission_id}")
def admin_approve(submission_id: int, _admin: str = Depends(admin_required)):
    """Mark a submission as approved."""
    db_update_status(DB_PATH, submission_id, "approved")
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/reject/{submission_id}")
def admin_reject(submission_id: int, _admin: str = Depends(admin_required)):
    """Mark a submission as rejected."""
    db_update_status(DB_PATH, submission_id, "rejected")
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin/video/{submission_id}")
def admin_serve_video(submission_id: int, _admin: str = Depends(admin_required)):
    """Stream a video file for admin preview."""
    from database import db_get_submission
    sub = db_get_submission(DB_PATH, submission_id)
    if not sub:
        raise HTTPException(404, "Submission not found")
    video_path = VIDEOS_DIR / sub["filename"]
    if not video_path.is_file():
        raise HTTPException(404, "Video file missing on disk")
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/admin/export")
def admin_export(_admin: str = Depends(admin_required)):
    """Export all approved submissions as a ZIP + CSV."""
    approved = [s for s in db_list_submissions(DB_PATH) if s["status"] == "approved"]
    if not approved:
        raise HTTPException(400, "No approved submissions to export yet")

    # Build a temp ZIP
    timestamp = int(time.time())
    zip_path = EXPORTS_DIR / f"isl_export_{timestamp}.zip"

    # Build CSV string
    csv_rows = ["clip_id,split,english_text,signer_id"]
    for i, s in enumerate(approved, start=1):
        clip_id = f"clip_{i:04d}"
        # Split rule: 0 = test, 1 = dev, else train
        last_digit = i % 10
        split = "test" if last_digit == 0 else ("dev" if last_digit == 1 else "train")
        text = s["english_text"].replace(",", " ")
        signer = (s["signer_name"] or "anon").replace(",", "_")
        csv_rows.append(f"{clip_id},{split},{text},{signer}")
    csv_content = "\n".join(csv_rows)

    # Make ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.csv", csv_content)
        for i, s in enumerate(approved, start=1):
            src = VIDEOS_DIR / s["filename"]
            if src.is_file():
                zf.write(src, arcname=f"raw_videos/clip_{i:04d}.mp4")

    return FileResponse(
        zip_path,
        filename=zip_path.name,
        media_type="application/zip",
    )


# ============================================================================
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
# ============================================================================
