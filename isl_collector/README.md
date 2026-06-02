# ISL Sign Language Data Collection Pipeline

A FastAPI web app where anyone can upload an ISL sign video + the English meaning.
Admin reviews submissions. Approved data accumulates on `/workspace`, ready to feed
into the Sign2GPT ISL training pipeline.

## How it works

```
[Anyone on the internet]
        │
        │ visits https://your-pod-url/
        │ uploads video + types English text
        │
        ▼
[FastAPI app on RunPod CPU pod]
        │
        │ saves video to /workspace/isl_data/videos/
        │ records row in /workspace/isl_data/data.db
        │
        ▼
[You at /admin]
        │
        │ password-protected dashboard
        │ watches each video, approves/rejects
        │ clicks "Export ZIP" when ready
        │
        ▼
[Approved data ready for ISL training]
```

## Quick start — local testing on your laptop

```bash
# In this directory:
pip install -r requirements.txt

# Run locally with a test data dir (NOT /workspace)
export ISL_DATA_ROOT=./test_data
export ADMIN_PASSWORD=mypassword
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Visit http://localhost:8000 to test the upload form
# Visit http://localhost:8000/admin with user=admin password=mypassword to review
```

## Deploy on RunPod (production)

### Step 1 — Pick the cheapest CPU pod

In RunPod:
1. Deploy → search **CPU pods** (no GPU needed)
2. Pick the cheapest available (e.g., 2 vCPU + 4 GB RAM, ~$0.04-0.10/hr)
3. Same datacenter as your `sign2gpt-data` network volume
4. Container disk: 20 GB
5. Mount network volume `sign2gpt-data` at `/workspace`
6. **Expose port 8000** (HTTP) — this gives you a public URL
7. Template: any Ubuntu/Python (e.g., `runpod/pytorch:2.4.0`)
8. Deploy

### Step 2 — Set up the app

In Web Terminal:

```bash
# Clone the repo
cd /workspace
git clone -b validation/phoenix-12h-run https://github.com/aj-17m/Sign2GPT.git ISL_app
# (Or use existing Sign2GPT/ clone)
cd Sign2GPT/isl_collector

# Install dependencies (~30 sec)
pip install -r requirements.txt

# Set environment variables (CHANGE THE PASSWORD)
export ISL_DATA_ROOT=/workspace/isl_data
export ADMIN_PASSWORD=your_secret_password_here
export MAX_VIDEO_MB=50

# Run the app (in tmux so it survives terminal close)
apt-get install -y tmux
tmux new -s isl_app
uvicorn main:app --host 0.0.0.0 --port 8000

# Detach: Ctrl+B then D
```

### Step 3 — Get your public URL

RunPod automatically provides a public proxy URL for exposed ports:
```
https://<your-pod-id>-8000.proxy.runpod.net
```

Find this URL in:
- RunPod web UI → your pod → "Connect" → look for "HTTP Service" port 8000
- Or check pod logs for the exposed URL

### Step 4 — Test

1. Visit the URL → see upload form
2. Upload a test video
3. Visit `<URL>/admin`, log in with your password, approve it
4. Done — pipeline working publicly

## Endpoints

| URL | Auth | Purpose |
|---|---|---|
| `/` | Public | Upload form |
| `/submit` | Public | POST endpoint (form submits here) |
| `/health` | Public | Health check |
| `/admin` | Password | Dashboard with all submissions |
| `/admin/video/{id}` | Password | Stream a video for preview |
| `/admin/approve/{id}` | Password | Approve a submission |
| `/admin/reject/{id}` | Password | Reject a submission |
| `/admin/export` | Password | Download ZIP with approved data + CSV |

## Data storage

All data is on the network volume (`/workspace/isl_data/`):

```
/workspace/isl_data/
├── videos/          ← raw uploaded MP4s (one per submission)
├── data.db          ← SQLite database (tracking all submissions)
└── exports/         ← generated ZIP files (one per export)
```

This persists across pod terminations because it's on the network volume.

## After collecting enough data — train ISL model

Once you have ~500-1000 approved submissions:

1. **Export from admin dashboard:** Click "Download ZIP"
2. **The ZIP contains:**
   - `raw_videos/clip_NNNN.mp4` — all approved videos, renamed sequentially
   - `annotations.csv` — clip_id, split, english_text, signer_id (already in the right format!)

3. **Use the existing ISL pipeline:**
   ```bash
   # On a RunPod GPU pod with /workspace mounted
   cd /workspace/Sign2GPT

   # Unzip into the right location
   unzip /workspace/isl_data/exports/isl_export_*.zip -d /workspace/data/isl/

   # Run the existing ISL training pipeline (see ISL_TRAINING_GUIDE.md)
   python scripts/isl/mp4_to_frames.py --input_dir /workspace/data/isl/raw_videos --output_dir /workspace/data/isl/frames
   # ... etc, follow ISL_TRAINING_GUIDE.md
   ```

## Costs

| Item | Cost | Notes |
|---|---|---|
| RunPod CPU pod (24/7) | ~$30-70/month | Cheapest CPU pod, varies by region |
| Network volume (150 GB) | ~$10/month | Already paying for this |
| Bandwidth | Free | RunPod includes data transfer |
| **Total monthly** | **~$40-80** | While collecting data |

When ready to train ISL (every few weeks):
- Spin up GPU pod ~$20-50 per training run
- Same network volume → all data already there
- Terminate after training

## Customization

### Change admin password
Edit the `ADMIN_PASSWORD` environment variable, restart the server.

### Increase upload size limit
Edit `MAX_VIDEO_MB` (default 50). Higher = more disk used per video.

### Add reCAPTCHA / anti-spam
Out of scope for v1. If spam becomes a problem, add Cloudflare in front of the pod
(free tier handles bot protection).

### Custom domain
Use RunPod proxy URL OR set up a custom domain → Cloudflare Tunnel → your pod.

## Troubleshooting

**"Can't connect to port 8000"**
→ Make sure port 8000 is exposed in pod config. Check Connect tab → HTTP Service.

**"401 Unauthorized at /admin"**
→ Wrong password. Username is always `admin`. Set the password via `ADMIN_PASSWORD` env var.

**"413 File too large"**
→ Upload exceeds `MAX_VIDEO_MB`. Increase the limit or compress the video.

**"500 Internal Server Error"**
→ Check `/workspace/isl_data/` exists and is writable. Check pod logs:
   `tmux attach -t isl_app` to see errors.

**Server dies after terminal close**
→ Must run inside `tmux` (see Step 2 above).

## Project context

This is part of the [Sign2GPT ISL Project](https://github.com/aj-17m/Sign2GPT). The
data collected here is meant to train an ISL → English translation model based on the
[Sign2GPT ICLR 2024 paper architecture](https://github.com/ryanwongsa/Sign2GPT).

See `../PROJECT_HANDOFF.md` and `../ISL_TRAINING_GUIDE.md` for the full project context.
