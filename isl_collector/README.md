# ISL Data Collection Pipeline (AWS S3 + Render edition)

A free-to-run web app where anyone can upload an ISL sign video + English meaning.
Videos go to AWS S3 (or any S3-compatible service). The app itself runs free on
Render. Total cost during collection: **$0/month**.

When you have enough data, run one command on a RunPod GPU pod to sync everything
down for ISL training.

## Architecture

```
                                ANYONE
                                  │
                                  │ visits your public URL
                                  ▼
                  ┌────────────────────────────┐
                  │  Render free tier          │
                  │  (FastAPI web app)         │
                  └────────────────────────────┘
                       │              │
       receives video  │              │ admin views/approves
       + English text  ▼              ▼
                  ┌────────────┐  ┌─────────────────┐
                  │  AWS S3    │  │  /admin (you)   │
                  │  isl-videos│  │                 │
                  └────────────┘  └─────────────────┘
                       │
                       │ when ready to train (every few weeks)
                       ▼
                  ┌────────────────────────────┐
                  │  RunPod GPU pod            │
                  │  aws s3 sync s3://...      │
                  │  -> /workspace/data/isl/   │
                  │  then train ISL model      │
                  └────────────────────────────┘
```

## Cost

| Service | Free tier | After free |
|---|---|---|
| Render (web app hosting) | 750 hours/month free | $7/month |
| AWS S3 (storage) | 5 GB for 12 months | $0.023/GB/month |
| AWS S3 (bandwidth out) | 100 GB/month for 12 months | $0.09/GB |
| RunPod GPU pod (only when training) | n/a | ~$20-30 per training run |
| **Monthly during collection** | | **$0** |

After 12 months of AWS free tier, AWS S3 costs ~$0.20/month for 10 GB.
You can also switch to Cloudflare R2 anytime (10 GB free forever).

## Setup — Part 1: AWS S3 (one-time, ~15 min)

### 1.1 Create an AWS account

1. Go to https://aws.amazon.com → Create Account
2. Credit card required (free tier won't charge unless you exceed limits)
3. Sign in to the AWS Console

### 1.2 Create an S3 bucket

1. Search "S3" → click S3
2. Click "Create bucket"
3. Bucket name: `isl-videos` (or any unique name)
4. Region: pick one near you (e.g., `ap-south-1` for India, `us-east-1` for US)
5. **Block all public access** → keep enabled (default)
6. Click "Create bucket"

### 1.3 Create IAM credentials

1. Search "IAM" → click IAM
2. Users → Add users
3. User name: `isl-collector-app`
4. Click Next → "Attach policies directly" → search "AmazonS3FullAccess" → check it
5. Click Next → Create user
6. Click on the new user → Security credentials tab
7. Click "Create access key" → "Application running outside AWS" → Next → Create
8. **Save these somewhere safe:**
   - Access key ID (looks like `AKIA...`)
   - Secret access key (long random string)
   - **You won't see the secret again — store it now**

## Setup — Part 2: Deploy to Render (one-time, ~15 min)

### 2.1 Create Render account

1. Go to https://render.com → Sign up (use GitHub login - easier)
2. No credit card needed for free tier

### 2.2 Connect your GitHub repo

1. Click "New +" → "Web Service"
2. Connect your GitHub account → select `aj-17m/Sign2GPT` repo
3. Configure:
   - **Name:** `isl-collector` (or any name - this becomes part of the public URL)
   - **Region:** pick same as your S3 region if possible
   - **Branch:** `validation/phoenix-12h-run`
   - **Root Directory:** `isl_collector`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free

### 2.3 Add environment variables in Render

Scroll down to "Environment Variables" and add:

| Key | Value |
|---|---|
| `S3_BUCKET` | `isl-videos` (or your bucket name) |
| `S3_REGION` | `us-east-1` (or your region) |
| `S3_ACCESS_KEY` | `AKIA...` (from IAM step 1.3) |
| `S3_SECRET_KEY` | your secret access key |
| `ADMIN_PASSWORD` | choose a strong password ONLY YOU KNOW |
| `MAX_VIDEO_MB` | `50` |
| `LOCAL_DB_PATH` | `/tmp/isl_data.db` |

(Skip `S3_ENDPOINT_URL` — only needed for non-AWS S3 like Cloudflare R2.)

### 2.4 Deploy

Click "Create Web Service". Render will:
1. Pull your code from GitHub
2. Run `pip install -r requirements.txt` (~2 min)
3. Start the app
4. Give you a public URL like `https://isl-collector.onrender.com`

After ~3 minutes, visit your URL. You should see the upload form.

### 2.5 Test it

1. Open URL in your browser
2. Upload a test video + type "test sentence" → Submit
3. You should see "Thanks for contributing!"
4. Visit `<URL>/admin` → username `admin`, password = your `ADMIN_PASSWORD`
5. You'll see your test submission with embedded video preview
6. Approve it → confirm it moves to "approved"

If all that works → you have a live public ISL collection pipeline. 🎉

## Setup — Part 3: Share + collect data (weeks)

Share your URL with:
- Indian Sign Language signers in your community
- Friends/family who can sign
- Deaf communities online (Reddit r/deaf, Facebook groups)
- Indian Sign Language interpreters
- Your college's signing club / NSS

Each visitor uploads (video + English meaning) at their own pace. You moderate
via `/admin`. Aim for ~1000 approved submissions for a usable first ISL model.

## Setup — Part 4: Train ISL model (when ready)

After collecting ~500-1000 approved submissions:

### 4.1 Option A: Download ZIP from admin (easiest)

1. Visit `<URL>/admin`
2. Click "Download ZIP" — gets a `.zip` with:
   - `raw_videos/clip_NNNN.mp4` (all approved videos, renamed sequentially)
   - `annotations.csv` (already in the format `ISL_TRAINING_GUIDE.md` expects)
3. Upload to RunPod via web UI to `/workspace/data/isl/`
4. Unzip → follow `ISL_TRAINING_GUIDE.md` Phase 4 onwards

### 4.2 Option B: Sync directly from S3 (faster for big datasets)

On a RunPod GPU pod with `sign2gpt-data` volume mounted:

```bash
# Set AWS credentials (same as Render env vars)
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-east-1

# Sync everything from S3 to /workspace
cd /workspace/Sign2GPT
bash isl_collector/sync_s3_to_runpod.sh isl-videos /workspace/data/isl/raw_videos

# This will download ALL submissions (including non-approved).
# If you only want approved, use Option A (Download ZIP from admin).
```

After that, follow `ISL_TRAINING_GUIDE.md`:
- Phase 4: `python scripts/isl/mp4_to_frames.py`
- Phase 5: build pseudo-gloss
- Phase 6: LMDB conversion
- Phase 8: train

## Local testing (before deploying)

Run on your laptop to verify everything works:

```bash
# Install dependencies
cd isl_collector
pip install -r requirements.txt

# Create a .env file
cp .env.example .env
# Edit .env with your S3 credentials and admin password

# Load env vars and start
export $(cat .env | xargs)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Visit http://localhost:8000
```

## Troubleshooting

**"InvalidAccessKeyId" or "SignatureDoesNotMatch"**
→ S3_ACCESS_KEY or S3_SECRET_KEY is wrong. Re-check IAM credentials.

**"NoSuchBucket"**
→ S3_BUCKET name typo, or region mismatch.

**Render deploy fails: "ImportError"**
→ A package missing. Check `requirements.txt` includes everything.

**Render free tier sleeps after 15 min idle**
→ Normal. First request after sleep takes ~30 sec to wake up. Use a keepalive
service (e.g., UptimeRobot free) to ping `/health` every 10 min to keep it warm.

**Data lost after Render restart**
→ Check the boot logs for "Restored database from s3://" message.
   If it says "No existing database in S3", the .db backup never made it
   to S3. Check S3 credentials and try again.

**413 File too large**
→ Increase `MAX_VIDEO_MB` env var. Render free tier has limited RAM (~512 MB),
   so don't go above ~80 MB.

**Want to use Cloudflare R2 instead of AWS S3**
→ Add `S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com` to env vars.
  R2 has 10 GB free forever (vs S3's 5 GB for 12 months only).

## Security notes

- **Change ADMIN_PASSWORD** before deploying. Default `changeme` is not secure.
- HTTP Basic auth is fine for one admin, but for multiple admins use a real auth system.
- Render's URLs are HTTPS by default — no extra config needed.
- IAM keys: limit to S3 access only (don't use root account keys).
- Don't commit `.env` to git (already in `.gitignore`).

## What's in the code

- `main.py` — FastAPI app with upload + admin endpoints
- `database.py` — SQLite helpers + S3 backup/restore
- `templates/index.html` — Public upload form
- `templates/thanks.html` — Post-submit page
- `templates/admin_dashboard.html` — Admin review queue
- `static/style.css` — Mobile-friendly styling
- `requirements.txt` — Python deps (FastAPI + boto3)
- `sync_s3_to_runpod.sh` — Run on RunPod to pull videos for training
- `.env.example` — Template for environment variables

## Project context

Part of the [Sign2GPT ISL Project](https://github.com/aj-17m/Sign2GPT). The data
collected here trains an ISL → English translation model based on Sign2GPT (ICLR 2024).

See `../PROJECT_HANDOFF.md` and `../ISL_TRAINING_GUIDE.md` for full project context.
