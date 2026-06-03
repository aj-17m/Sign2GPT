# ISL Operations Guide

Everything you need to operate the ISL collection + training pipeline.

**Quick reference — your URLs and IDs:**

| What | Where |
|---|---|
| Public upload site | https://isl-collector-73ce.onrender.com |
| GitHub repo | https://github.com/aj-17m/Sign2GPT (branch `validation/phoenix-12h-run`) |
| RunPod network volume | `p014akuq8i` (ISL data) + `sign2gpt-data` (PHOENIX, separate) |
| S3 endpoint | `https://s3api-eu-ro-1.runpod.io` |
| Region | `eu-ro-1` |

---

## 1. Check uploaded videos

### Option A — From your laptop using AWS CLI (no pod needed, free)

**Windows Command Prompt:**
```cmd
pip install awscli

set AWS_ACCESS_KEY_ID=YOUR_RUNPOD_ACCESS_KEY
set AWS_SECRET_ACCESS_KEY=YOUR_RUNPOD_SECRET_KEY

aws s3 ls s3://p014akuq8i/submissions/ --endpoint-url https://s3api-eu-ro-1.runpod.io --region eu-ro-1
```

**Windows PowerShell:**
```powershell
pip install awscli

$env:AWS_ACCESS_KEY_ID = "YOUR_RUNPOD_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_RUNPOD_SECRET_KEY"

aws s3 ls s3://p014akuq8i/submissions/ --endpoint-url https://s3api-eu-ro-1.runpod.io --region eu-ro-1
```

**Mac / Linux / Git Bash:**
```bash
pip install awscli

export AWS_ACCESS_KEY_ID="YOUR_RUNPOD_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_RUNPOD_SECRET_KEY"

aws s3 ls s3://p014akuq8i/submissions/ --endpoint-url https://s3api-eu-ro-1.runpod.io --region eu-ro-1
```

You'll see a list of uploaded files with timestamps.

Download a specific video to review:
```bash
aws s3 cp s3://p014akuq8i/submissions/upload_FILENAME.mp4 review.mp4 --endpoint-url https://s3api-eu-ro-1.runpod.io --region eu-ro-1
```

---

### Option B — From a RunPod CPU pod (~$0.05/hour)

1. **Deploy a CPU pod** in RunPod:
   - GPU: None (CPU only — cheapest)
   - Network volume: `p014akuq8i` mounted at `/workspace`
   - Datacenter: `eu-ro-1`
   - Container disk: 10 GB
2. **Open Web Terminal**, then:

```bash
# Install sqlite for database queries
apt-get update && apt-get install -y sqlite3

# List uploaded videos (newest first)
ls -lht /workspace/submissions/

# Total count
ls /workspace/submissions/ | wc -l

# Total size on disk
du -sh /workspace/submissions/

# Show last 30 submissions (id, signer, email, text, time)
sqlite3 /workspace/_state/isl_data.db "
SELECT id, signer_name, email, english_text, datetime(submitted_at) as submitted
FROM submissions
ORDER BY id DESC
LIMIT 30;
"

# Count by signer
sqlite3 /workspace/_state/isl_data.db "
SELECT signer_name, COUNT(*) as videos
FROM submissions
GROUP BY signer_name
ORDER BY videos DESC;
"

# Find duplicates
sqlite3 /workspace/_state/isl_data.db "
SELECT english_text, COUNT(*) as n
FROM submissions
GROUP BY english_text
HAVING n > 1
ORDER BY n DESC;
"

# Vocabulary breakdown — most common words
sqlite3 /workspace/_state/isl_data.db "SELECT english_text FROM submissions;" | tr ' ' '\n' | sort | uniq -c | sort -rn | head -20

# Delete a bad upload (replace 42 with the actual id)
sqlite3 /workspace/_state/isl_data.db "DELETE FROM submissions WHERE id=42;"
rm /workspace/submissions/upload_BAD_FILENAME.mp4
```

**Remember to terminate the CPU pod when done to stop billing.**

---

## 2. Start training (one-command)

### Step 1 — Deploy a GPU pod in RunPod

| Setting | Value |
|---|---|
| GPU | A100 PCIe 80GB (~$1.69/hr, recommended) OR L4 24GB (~$0.43/hr, cheaper, slower) |
| **Do NOT use** | Blackwell GPUs (RTX PRO 4000 Blackwell, RTX 5090, etc.) — won't work with torch 2.4 |
| Template | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Container disk | 30 GB |
| Network volume | `p014akuq8i` mounted at `/workspace` |
| Datacenter | `eu-ro-1` (same as network volume) |

### Step 2 — Open Web Terminal and paste this ONE line:

```bash
curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh | bash
```

That's it.

The script will:
1. Verify GPU is supported
2. Verify network volume is mounted
3. Count uploaded videos (warns if too few)
4. Install all dependencies (~8-10 min)
5. Clone the repo
6. Start training inside a tmux session

After it prints "Training is running in the background" — close the browser tab. Training continues.

### Step 3 — Wait, then check results

Training time depends on dataset size:

| Submissions | Time on A100 | Cost on A100 | Time on L4 | Cost on L4 | Expected BLEU |
|---|---|---|---|---|---|
| 50 | ~1h | ~$1.70 | ~3h | ~$1.30 | 0-2 (poor) |
| 200 | ~3h | ~$5 | ~7h | ~$3 | 2-4 (weak) |
| 500 | ~6h | ~$10 | ~14h | ~$6 | 4-6 (demo OK) |
| 1000 | ~10h | ~$17 | ~22h | ~$10 | 6-9 (decent) |
| 5000 | ~18h | ~$30 | ~40h | ~$17 | 10-15 (real) |

---

## 3. Monitor training progress

From the GPU pod's Web Terminal:

```bash
# See current epoch
grep "Epoch\[" /workspace/results/isl_stage1.log 2>/dev/null | tail -3
grep "Epoch\[" /workspace/results/isl_stage2.log 2>/dev/null | tail -3

# See latest BLEU score
grep "valid/ableu_bleu4\|valid/class_f1_score" /workspace/results/isl_*.log 2>/dev/null | tail -10

# Watch live (Ctrl+C to stop watching, training continues)
tail -f /workspace/isl_training_*.log

# Reattach the live training session
tmux attach -t isl_train

# Detach again (training continues)
# Press: Ctrl+B then D

# Check tmux is still alive
tmux ls
```

---

## 4. Update training epochs

Default is **60 epochs each stage** (60 + 60 = 120 total). For longer training (more epochs = better quality, more time/cost):

### One-command with custom epochs

```bash
# Quick test: 10 epochs each (fast, low quality)
ISL_TRAIN_ARGS="--stage1-epochs 10 --stage2-epochs 10" \
    bash <(curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh)

# Demo quality: 30 epochs each
ISL_TRAIN_ARGS="--stage1-epochs 30 --stage2-epochs 30" \
    bash <(curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh)

# Default: 60 epochs each (good balance)
curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh | bash

# Paper-quality: 100 epochs each (more cost, more time, better results)
ISL_TRAIN_ARGS="--stage1-epochs 100 --stage2-epochs 100" \
    bash <(curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh)
```

### What epoch count to pick

| Use case | Stage 1 | Stage 2 | Why |
|---|---|---|---|
| Pipeline test | 10 | 10 | Just verify it trains without crashing |
| Demo quality | 30 | 30 | Decent results, modest cost |
| **Production** | **60** | **60** | **Recommended default — best value** |
| Paper-quality | 100 | 100 | Marginal gains, much higher cost |

---

## 5. Retrain after adding new data

When you collect more videos (e.g., 500 → 1000), use `--fresh-start` to retrain on the FULL updated dataset:

```bash
ISL_TRAIN_ARGS="--fresh-start" \
    bash <(curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh)
```

**Why `--fresh-start`:** Without it, training would try to resume from the old 500-clip model. The old model has wrong tensor sizes (old vocabulary, old class count) and won't load correctly.

`--fresh-start` deletes:
- Old training checkpoints
- Cached PNG frames
- Cached LMDB
- Cached vocabulary

…then trains a fresh model on your current full dataset.

### Combine `--fresh-start` with custom epochs

```bash
ISL_TRAIN_ARGS="--fresh-start --stage1-epochs 60 --stage2-epochs 60" \
    bash <(curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh)
```

### When to retrain (rough guide)

| Dataset growth | Retrain? |
|---|---|
| <25% more data | Don't bother — marginal gain |
| 50-100% more data | YES — significant improvement expected |
| 2-5× more data | YES — major improvement |
| Just to test a new approach | YES (use 10-epoch quick run first) |

**Important:** ALWAYS train on ALL clips, not just new ones. Training only on new data causes "catastrophic forgetting" — the model forgets what it learned from old data.

---

## 6. After training finishes

When you reattach tmux and see the training has completed:

```bash
# See final BLEU score (this is your model quality metric)
grep "valid/ableu_bleu4" /workspace/results/isl_stage2.log | tail -5
grep "test/ableu_bleu4" /workspace/results/isl_stage2.log | tail -5

# See sample translations the model produced
grep -B 1 -A 3 "ABLEU:" /workspace/results/isl_stage2.log | head -100

# Check trained checkpoints exist
ls -lh /workspace/checkpoints/isl_stage2_config/isl_stage2_config/
# Look for: best_result_checkpoint_*.pt
```

**Then terminate the GPU pod** in RunPod UI to stop billing. The trained model is safely stored on the network volume.

---

## 7. Run inference on a new ISL video

After training, test the model on a video:

```bash
# On a GPU pod with the network volume mounted + same setup
cd /workspace/Sign2GPT
python scripts/infer_video.py --video /path/to/your_isl_video.mp4
```

Output is the model's English translation.

---

## 8. Common troubleshooting

| Problem | Fix |
|---|---|
| `aws: command not found` | Run `pip install awscli`, then close + reopen terminal |
| Wrong shell syntax (export/set) | See section 1 — different syntax for CMD, PowerShell, Bash |
| `no kernel image is available for execution on the device` | You're on a Blackwell GPU. Terminate, deploy A100 or L4 instead |
| Quickstart says "no /workspace/submissions" | No videos uploaded yet OR wrong volume mounted (must be `p014akuq8i`) |
| Training stuck on first batch (10+ min) | Normal — model + data loading takes a few minutes on first run |
| Out of memory | Reduce epochs, or use smaller batch (config auto-detects but small GPU might need help) |
| Tmux session dies | Check `/workspace/isl_training_*.log` — output was tee'd there so it survives |
| Need to kill training | `tmux kill-session -t isl_train` |

---

## 9. Cost summary

| Phase | Service | Cost |
|---|---|---|
| **Always paying** | RunPod network volume `p014akuq8i` (50-150 GB) | ~$3-10/month |
| **Always paying** | Render web app (free tier) | $0/month |
| **Only when training** | RunPod GPU pod (A100 ~$1.69/hr or L4 ~$0.43/hr) | ~$10-30 per training run |
| **Only when checking** | RunPod CPU pod (cheap, ~$0.05/hr) | ~$0.05 per inspection session |

**Monthly fixed cost: ~$3-10. Variable cost: only when you actively train.**

---

## 10. Full workflow at a glance

```
SETUP (already done):
  ✅ Render web app live at https://isl-collector-73ce.onrender.com
  ✅ S3 credentials configured in Render
  ✅ Network volume p014akuq8i created
  ✅ GitHub repo with all code

DAILY/WEEKLY (during collection):
  - Share URL with potential contributors
  - Occasionally check uploads (Section 1)
  - Manually delete spam/bad uploads if any (Section 1, Option B)

WHEN READY TO TRAIN (every 500-1000 new clips):
  1. Deploy GPU pod with network volume p014akuq8i (Section 2)
  2. Paste the curl one-liner (with --fresh-start if retraining)
  3. Walk away for hours/days
  4. Come back, check BLEU score (Section 6)
  5. Terminate pod

WHEN ENOUGH DATA + GOOD MODEL:
  - Test inference on new ISL videos (Section 7)
  - Build FastAPI server for production deployment
  - Build Android app that hits the server
```

---

## 11. Files / scripts reference

All files are on GitHub at `aj-17m/Sign2GPT` branch `validation/phoenix-12h-run`:

| File | Purpose |
|---|---|
| `isl_quickstart.sh` | The one-command training launcher |
| `run_isl_training.py` | Master training pipeline orchestrator |
| `isl_collector/` | Web app (deployed on Render) |
| `scripts/isl/export_for_training.py` | DB → training-format converter |
| `scripts/isl/mp4_to_frames.py` | MP4 → PNG frames |
| `scripts/isl/build_isl_csvs.py` | annotations.csv → train/dev/test splits |
| `scripts/pseudo_gloss_en.py` | Build English pseudo-gloss vocabulary |
| `scripts/phoenix2014t/image_lmdb_creator.py` | PNG → LMDB |
| `scripts/infer_video.py` | Run trained model on a new video |
| `configs/isl/isl_stage1_config.py` | Stage 1 training config |
| `configs/isl/isl_stage2_config.py` | Stage 2 training config |

---

## 12. The 3 most-used commands (memorize these)

**Check uploads (from your laptop):**
```bash
aws s3 ls s3://p014akuq8i/submissions/ --endpoint-url https://s3api-eu-ro-1.runpod.io --region eu-ro-1
```

**Start training (paste on a fresh GPU pod with volume mounted):**
```bash
curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh | bash
```

**Retrain after adding more data:**
```bash
ISL_TRAIN_ARGS="--fresh-start" bash <(curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh)
```

That's it. The whole pipeline operationally.

---

*Last updated: 2026 — by Ajay's AI collaborator*
