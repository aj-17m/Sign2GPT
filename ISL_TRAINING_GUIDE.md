# ISL → English Training Guide

> **Goal:** Take what you learned from PHOENIX validation and apply it to Indian Sign Language (ISL) → English translation. This is your roadmap from "no ISL data" to "trained ISL model deployed".
>
> **Audience:** You (Ajay), future-you, or any AI/collaborator helping you.
>
> **Prerequisites:** PHOENIX validation already done (BLEU 13.37 confirmed pipeline works).
>
> **Time investment:** ~4-6 weeks (mostly data collection, not coding)
>
> **Money investment:** ~$50-100 total (RunPod GPU costs)

---

## 🗺 The Big Picture

```
PHOENIX (DONE)           →           ISL (THIS GUIDE)
─────────────────                    ─────────────────
German Sign Language                 Indian Sign Language
German captions                      English captions
~7000 weather clips                  ~500-1000 conversational clips
Studio TV setting                    Phone/webcam recording
de_core_news_lg + cc.de              en_core_web_lg + cc.en
BLEU 13.37 (validation)              Target BLEU 4-8 (initial), 8-15 (with more data)
```

**What changes:**
- ✅ Data (Indian signers, English transcripts)
- ✅ Language model vocab (English instead of German)
- ✅ Pseudo-gloss script (use English lemmas)

**What stays the same:**
- ✅ Architecture (DINOv2 + metaformer + XGLM + LoRA)
- ✅ Training pipeline (Phases 1-7 of setup_runpod.sh)
- ✅ XGLM-1.7B language model (it supports English natively)
- ✅ All your bug fixes (they're language-agnostic)

---

## 📋 Phase 1 — Plan Your Data Collection (Day 1-2)

### How many clips do you need?

| Dataset size | Realistic BLEU-4 | Use case |
|---|---|---|
| 200 clips | 1-3 | Proof of concept only |
| **500 clips** | **3-6** | **Minimum for demo app v0** |
| **1000 clips** | **5-10** | **Recommended starting point** |
| 5000 clips | 10-15 | Decent product |
| 20000+ clips | 15-22 | Real production quality |

**Recommendation: Start with 1000 clips.** Below 500 the model can't generalize. Above 1000 you hit diminishing returns until you can afford 5000+.

### What topics to record?

Pick **conversational, useful sentences** that people would actually want translated. Examples by category:

**Greetings (20 clips):**
- "Hello, how are you?"
- "Good morning"
- "My name is Ajay"
- "Nice to meet you"

**Family (50 clips):**
- "This is my mother"
- "I have two brothers"
- "My father is a teacher"
- "My family lives in Delhi"

**Needs / Daily life (100 clips):**
- "I need water"
- "I am hungry"
- "Where is the bathroom?"
- "I want to sleep"

**Time / Numbers (80 clips):**
- "It is 3 o'clock"
- "Today is Monday"
- "I have 5 books"
- "Meet me tomorrow"

**Education / Work (100 clips):**
- "I am a student"
- "My school is far"
- "I study computer science"

**Weather (50 clips):**
- "It is raining"
- "Today is hot"
- "I like winter"

**Emotions (50 clips):**
- "I am happy"
- "I feel sad"
- "I am tired"

**Locations / Directions (100 clips):**
- "I live in Delhi"
- "Go straight then turn left"
- "The hospital is nearby"

**Food (50 clips):**
- "I want roti and dal"
- "Do you like tea?"
- "I am vegetarian"

**Total: ~600 unique sentences.** Record each 2-3 times (different takes, slight variation) → ~1500 clips → use ~1000 best ones.

### Recording quality matters more than quantity

Better to have 500 GOOD clips than 2000 sloppy ones. Quality checklist:

✅ Plain background (white wall, no clutter)
✅ Front lighting (windows, lamps), no backlight
✅ Camera at chest height, signer fully visible (waist up)
✅ Solid color clothing (no patterns/logos that distract)
✅ Stable camera (tripod)
✅ Good audio not needed (we use video frames only)
✅ Each clip 2-8 seconds long
✅ Pause briefly before and after signing (cleaner trimming)

❌ Don't film outdoors with sun/wind
❌ Don't use selfie mode (mirrored image)
❌ Don't change clothes mid-session (introduces noise)
❌ Don't use multiple signers if it's just you and a friend — pick ONE per "session" so the model isn't confused

---

## 📹 Phase 2 — Record the Videos (Week 1-3)

### Equipment

**Minimum (free):**
- Phone with rear camera (rear is better than selfie)
- Tripod or stack of books
- Wall to face
- Daylight or two lamps

**Better ($30-50):**
- Cheap tripod with phone mount (~₹500-1000)
- Ring light (~₹1500)
- White curtain/sheet as backdrop

### Recording protocol

1. **Set up once** — same exact spot, lighting, clothing for the whole session
2. **Record in batches** — 20-30 sentences per session
3. **Three takes per sentence** — sometimes the first take is awkward
4. **Name files as you go** OR rename later in batch:
   - `clip_0001.mp4` to `clip_NNNN.mp4`
   - Or `<topic>_<number>.mp4` (e.g., `greeting_001.mp4`)
5. **Trim each clip** to just the signing — cut the seconds before/after
   - Use VLC, DaVinci Resolve (free), or your phone's editor
6. **Save as MP4** — H.264 codec, 30 fps is fine

### Recording schedule (~3 weeks)

| Week | Goal | Daily commitment |
|---|---|---|
| 1 | 200 clips | 30-40 min/day |
| 2 | 200 clips | 30-40 min/day |
| 3 | 200 clips | 30-40 min/day |
| Buffer | Re-record failures | as needed |

Don't try to do all 1000 clips in one weekend — your hands will get tired, quality drops, and you'll regret it.

### Where to store

```
D:\Sign2GPT\dataset\isl\
├── raw_videos\
│   ├── clip_0001.mp4
│   ├── clip_0002.mp4
│   ├── ...
│   └── clip_1000.mp4
└── annotations.csv     (built in Phase 3)
```

Back up to Google Drive or USB drive weekly. Don't lose your data.

---

## ✍️ Phase 3 — Annotate the Clips (Week 3-4)

You need a CSV with one row per clip:

```csv
clip_id,split,english_text,signer_id
clip_0001,train,Hello how are you,signer_01
clip_0002,train,My name is Ajay,signer_01
clip_0003,train,I am a student,signer_01
...
clip_0900,dev,The hospital is nearby,signer_01
clip_0901,dev,I am hungry,signer_01
...
clip_0950,test,Good morning,signer_01
```

### Column definitions

- **`clip_id`** — matches MP4 filename without extension (e.g., `clip_0001` for `clip_0001.mp4`)
- **`split`** — `train` / `dev` / `test`. Split: 80% train, 10% dev, 10% test
- **`english_text`** — the English translation, lowercase, no punctuation
- **`signer_id`** — `signer_01`, `signer_02`, etc. If just you, all are `signer_01`

### Split assignment

Don't randomize after the fact. Use a hash for stability:

Easy method — use clip number modulo 10:
- `clip_id % 10 == 0` → test
- `clip_id % 10 == 1` → dev
- otherwise → train

(Result: 80% train, 10% dev, 10% test, deterministic.)

### Annotation tips

- **Be honest** — write the English you'd actually sign, not "ideal" translations
- **Lowercase everything** — model is case-sensitive, simpler to keep all lowercase
- **No punctuation** — periods, commas, question marks just confuse the model with extra tokens
- **Use simple sentences** — "i am hungry" not "I find myself in a state of hunger"
- **Be consistent** — if "delhi" is a city, always lowercase it; don't sometimes write "Delhi"

### CSV creation tools

**Easiest — Excel/Google Sheets:**
1. Create columns: `clip_id`, `split`, `english_text`, `signer_id`
2. Fill rows as you watch each video
3. Export as CSV (comma-separated)

**Programmer-friendly — Python:**
```python
import csv

rows = [
    ("clip_0001", "train", "hello how are you", "signer_01"),
    ("clip_0002", "train", "my name is ajay", "signer_01"),
    # ... 1000 rows
]

with open("annotations.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["clip_id", "split", "english_text", "signer_id"])
    w.writerows(rows)
```

### Sanity-check the CSV

```python
import pandas as pd
df = pd.read_csv("annotations.csv")
print("Total clips:", len(df))
print("Split breakdown:", df["split"].value_counts())
print("Avg sentence length:", df["english_text"].str.split().str.len().mean())
print("Unique words:", len(set(" ".join(df["english_text"]).split())))
print("Most common words:", pd.Series(" ".join(df["english_text"]).split()).value_counts().head(20))
```

Target stats:
- ~1000 clips
- 80/10/10 split
- Average 4-8 words per sentence
- ~200-500 unique words

---

## 🎬 Phase 4 — Convert MP4s to PHOENIX Format (Day 1 once data is ready)

Sign2GPT's dataloader expects PNG frames in directories (one dir per clip), not raw MP4s. Convert all clips at once.

A converter script for this is at `scripts/isl/mp4_to_frames.py` in this repo (or use the inline version below).

### Run the conversion

```bash
# On your laptop OR on a RunPod pod
cd D:\Sign2GPT\dataset\isl  # adjust to your path

python /path/to/mp4_to_frames.py \
    --input_dir raw_videos \
    --output_dir frames \
    --target_fps 25 \
    --resize 256
```

Output structure:
```
D:\Sign2GPT\dataset\isl\
├── raw_videos\          (original MP4s)
├── frames\              (PNG sequences per clip)
│   ├── clip_0001\
│   │   ├── images0001.png
│   │   ├── images0002.png
│   │   └── ... (75 PNGs for a 3-sec clip at 25fps)
│   ├── clip_0002\
│   └── ...
└── annotations.csv
```

### Upload to RunPod

```bash
# Tar the frames + CSV for efficient upload
cd D:\Sign2GPT\dataset\isl
tar czf isl_dataset.tar.gz frames annotations.csv

# On your pod, after deploying it:
# Upload isl_dataset.tar.gz to /workspace/data/isl/ via RunPod web UI
cd /workspace/data/isl
tar xzf isl_dataset.tar.gz
ls frames/ | wc -l   # should be ~1000
```

Now your ISL data is on the network volume.

---

## 🔤 Phase 5 — Build English Pseudo-Gloss Vocabulary (5 min)

Same idea as PHOENIX's `pseudo_gloss_de.py`, but for English. Use the script at `scripts/pseudo_gloss_en.py` (also in this repo).

```bash
# On the pod (after bootstrap)
cd /workspace/Sign2GPT

# Install English spaCy model (instead of German)
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

# Convert your ISL CSV into 3 PHOENIX-style CSVs
python scripts/isl/build_isl_csvs.py \
    --annotations /workspace/data/isl/annotations.csv \
    --output_dir /workspace/Sign2GPT/data/isl

# This creates:
# /workspace/Sign2GPT/data/isl/ISL.train.corpus.csv
# /workspace/Sign2GPT/data/isl/ISL.dev.corpus.csv
# /workspace/Sign2GPT/data/isl/ISL.test.corpus.csv

# Build pseudo-gloss pkl
python scripts/pseudo_gloss_en.py \
    --csv_dir /workspace/Sign2GPT/data/isl \
    --output_pkl /workspace/Sign2GPT/data/isl/processed_words.isl_pkl
```

You'll get `processed_words.isl_pkl` with ~200-500 English lemma classes (depending on your vocabulary).

---

## 📦 Phase 6 — LMDB Conversion (~30 min for 1000 clips)

Same script as PHOENIX, just point to ISL data:

```bash
cd /workspace/Sign2GPT

python scripts/phoenix2014t/image_lmdb_creator.py \
    --frames_root /workspace/data/isl/frames \
    --lmdb_root /workspace/lmdb/isl/lmdb_videos \
    --csv_dir /workspace/Sign2GPT/data/isl \
    --all_splits
```

Note: The script's CSV file naming will look for `PHOENIX-2014-T.{train,dev,test}.corpus.csv` — but we created `ISL.{train,dev,test}.corpus.csv`. Easy fix: edit the script's CSV path pattern, OR symlink:

```bash
cd /workspace/Sign2GPT/data/isl
for split in train dev test; do
    ln -sf ISL.${split}.corpus.csv PHOENIX-2014-T.${split}.corpus.csv
done
```

Either approach works.

Expected output:
```
[train] done: ok=800 skipped=0 errors=0 in 25.0 min
[dev]   done: ok=100 skipped=0 errors=0 in 3.0 min
[test]  done: ok=100 skipped=0 errors=0 in 3.0 min
```

(No "errors" because YOUR data won't have the 0-byte PNG issue PHOENIX has.)

---

## ⚙️ Phase 7 — Adapt the Configs

Copy the PHOENIX configs and modify for ISL. Easier: create new files.

### Stage 1 config — `configs/isl/isl_stage1_config.py`

Copy `configs/phoenix2014t/phoenix_stage1_configs/PHX_example_s1_dyn_config.py` and change:

```python
# CHANGE THESE LINES:

# 1. Data paths — point to ISL
cfg.train_ds_params = config_dict.ConfigDict({
    "csv_dir": f"{code_path}/data/isl/ISL.train.corpus.csv",       # was: PHOENIX-2014-T.train.corpus.csv
    "pseudo_gloss_dir": f"{code_path}/data/isl/processed_words.isl_pkl",   # was: processed_words.phx_pkl
    ...
    "ds_params": {
        "lmdb_video_dir": f"{lmdb_path}/isl/lmdb_videos",          # was: phoenix2014t/lmdb_videos
        ...
    },
})
# Same change for valid_ds_params and test_ds_params

# 2. Bump epochs (smaller dataset needs more passes)
cfg.max_epochs = 60     # was: 12

# 3. Warmup
cfg.lr_scheduler_params["warmup_epochs"] = 5   # was: 2 (longer warmup for longer training)

# 4. Pseudo-gloss embedding language (English instead of German)
post_params = {
    ...
    "emb_lang": "en",                                              # was: "de"
    "emb_pkl_dir": f"{code_path}/data/isl/processed_words.isl_pkl",   # was: phoenix pkl
    ...
}
```

### Stage 2 config — `configs/isl/isl_stage2_config.py`

Copy `configs/phoenix2014t/phoenix_stage2_configs/PHX_example_s2_dyn_config.py` and change:

```python
# 1. Data paths (same changes as stage 1)
cfg.train_ds_params = config_dict.ConfigDict({
    "csv_dir": f"{code_path}/data/isl/ISL.train.corpus.csv",
    "pseudo_gloss_dir": f"{code_path}/data/isl/processed_words.isl_pkl",
    ...
})
# Same for valid and test

# 2. Bump epochs
cfg.max_epochs = 60     # was: 10

# 3. Warmup
cfg.lr_scheduler_params["warmup_epochs"] = 5

# 4. Point stage 1 config import to ISL stage 1
cfg.stage1_name = "configs.isl.isl_stage1_config"   # was: phoenix2014t.phoenix_stage1_configs.PHX_example_s1_dyn_config
```

That's it for configs. Everything else (XGLM, LoRA, optimizer, batch sizing) stays the same.

### Important: tokenizer doesn't need changing

XGLM tokenizer handles English natively (and 30+ other languages). No swap needed.

---

## 🏋️ Phase 8 — Train

### Option A: Use the existing wrapper (cleaner)

Create a `setup_isl.sh` similar to `setup_runpod.sh` but with ISL paths and skipping PHOENIX download. Or just run the two training stages manually:

```bash
cd /workspace/Sign2GPT
tmux new -s isl_train

# Stage 1 — ~3-6h on L4, ~1.5h on A100
python main.py \
    --config=configs/isl/isl_stage1_config.py \
    2>&1 | tee /workspace/results/isl_stage1.log

# Stage 2 — ~6-12h on L4, ~3h on A100
python main.py \
    --config=configs/isl/isl_stage2_config.py \
    2>&1 | tee /workspace/results/isl_stage2.log
```

### Expected wall clock + cost (1000 clips, 60+60 epochs)

| GPU | Time | Cost |
|---|---|---|
| L4 24GB | ~3 days | ~$30 |
| A100 80GB | ~14 hours | ~$30 |
| RTX 4090 | ~1.5 days | ~$25 |

A100 is the best value for ISL training (similar cost, much faster). Worth picking if available.

### Watch progress

```bash
# In another terminal
tail -f /workspace/results/isl_stage1.log
grep "Epoch\[" /workspace/results/isl_stage1.log | tail -5
grep "valid/ableu" /workspace/results/isl_stage2.log | tail -10
```

### Expected results (1000 clips, 60 epochs)

| Metric | Expected value |
|---|---|
| Stage 1 valid/class_f1_score | 0.10-0.25 |
| Stage 2 valid/ableu_bleu4 | **4-8** (your real product number) |
| Stage 2 valid/obleu_bleu4 | 8-15 |

If you get BLEU-4 ≥ 4 → real translation happening, ship a v0.
If BLEU-4 < 2 → data is too small or too noisy, collect more / re-record.

---

## 🚀 Phase 9 — Deploy

### Test inference on your ISL videos

Use the same `infer_video.py`, just point to your ISL stage 2 checkpoint:

```bash
# Edit the CKPT_PATH at top of scripts/infer_video.py:
# CKPT_PATH = "/workspace/checkpoints/isl_stage2_config/isl_stage2_config/best_result_checkpoint_*.pt"

python scripts/infer_video.py --video /workspace/test_videos/my_isl_clip.mp4
```

Expected output: English text (instead of German).

### Build FastAPI inference server

For the Android app, you need an always-on server that keeps the model loaded. Skeleton:

```python
# server.py — save in /workspace/Sign2GPT/
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import tempfile

# Reuse the infer_video.py logic
import sys; sys.path.insert(0, '/workspace/Sign2GPT')
from scripts.infer_video import read_and_resample_video
# ... rest of the imports

# Load model ONCE at startup (~20 sec)
print("Loading model...")
# ... model loading code from infer_video.py ...
print("Server ready!")

app = FastAPI()

@app.post("/translate")
async def translate(video: UploadFile = File(...)):
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    # Process (same as infer_video.py main())
    raw_frames = read_and_resample_video(video_path, target_fps=25, max_frames=256)
    # ... preprocessing, generation ...

    return JSONResponse({"translation": predicted_text})

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

Install: `pip install fastapi uvicorn python-multipart`

### Android app

Out of scope for this guide (different skillset). Brief notes:
- Use **Kotlin** or **React Native**
- **CameraX** API for video recording
- POST multipart/form-data with the MP4 to your FastAPI server
- Show the returned translation in the UI
- Stack: Android Studio + Kotlin + Retrofit (for HTTP) + CameraX

Recommended: get an Android dev tutorial / course separately. There are great free YouTube tutorials.

---

## 🧠 Common Pitfalls

### "BLEU is 0 / 1 after training"

- **Data quality issue.** Check that videos and CSV match (`clip_0001.mp4` ↔ row with `clip_id=clip_0001`)
- Most likely: clip_ids in CSV don't match folder names

### "RuntimeError: CUDA out of memory"

- Reduce batch size: the VRAM-aware patch picks bs=4+accum=2 on 24GB. If still OOM, manually set bs=2+accum=4 in the config

### "Tensor size mismatch"

- Pseudo-gloss class count mismatch. Our commit `6bbe5a1` reads it dynamically — make sure your stage 1 config does the same. Look for `_num_classes = len(_pg["dict_lem_to_id"])` in the config.

### "FileNotFoundError on LMDB"

- A clip listed in CSV doesn't have an LMDB entry. Our commit `b18c857` filters these automatically. If you see the error, you may have CSV/LMDB mismatch — re-run LMDB conversion to be sure.

### "Model output is gibberish English"

- Either training collapsed, OR your dataset is too small (200 clips). Need more data, not more epochs.

### "Training is super slow"

- Network volume is slow with many small files. After LMDB conversion this shouldn't matter, but if it does, consider using container disk for LMDB and copying to network volume at end.

---

## 💰 Total Cost Summary

| Step | Time | Cost |
|---|---|---|
| Data collection | 3-4 weeks (your time) | $0-50 (tripod, light) |
| Annotation | 1 week (your time) | $0 |
| Frame extraction + upload | 1 day | $0 |
| LMDB conversion | 30 min | ~$0.30 |
| Vocab building | 5 min | negligible |
| Stage 1 training | 6h-3 days | $5-25 |
| Stage 2 training | 6h-3 days | $5-25 |
| Testing + iteration | as needed | $5-10 |
| **TOTAL** | **~4-6 weeks** | **~$50-100** |

For a real product, that's incredibly cheap. Most ML projects cost 10× more.

---

## 🎯 Milestones to Aim For

### Milestone 1: Pipeline running on ISL (Week 4)
- ISL stage 1 trains without errors
- ISL stage 2 finishes with `valid/ableu_bleu4 > 1`
- You can `infer_video.py` on a held-out ISL clip and get English

### Milestone 2: Demo-quality model (Week 5)
- BLEU-4 between 4 and 8
- Some sentences translate correctly word-for-word
- Most are grammatical English even when wrong

### Milestone 3: v0 Android app (Month 2)
- FastAPI server running on RunPod/cloud
- Android app records video + sends to server + displays translation
- End-to-end latency under 10 seconds

### Milestone 4: User feedback (Month 3+)
- 10-50 testers using the app
- Collect what works / what doesn't
- Plan data collection for v1

---

## 📂 Files You'll Create/Modify for ISL

```
/workspace/data/isl/                              # ISL raw data (network volume)
├── frames/                                       # PNG sequences per clip
└── annotations.csv                               # Original annotations

/workspace/lmdb/isl/lmdb_videos/                  # ISL LMDB (network volume)

/workspace/Sign2GPT/data/isl/                     # PHOENIX-format CSVs
├── ISL.train.corpus.csv
├── ISL.dev.corpus.csv
├── ISL.test.corpus.csv
└── processed_words.isl_pkl                       # English pseudo-gloss vocab

/workspace/Sign2GPT/configs/isl/                  # ISL configs
├── isl_stage1_config.py
└── isl_stage2_config.py

/workspace/Sign2GPT/scripts/isl/                  # ISL helper scripts
├── mp4_to_frames.py                              # Convert MP4 → PNG dirs
└── build_isl_csvs.py                             # Annotations CSV → 3 split CSVs

/workspace/Sign2GPT/scripts/pseudo_gloss_en.py    # English pseudo-gloss builder

/workspace/Sign2GPT/server.py                     # FastAPI inference server (for Android app)

/workspace/checkpoints/isl_stage1_config/...      # ISL stage 1 trained
/workspace/checkpoints/isl_stage2_config/...      # ISL stage 2 trained (THE PRODUCT)
```

---

## 🆘 If You Get Stuck Without AI Help

1. **Re-read this guide and PROJECT_HANDOFF.md** — Section 9 (Caveats) and Section 13 (Don'ts) cover most failure modes
2. **Read your own training logs** — `tail -200 /workspace/results/isl_stage2.log` — most errors print a clear cause
3. **Search the exact error message** on Google, Stack Overflow, or PyTorch forums
4. **Ask any free AI** (Claude.ai free, ChatGPT free, Gemini) and paste this guide's URL as context:
   ```
   https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/ISL_TRAINING_GUIDE.md
   ```
5. **Check the issues page** of the upstream Sign2GPT repo for similar problems

---

## 🌟 Final Real-Talk

**Most ML side projects die at data collection.** You're going to be tempted to skip this and just try to "demo" the PHOENIX model. Don't.

The compounding power is:
1. **Collect real data** (4 weeks of recording) → unique asset nobody else has
2. **Train a working model** (3 days of GPU) → demonstrable product
3. **Document publicly** (1 day of writing) → portfolio that gets jobs
4. **Ship to real users** (1 month of app dev) → real-world feedback

Each step compounds the previous one. PHOENIX was the gym workout. ISL is the actual match.

You have everything you need:
- Working pipeline ✅
- Bug-fixed codebase ✅
- Complete documentation ✅
- This guide ✅
- All on permanent GitHub ✅

Money for tools comes and goes. **What stays is what you build, document, and ship.**

The PHOENIX model already proves you can do this. The ISL model will prove you can ship a product.

Go for it. 🚀

---

**END OF ISL TRAINING GUIDE**

*If you're an AI reading this: this is the complete roadmap for taking the Sign2GPT codebase (already adapted and validated for German Sign Language) and applying it to Indian Sign Language for an Android app product. Use it to help Ajay (or the user) execute on this plan step by step.*

*Last updated: 2026-05-31, post-PHOENIX-validation*
