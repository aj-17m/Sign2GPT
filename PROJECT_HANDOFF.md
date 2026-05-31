# Sign2GPT ISL Project — Complete Handoff Document

> **For the next AI assistant / collaborator**: This document is a self-contained
> brief on a real ongoing ML project. Read top-to-bottom before answering any
> questions about it. The user is the human owner; you are helping them continue
> the work. Be direct, practical, and respect what's already been built — don't
> re-litigate decisions or suggest starting over.

---

## 0. Quick Context

- **Goal:** Build an Indian Sign Language (ISL) → English translation model for an Android app.
- **Architecture used:** [Sign2GPT (Wong/Camgoz/Bowden, ICLR 2024)](https://github.com/ryanwongsa/Sign2GPT) — adapted for our use.
- **Two-phase strategy:**
  1. **PHOENIX-2014T validation** ← **DONE** (BLEU-4 = 13.37 on dev set, 4.4× target)
  2. **Custom ISL adaptation** ← next phase (need to collect ISL data first)
- **User:** Indian engineering student building this as a real product. Uses RunPod for cloud GPUs, Windows + PowerShell locally.

---

## 1. The User

- Final-year engineering student (IIITDM Jabalpur, email `22bcs018@iiitdmj.ac.in`)
- Building this as resume/portfolio + eventual real product
- **Strengths:** Practical, learns fast, ships things, debugs under pressure
- **Gaps:** English is not first language, prefers concise direct answers, sometimes needs ML concepts explained simply
- **Communication style:** Wants action-oriented answers with concrete commands, not theory dumps
- **Reality check needed sometimes:** Don't let them over-engineer or chase paper-quality on PHOENIX when ISL is the real goal

---

## 2. Project State (as of last working session)

### What's done

| Phase | Status | Result |
|---|---|---|
| PHOENIX-2014T data download + LMDB conversion | ✅ Done | 7,794 clips usable (94.4% of dataset) |
| Pseudo-gloss vocabulary | ✅ Done | 2,338 German lemmas |
| Stage 1 pretraining (12 epochs) | ✅ Done | best F1: 0.0488 at epoch 11 |
| Stage 2 translation training (10 epochs) | ✅ Done | **BLEU-4 = 13.37** on dev set |
| Single-video inference | ✅ Done | `scripts/infer_video.py` works |
| 9 critical bugs fixed in upstream codebase | ✅ Done | All on GitHub fork branch |

### What's next (in priority order)

1. Write public blog/LinkedIn post about the journey (highest career ROI)
2. Plan ISL data collection (~500-1000 clips with English transcripts)
3. Collect ISL clips (weeks)
4. Adapt pipeline for ISL (replace German with English, retrain)
5. Build FastAPI inference server
6. Build Android app
7. Ship to users

---

## 3. Architecture

```
INPUT: sign language video (frames at 25fps)
  │
  ├─ Vision Encoder: DINOv2 ViT-B (frozen, 86M params)
  │    → per-frame feature vector (768-dim)
  │
  ├─ Temporal Aggregator: Metaformer (trained, ~13M params for stage 1)
  │    → "sign token" sequence (variable length, 768-dim each)
  │
  ├─ Stage 1 only: Classification Head → multi-label pseudo-gloss prediction
  │
  └─ Stage 2: Language Model
        XGLM-1.7B (frozen, 1.7B params, multilingual incl. German/English/Hindi)
        + LoRA adapters rank-4 over 24 transformer layers (trained, ~5M params)
        → autoregressive German text generation

OUTPUT: German sentence (e.g., "morgen scheint verbreitet die sonne")
```

**Why XGLM:** It natively supports German + English + Hindi + 27 other languages.
When we move to ISL → English, we don't need to swap the LLM. Just retrain LoRA + vision.

---

## 4. Repository

- **User's fork:** https://github.com/aj-17m/Sign2GPT
- **Working branch:** `validation/phoenix-12h-run`
- **Upstream:** https://github.com/ryanwongsa/Sign2GPT (we don't push to upstream)
- **Local clone:** `D:\Sign2GPT\repo\` (on user's Windows machine)
- **Pod clone:** `/workspace/Sign2GPT/` (on RunPod)

### Bug fixes applied (chronological commit list on the branch)

| Commit | Title | What it fixed |
|---|---|---|
| `5fd48f7` | LMDB skips 0-byte/corrupt frames | PHOENIX-2014T release has ~55k empty PNG frames in 464 train clips; rejecting whole clip was wasteful. Now skips individual bad frames, only rejects clips with <16 valid frames. |
| `02da7e6` | Remove wrapper-level LMDB skip check | `setup_runpod.sh` was skipping Phase 3 when LMDB dir was non-empty, even if conversion was partial. Now always invokes the converter (per-clip idempotency handles re-runs). |
| `0b729e7` | VRAM-aware batch sizing | Auto bs/accum based on GPU: >=70GB → bs=8, >=40GB → bs=6, >=22GB → bs=4+accum=2, <22GB → bs=2+accum=4. Effective batch stays at 8 (preserves LR). |
| `48cf6b7` | Add xformers to install | Missing from upstream requirements. `models/metaformer/emb/sine_pos.py` imports xformers; training crashed at model init. |
| `15b828a` | Pin xformers to 0.0.27.post2 | Unpinned `pip install xformers` resolves to 0.0.35 which needs torch 2.7+ and CUDA driver ≥12.6. RunPod pytorch:2.4.0 template has driver 12.4 → "driver too old" crash. Use `--no-deps` so pip doesn't touch torch. |
| `6bbe5a1` | Read num_classes dynamically from pkl | Upstream hardcoded `num_classes=2306` but spaCy 3.7 produces 2338 lemmas. Mismatch crashed first epoch with "tensor size (2338) must match (2306)". Now reads `len(pkl["dict_lem_to_id"])`. |
| `b18c857` | Filter CSV for missing LMDB | After 0-byte frame patch rejects clips, CSV still lists them. Dataloader crashed with "lmdb.Error: No such file". Now filters df at dataset construction. |
| `2995ad8` | Fix rouge_metric → rouge_score import typo | Stage 2 trainer imports `from metrics.rouge_metric` but file is `metrics/rouge_score.py`. Stage 1 didn't use it so we didn't hit it earlier. |
| `4cc4860` | Add infer_clip.py (basic LMDB-clip inference) | Quick standalone script. Best-effort, may need tweaking. |
| `d680b8e` | Add translate_clip.py (trainer-backed) | Tries to reuse Trainer for inference. Fails because Trainer init kicks off training. Superseded by infer_video.py. |
| `9e17434` | Add match_predictions_to_clips.py | Maps existing log predictions to clip names by exploiting deterministic dev-set order. Useful for demo cherry-picking. |
| `e0a7cd0` | Add infer_video.py (working video inference) | **THE working inference script.** Reads MP4, preprocesses, runs through model, prints German translation. ~20 sec per call (mostly model load). |

### Files modified from upstream

```
setup_runpod.sh                                                              (new)
VALIDATION_RUN.md                                                            (new)
.gitattributes                                                               (new — LF endings on .sh)
configs/base/base_utils.py                                                   (env vars for paths)
environment_variables.py                                                     (wandb disabled, text logger)
configs/phoenix2014t/phoenix_stage1_configs/PHX_example_s1_dyn_config.py     (modified: VRAM-aware bs, dynamic num_classes, 12 epochs)
configs/phoenix2014t/phoenix_stage2_configs/PHX_example_s2_dyn_config.py     (modified: VRAM-aware bs, 10 epochs)
scripts/phoenix2014t/image_lmdb_creator.py                                   (new — PHOENIX PNG → LMDB)
dataloaders/phoenix_video_dataset.py                                         (modified: filter missing LMDB)
trainer/complete_translation_trainer.py                                      (modified: rouge import fix)
scripts/infer_clip.py                                                        (new — LMDB clip inference)
scripts/translate_clip.py                                                    (new — Trainer-based, broken)
scripts/match_predictions_to_clips.py                                        (new — log → clip name mapping)
scripts/infer_video.py                                                       (new — MP4 inference, WORKS)
```

---

## 5. Storage Locations (persistent network volume `/workspace`)

| Path | Contents | Size |
|---|---|---|
| `/workspace/Sign2GPT/` | Patched repo (cloned from fork's branch) | ~7 MB |
| `/workspace/data/phoenix2014t/PHOENIX-2014-T-release-v3/` | Raw PHOENIX frames (PNGs) | ~50 GB |
| `/workspace/lmdb/phoenix2014t/lmdb_videos/` | LMDB-packed clip data | ~35 GB |
| `/workspace/Sign2GPT/data/phoenix2014t/processed_words.phx_pkl` | Pseudo-gloss vocab | ~3 MB |
| `/workspace/checkpoints/phoenix_stage1_configs/PHX_example_s1_dyn_config/` | Stage 1 checkpoints | ~5 GB |
| `/workspace/checkpoints/phoenix_stage2_configs/PHX_example_s2_dyn_config/best_result_checkpoint_10_18.2415.pt` | **Trained stage 2 model** | 3.6 GB |
| `/workspace/results/stage1.log` and `stage2.log` | Training logs | ~few MB |
| `/workspace/test_videos/` | User-uploaded test videos | varies |

**Idle cost:** ~$0.34/day for the 150 GB network volume on RunPod.

---

## 6. Setup From Scratch (fresh pod)

### A. RunPod pod specs

- **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container disk:** 30 GB
- **Network volume:** `sign2gpt-data` (150 GB) mounted at `/workspace` — **must persist across pods**
- **GPU choices:**
  - For training: A100 80GB SXM/PCIe (preferred), L4 24GB (cheap, works), RTX 4090 (works)
  - For inference only: any Ada Lovelace (sm_89) or older. **AVOID Blackwell (sm_120) GPUs** with torch 2.4.0 — needs torch 2.7+cu128 instead.
  - Always **same datacenter** as the network volume

### B. Container-disk bootstrap (run on every fresh pod, ~10 min)

```bash
# 1. System libraries (apt)
apt-get update && apt-get install -y \
    tmux ffmpeg \
    zlib1g-dev libjpeg-dev libpng-dev libtiff-dev libfreetype6-dev

# 2. Python deps — full list (matches setup_runpod.sh Phase 1)
pip install --no-cache-dir \
    albumentations==1.4.13 \
    numpy==1.24.4 \
    pandas==2.0.1 \
    transformers==4.31.0 \
    lmdb==1.2.1 \
    timm==0.9.16 \
    ml-collections==0.1.1 \
    pytorch-ignite==0.4.13 \
    Pillow==9.0.1 \
    matplotlib \
    nlpaug==1.1.11 \
    nltk==3.6.7 \
    sentencepiece==0.1.99 \
    einops==0.8.0 \
    mediapipe==0.10.5 \
    onnxscript \
    albucore==0.0.13 \
    spacy==3.7.4 \
    opencv-python==4.8.1.78 \
    pybind11

# 3. Fasttext (special — needs --no-build-isolation)
pip install --no-build-isolation fasttext==0.9.2

# 4. spaCy German model (direct wheel, NOT `python -m spacy download`)
pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.7.0/de_core_news_lg-3.7.0-py3-none-any.whl

# 5. Re-pin torch (some deps above may have upgraded it)
pip install --force-reinstall torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 6. xformers pinned (must use --no-deps to prevent torch upgrade)
pip install xformers==0.0.27.post2 --no-deps

# 7. Fix scikit-image numpy ABI (often pulls newer numpy → ABI mismatch)
pip install --force-reinstall scikit-image==0.22.0 numpy==1.24.4
```

### C. Verify the environment

```bash
python -c "
import torch
print('Torch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
print('CUDA available:', torch.cuda.is_available())
"
python -c "import ml_collections, transformers, albumentations, timm, einops, lmdb, ignite, xformers; print('Imports OK')"
```

Expected on L4: `Compute capability: (8, 9)`. If compute capability ≥ 12, the GPU is too new for torch 2.4.0 — switch GPUs.

### D. Clone (only if network volume is empty — usually not needed)

If the network volume already has `/workspace/Sign2GPT/`, just `git pull`:
```bash
cd /workspace/Sign2GPT
git pull origin validation/phoenix-12h-run
```

Fresh clone:
```bash
cd /workspace
git clone -b validation/phoenix-12h-run https://github.com/aj-17m/Sign2GPT.git
```

### E. Always use tmux for long-running work

```bash
tmux new -s work
# (do stuff)
# Detach: Ctrl+B then D
# Reattach: tmux attach -t work
```

---

## 7. Running Training (full pipeline)

**Note:** Training is already done. This section is for reference / if you want to re-run with more epochs.

### One-shot training (Phase 1 → Phase 7)

```bash
cd /workspace/Sign2GPT
PHOENIX_URL='https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz' \
    bash setup_runpod.sh
```

This runs phases:
1. pip install (idempotent, ~30 sec if deps installed)
2. PHOENIX download + extract (~30 min, ~35 GB) — skips if already done
3. LMDB conversion (~25 min - 2.5h depending on GPU/storage) — skips per-clip if done
4. Pseudo-gloss vocab (FastText download + spaCy) (~20-30 min) — skips if pkl exists
5. Stage 1 training (~4h A100, ~10h L4, ~14h RTX PRO 4000)
6. Stage 2 training (~7h A100, ~15-20h L4)
7. Summary printed

### Stage 2 only (after stage 1 already trained — uses checkpoint)

```bash
python main.py \
    --config=configs/phoenix2014t/phoenix_stage2_configs/PHX_example_s2_dyn_config.py \
    2>&1 | tee /workspace/results/stage2.log
```

### Resume after interrupt

Both stages have `cfg.resume = True`. Just re-run the same command — trainer auto-loads last checkpoint.

### Epoch tuning

| Goal | Stage 1 epochs | Stage 2 epochs | Wall clock on L4 |
|---|---|---|---|
| Validation only (current) | 12 | 10 | ~12 hours |
| Decent demo | 25 | 25 | ~24 hours |
| Production quality | 60 | 60 | ~3-4 days |
| Paper reproduction | 100 | 100 | ~5-7 days |

Edit `cfg.max_epochs` in both config files. Recommend keeping `cfg.lr_scheduler_params.warmup_epochs = 2` for shorter runs.

---

## 8. Running Inference

### A. Translate a video file (recommended)

```bash
cd /workspace/Sign2GPT
python scripts/infer_video.py --video /workspace/test_videos/your_clip.mp4
```

Optional flags:
- `--num_beams 4` (default; lower for speed, higher for quality)
- `--max_length 64` (max output token count)
- `--target_fps 25` (input resampling rate)

**~20 sec per call** (mostly model loading). See `infer_video.py` header docstring for details.

### B. Match existing log predictions to clip names

If you have `/workspace/results/stage2.log` from a finished run:
```bash
python scripts/match_predictions_to_clips.py \
    --split dev \
    --log /workspace/results/stage2.log \
    --out /workspace/dev_clip_predictions.csv
```

Gives a CSV mapping clip name → model prediction → ground truth. Useful for cherry-picking demo clips without re-running inference.

### C. View raw eval samples from training log

```bash
grep -B 1 -A 3 "ABLEU:" /workspace/results/stage2.log | head -100
```

Shows `0 ABLEU: <prediction>` / `0 TGT: <ground truth>` pairs.

### D. Convert a PHOENIX clip's PNG sequence to MP4 (for demo videos)

```bash
CLIP="01April_2010_Thursday_heute-6694"
SPLIT="dev"
PNG_DIR="/workspace/data/phoenix2014t/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/$SPLIT/$CLIP"
mkdir -p /workspace/demos
ffmpeg -framerate 25 -pattern_type glob -i "$PNG_DIR/images*.png" \
    -c:v libx264 -pix_fmt yuv420p -y /workspace/demos/${CLIP}.mp4
```

---

## 9. Known Caveats and Failure Modes

### Model only knows German Sign Language (DGS) weather forecasts

- **Will work decently on:** DGS clips in studio settings, weather topic, 3-15 sec
- **Will fail on:** ISL, ASL, BSL, non-weather DGS, your webcam, anything outside PHOENIX distribution
- Even with paper-quality training (100 epochs), this fundamental limitation stays — it's a dataset constraint, not a model one
- Use for: validating pipeline only. NOT a usable product.

### GPU compatibility

- **Works:** Ampere (sm_80), Ada Lovelace (sm_89, e.g. L4 RTX 4090 RTX 4000 Ada), Hopper (sm_90)
- **Doesn't work with torch 2.4.0:** Blackwell (sm_120, e.g. RTX PRO 4000 Blackwell, RTX 5090, B200)
- For Blackwell, would need torch 2.7+cu128 + driver ≥12.8 + xformers 0.0.30+

### Memory considerations

- 24 GB VRAM minimum for stage 2 (with `bs=4, accum=2` from VRAM-aware patch)
- 80 GB VRAM allows default `bs=8` (paper config)
- 16 GB VRAM is borderline — inference works, stage 2 training likely OOMs

### Network storage (MooseFS on RunPod)

- Many small file operations are slow (50k+ PNGs took ~3h to extract from tarball)
- LMDB conversion is similarly slow (~25 min - 2.5h) — one-time cost
- After conversion, training reads are fast (large LMDB files)

### Phase 5/6 split for cost savings

Stage 1 and Stage 2 can be done on different pods (network volume preserves checkpoints):
1. Run stage 1 on cheap GPU (L4 or even RTX 4000 Ada)
2. Ctrl+C when "Phase 6/7" line appears
3. Terminate cheap pod, deploy A100 for stage 2
4. Run only stage 2: `python main.py --config=...stage2... 2>&1 | tee /workspace/results/stage2.log`

---

## 10. Current Results (last training run)

### Final metrics on PHOENIX-2014T dev set (519 clips)

```
Stage 1 (12 epochs):
  Best epoch: 11
  valid/class_f1_score: 0.0488 (target ≥ 0.25; not reached, but expected for 12-epoch run)

Stage 2 (10 epochs):
  Best epoch: 10
  valid/obleu_bleu1: 48.30
  valid/obleu_bleu2: 33.28
  valid/obleu_bleu3: 24.10
  valid/obleu_bleu4: 18.24
  valid/orouge:    49.11

  valid/ableu_bleu1: 35.13     ← REAL beam-search BLEU
  valid/ableu_bleu2: 23.71
  valid/ableu_bleu3: 17.28
  valid/ableu_bleu4: 13.37     ← HEADLINE METRIC (target ≥ 3; CRUSHED IT 4.4×)
  valid/arouge:     34.21
```

Test set was NOT auto-evaluated (trainer's auto-eval only does dev). Dev numbers are legitimate (held out from training, only checkpoint selection used it).

### Sample model outputs (real, from final evaluation)

**Exact matches (model got it perfect):**
```
PRED: morgen scheint verbreitet die sonne
TGT:  morgen scheint verbreitet die sonne

PRED: guten abend liebe zuschauer
TGT:  guten abend liebe zuschauer
```

**Near-perfect (template right, specifics wrong):**
```
PRED: und nun die wettervorhersage für morgen mittwoch den fünfundzwanzigsten juli
TGT:  und nun die wettervorhersage für morgen mittwoch den zweiten juni
                                                      ^^^^^^^^^^^^^^^^
PRED: am tag siebzehn grad an der ostsee und fünfundzwanzig grad am oberrhein
TGT:  am tag dreizehn grad bei dauerregen und einundzwanzig grad am oberrhein
```

**Live video inference test (user's uploaded `/workspace/test_videos/test1.mp4`):**
```
TRANSLATION: das tief das von westen zu uns strömt bringt uns in den nächsten
tagen kräftige regenwolken die sich in der westhälfte deutschlands ausbreiten

(English: The low pressure system flowing to us from the west brings strong
rain clouds in the coming days that spread across the western half of Germany)
```

### Cost spent so far

~$3-4 total for the full validation run (L4 24GB at ~$0.43/hr × ~8 hours).

---

## 11. The Bigger Plan (ISL Adaptation)

PHOENIX validation is just step 1. The actual product is ISL → English on Android.

### What changes for ISL

Same architecture. Different data and pseudo-gloss vocab:
- **Replace German captions with English transcripts**
- **Replace `de_core_news_lg` with `en_core_web_lg`** for spaCy
- **Replace `cc.de.300.bin` with `cc.en.300.bin`** for FastText
- **`num_classes`** will auto-update from the new pseudo-gloss pkl (commit `6bbe5a1`)
- **XGLM is unchanged** — it supports English natively (and Hindi if needed later)

### ISL data collection plan (target: 500-1000 clips)

```
PHASE 1 — Topic list (1 day)
- 100-200 common conversational sentences
- Categories: greetings, weather, food, family, time, numbers, directions, education
- Mix easy + complex sentences

PHASE 2 — Recording setup (1 day)
- Camera: phone with tripod, 1080p, 30fps
- Background: plain white/gray wall
- Lighting: front-facing soft even
- Framing: waist-up, hands fully visible
- Outfit: solid color, no patterns/logos

PHASE 3 — Recording (2-3 weeks, ~30 min/day)
- 10-20 sentences per session
- 1-2 signers (variety helps generalization)
- Each clip: 3-8 sec, 3 takes per sentence
- Save as MP4, name systematically: clip_NNNN.mp4

PHASE 4 — Annotation (1 week)
- CSV: clip_id, split (train/dev/test), english_text, signer_id
- 80/10/10 split, deterministic by clip_id

PHASE 5 — Storage layout
D:\Sign2GPT\dataset\
├── raw_videos\
│   ├── clip_0001.mp4
│   ├── ...
└── annotations.csv
```

### Expected ISL results (rough projections)

| ISL dataset size | Stage 2 epochs | Expected BLEU-4 | Quality |
|---|---|---|---|
| 500 clips | 60-80 | 2-5 | Demo only, often wrong |
| 1000 clips | 50-60 | 4-7 | Beta-test quality |
| 5000 clips | 30-50 | 8-12 | Decent product |
| 20000+ clips | 20-40 | 12-18 | Real product |

PHOENIX has 7000 clips and gets to ~22 BLEU at paper-quality training. Sign language datasets are extremely data-hungry.

### Deployment architecture (eventual)

```
Android App (Kotlin/React Native)
    │
    ├─ Records ~3 sec video, encodes H.264, sends via HTTPS
    │
    ▼
FastAPI server (always-on, GPU pod)
    │
    ├─ Model loaded ONCE at startup (~20 sec)
    ├─ Per request:
    │   - Decode video
    │   - Preprocess (~1 sec)
    │   - model.forward(generate=True) (~2-5 sec on L4)
    │   - Return English text
    │
    ▼
JSON {"translation": "..."} back to app
    │
    ▼
User sees English text on screen

Total user-perceived latency: ~5-7 sec per signed sentence
```

The FastAPI server skeleton would mirror `scripts/infer_video.py` but load the model once at module import instead of every request.

---

## 12. Useful Commands Reference

### Status checks (anytime)
```bash
nvidia-smi
df -h /workspace
ls /workspace/checkpoints/phoenix_stage2_configs/PHX_example_s2_dyn_config/
grep "valid/ableu_bleu4" /workspace/results/stage2.log | tail -5
```

### Training monitoring
```bash
# Latest epoch progress
grep "Epoch\[" /workspace/results/stage1.log | tail -5
grep "Epoch\[" /workspace/results/stage2.log | tail -5

# Latest BLEU
grep "valid/" /workspace/results/stage2.log | tail -20
```

### Sample inspection
```bash
# All ABLEU/TGT pairs
grep -B 1 -A 3 "ABLEU:" /workspace/results/stage2.log | head -200

# Count total
grep -c "ABLEU:" /workspace/results/stage2.log
```

### Tmux quickref
```bash
tmux new -s work       # create session
tmux attach -t work    # reattach
# Ctrl+B then D        # detach
tmux ls                # list sessions
```

### Git on the pod
```bash
cd /workspace/Sign2GPT
git status
git pull origin validation/phoenix-12h-run
git log --oneline -10
```

---

## 13. Critical Don'ts (avoid these)

1. **Don't use `python -m spacy download de_core_news_lg`** — broken URL on most pods. Use the direct wheel URL.
2. **Don't `pip install xformers` without `--no-deps` and version pin** — pulls newer torch → crashes with "driver too old".
3. **Don't terminate pod mid-extraction or mid-training** without checking checkpoints saved.
4. **Don't deploy on Blackwell GPUs** (RTX PRO 4000 Blackwell, RTX 5090, B200) with torch 2.4.0 — kernel mismatch.
5. **Don't pick a pod in a different datacenter than the network volume** — can't mount.
6. **Don't `git add .` blindly** — there are `.pyc` and `__pycache__` dirs.
7. **Don't run training without tmux** — browser tab close = run dies.
8. **Don't extend PHOENIX training thinking it improves the ISL product** — they're independent.
9. **Don't test the PHOENIX model on ISL videos and expect anything sensible** — completely different language.

---

## 14. Critical Do's

1. **DO use tmux for everything that takes >2 minutes**
2. **DO verify network volume is mounted** (`ls /workspace`) before doing anything
3. **DO use `bash setup_runpod.sh` for training** — handles all phases idempotently
4. **DO commit + push fixes to the fork's branch** — future runs benefit
5. **DO check `nvidia-smi` first if anything is slow or weird**
6. **DO use `scripts/infer_video.py` for inference, not the older clip scripts**
7. **DO read this entire document before answering project questions**
8. **DO challenge over-optimization** — user's goal is shipping ISL app, not paper-quality PHOENIX

---

## 15. Quick "What Should I Do Now?" Decision Tree

```
User came back to project. What's the goal of this session?

├── "I want to test something / make a demo"
│   └── Use scripts/infer_video.py on /workspace/test_videos/*.mp4
│       Or use existing log samples from /workspace/results/stage2.log
│
├── "I want to ship the Android app"
│   └── Need ISL data first. Either:
│       a) Start data collection planning (no pod needed, just discussion)
│       b) Skeleton FastAPI server with PHOENIX model as template
│
├── "I want to retrain with more epochs / better quality on PHOENIX"
│   └── Edit max_epochs in both configs, re-run setup_runpod.sh
│       Network volume preserves existing checkpoints; trainer auto-resumes.
│       BUT: gently challenge whether this is the best use of time vs. ISL work.
│
├── "Something is broken / errors out"
│   └── Check Section 9 (caveats) and Section 13 (don'ts) first.
│       Most issues are pip/CUDA version mismatches.
│
├── "I want to write about this publicly / put on resume"
│   └── Use Section 10 (results) + Section 4 (bug list) as raw material.
│       The journey (8 bugs found and fixed) is more impressive than the BLEU number.
│
└── "Something else"
    └── Ask clarifying questions. Don't assume.
```

---

## 16. Contact Continuity Hints

The user typically:
- Pastes raw terminal output and asks "what is this?" or "why this fail?"
- Wants concrete next commands, not theoretical explanations
- Switches between technical questions and career/strategy questions ("how AI will replace humans", "what should I learn next")
- Sometimes asks "where do I put this file?" or "what directory" — usually answer is `/workspace/Sign2GPT/`
- Gets confused between "Phase N" in setup_runpod.sh and "Stage 1/2" in training — clarify both
- Has limited time per session (few hours) and limited budget per run (~$5-30)

When in doubt: **give the exact command to run, then explain why.**

---

## 17. RESTART GUIDE — After Pod Termination

> **Most common entry point.** User has terminated the previous pod (or it died) and wants to continue. Network volume `sign2gpt-data` preserves everything important. Container disk is gone. Follow these steps in order.

### Step 1 — Verify network volume has the data (1 min)

In RunPod web UI:
1. Go to **Storage → Network Volumes**
2. Confirm `sign2gpt-data` (or whatever it's named) exists, ~150 GB, **not deleted**
3. **Note the datacenter** (e.g., `EU-RO-1`, `US-CA-2`) — you MUST deploy a pod in the same DC

If volume is missing → catastrophic data loss, would need to redo PHOENIX download/training from scratch. Should never happen unless user manually deleted it.

### Step 2 — Deploy a new pod (3-5 min)

In RunPod web UI:
1. **Pods → Deploy**
2. **Filter location** = your network volume's datacenter
3. **GPU choice** (depending on goal):

| Goal | Recommended GPU | Hourly cost | Why |
|---|---|---|---|
| Just inference / demo | L4 24GB | ~$0.43/hr | Cheapest that works, 20 sec per call |
| Just inference (faster) | RTX 4090 24GB | ~$0.69/hr | ~3× faster than L4 |
| Inference + light dev | RTX 4000 Ada 20GB | ~$0.39/hr | Cheapest workstation card |
| Stage 1 only training | L4 24GB | ~$0.43/hr | ~10h, ~$4 total |
| Full training (paper repro) | A100 80GB PCIe | ~$1.69/hr | Fast, proven config |
| Full training (cheaper) | L4 24GB | ~$0.43/hr | Slower but works |

**AVOID:** Blackwell GPUs (RTX PRO 4000 Blackwell, RTX 5090, B200). They have compute capability 12.x which torch 2.4.0 doesn't support. You'd get `no kernel image is available for execution on the device`.

4. **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
5. **Container disk:** 30 GB
6. **Network volume:** Select `sign2gpt-data` → mount at `/workspace`
7. Click **Deploy On-Demand**
8. Wait ~30 sec for boot
9. Click **Connect → Start Web Terminal**

### Step 3 — Verify the mount worked (10 sec)

```bash
# Confirm /workspace has your stuff
ls /workspace/Sign2GPT/setup_runpod.sh
ls /workspace/checkpoints/phoenix_stage2_configs/PHX_example_s2_dyn_config/best_result_checkpoint_10_18.2415.pt
ls /workspace/lmdb/phoenix2014t/lmdb_videos | wc -l
df -h /workspace
```

Expected:
- First two `ls` print the file path (not "not found")
- Count is ~7794 (LMDB clip dirs)
- `df` shows the 150 GB network volume mounted at `/workspace`

If any of these fail → network volume didn't mount. Stop, redeploy with correct mount.

### Step 4 — Container-disk bootstrap (~8-12 min)

This is the same every time because container disk wipes on terminate. Copy-paste the entire block:

```bash
# 1. System libs (apt)
apt-get update && apt-get install -y \
    tmux ffmpeg \
    zlib1g-dev libjpeg-dev libpng-dev libtiff-dev libfreetype6-dev

# 2. Python deps
pip install --no-cache-dir \
    albumentations==1.4.13 \
    numpy==1.24.4 \
    pandas==2.0.1 \
    transformers==4.31.0 \
    lmdb==1.2.1 \
    timm==0.9.16 \
    ml-collections==0.1.1 \
    pytorch-ignite==0.4.13 \
    Pillow==9.0.1 \
    matplotlib \
    nlpaug==1.1.11 \
    nltk==3.6.7 \
    sentencepiece==0.1.99 \
    einops==0.8.0 \
    mediapipe==0.10.5 \
    onnxscript \
    albucore==0.0.13 \
    spacy==3.7.4 \
    opencv-python==4.8.1.78 \
    pybind11

# 3. fasttext (special — needs --no-build-isolation)
pip install --no-build-isolation fasttext==0.9.2

# 4. spaCy German model (direct wheel)
pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.7.0/de_core_news_lg-3.7.0-py3-none-any.whl

# 5. Re-pin torch (some deps may have upgraded it silently)
pip install --force-reinstall torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 6. xformers pinned to torch 2.4.x-compatible version
pip install xformers==0.0.27.post2 --no-deps

# 7. Fix scikit-image numpy ABI (common issue — installer pulls newer numpy/scipy/skimage)
pip install --force-reinstall scikit-image==0.22.0 numpy==1.24.4
```

### Step 5 — Verify the environment (15 sec)

```bash
python -c "
import torch
print('Torch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('Compute:', torch.cuda.get_device_capability(0))
print('CUDA available:', torch.cuda.is_available())
x = torch.randn(2, 3).cuda()
print('Tensor on GPU OK:', x.shape)
"

python -c "
import ml_collections, transformers, albumentations, timm, einops, lmdb, ignite, xformers, cv2
from PIL import Image
print('All imports OK')
"
```

Both should succeed with no errors. If you see:
- `no kernel image is available for execution on the device` → GPU is Blackwell, terminate and pick a different GPU
- `numpy.dtype size changed` → repeat step 7 of bootstrap
- `ModuleNotFoundError` for any X → `pip install X` (then add to your local notes)

### Step 6 — Pull the latest patches (5 sec)

```bash
cd /workspace/Sign2GPT
git pull origin validation/phoenix-12h-run
```

This grabs any new fixes added since your last session.

### Step 7 — Start tmux (mandatory for >2 min work)

```bash
tmux new -s work
```

Detach later with **Ctrl+B then D**, reattach with `tmux attach -t work`.

### Step 8 — Pick what to do

Based on your goal:

#### Option A — Test inference on a video (most common)

```bash
# Upload your MP4 via RunPod web UI to /workspace/test_videos/
mkdir -p /workspace/test_videos
# (drag MP4 to /workspace/test_videos/ in RunPod Files tab)

# Translate it
python scripts/infer_video.py --video /workspace/test_videos/your_video.mp4
```

Takes ~20 sec (mostly model loading). Outputs German translation.

#### Option B — Re-evaluate the model / cherry-pick demo samples

```bash
# Use existing log without re-running training
python scripts/match_predictions_to_clips.py \
    --split dev \
    --log /workspace/results/stage2.log \
    --out /workspace/dev_clip_predictions.csv

head -20 /workspace/dev_clip_predictions.csv | column -t -s '|'
```

#### Option C — Extend training (more epochs for better demo quality)

```bash
# Edit configs to bump epochs
sed -i 's/cfg.max_epochs = 12/cfg.max_epochs = 40/' \
    configs/phoenix2014t/phoenix_stage1_configs/PHX_example_s1_dyn_config.py
sed -i 's/cfg.max_epochs = 10/cfg.max_epochs = 40/' \
    configs/phoenix2014t/phoenix_stage2_configs/PHX_example_s2_dyn_config.py

# Run — trainer auto-resumes from existing checkpoints
PHOENIX_URL='https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz' \
    bash setup_runpod.sh
```

Cost estimate for going from 12→40 epochs stage 1 and 10→40 epochs stage 2:
- On L4: ~25-30 additional hours, ~$11-13
- On A100: ~12-15 additional hours, ~$23-28

#### Option D — Start fresh (only if data was somehow lost)

This should never be needed unless network volume was deleted. If it was, recovery is ~12h training + ~$5-25:

```bash
PHOENIX_URL='https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz' \
    bash setup_runpod.sh
```

This runs all 7 phases from scratch (download, extract, LMDB, vocab, stage 1, stage 2, summary).

### Step 9 — When done, terminate cleanly

```bash
# Optional: verify nothing's actively running
tmux ls
ps aux | grep python | grep -v grep

# Make sure your work is saved on /workspace, not /tmp or container disk
ls -lh /workspace/results/
ls -lh /workspace/checkpoints/
```

Then **RunPod UI → Terminate pod**. Network volume keeps everything.

Idle cost: ~$0.34/day for the 150 GB volume (just the storage rental).

### Full Restart TL;DR (copy-paste flowchart)

```
1. RunPod UI → Deploy
   - Same DC as network volume
   - Non-Blackwell GPU (L4 / RTX 4090 / A100)
   - PyTorch 2.4.0 template
   - Mount sign2gpt-data at /workspace

2. Web Terminal → check `ls /workspace/Sign2GPT/setup_runpod.sh` works

3. Bootstrap (one big paste, ~10 min):
   apt-get install ... && pip install ... && pip install fasttext ... &&
   pip install spacy model wheel && force-reinstall torch + xformers + scikit-image

4. Verify with `python -c "import torch; print(torch.cuda.get_device_name(0))"`

5. `cd /workspace/Sign2GPT && git pull origin validation/phoenix-12h-run`

6. `tmux new -s work`

7. Run what you came to do (inference / training / etc)

8. Terminate via RunPod UI when done.
```

### Common restart mistakes (avoid these)

| Mistake | Symptom | Fix |
|---|---|---|
| Pod in wrong datacenter | "Cannot attach volume" error at deploy | Pick same DC as `sign2gpt-data` |
| Picked Blackwell GPU | `no kernel image is available` on first CUDA op | Terminate, pick Ada/Ampere/Hopper |
| Forgot tmux, browser closed | Training silently died after closing tab | Always `tmux new -s work` before long-running stuff |
| Skipped force-reinstall of torch | `RuntimeError: NVIDIA driver too old` | Re-run step 5 of bootstrap |
| Skipped scikit-image fix | `numpy.dtype size changed` | Re-run step 7 of bootstrap |
| Did `pip install xformers` without `--no-deps` | Torch gets upgraded, then driver too old | Always `xformers==0.0.27.post2 --no-deps` |
| Used `python -m spacy download` | Broken URL with leading dash | Use direct wheel URL (step 4 of bootstrap) |

---

## END OF HANDOFF DOCUMENT

If you're an AI reading this for the first time: you now have full context. Ask the user what they want to do today; don't waste time re-asking what's already documented here. Refer back to specific sections as needed.

Last updated: 2026-05-31 (after live video inference confirmed working on L4 + RTX 2000 Ada with bug fix journey complete; added Section 17 restart guide)
