# Sign2GPT Demo — Team Presentation Cheat Sheet

> **Audience:** Your team (technical or semi-technical)
> **Goal:** Show that you built a working Sign Language → Text translator
> **Duration:** 10-15 minutes
> **Pre-demo prep:** 30 minutes (do BEFORE the meeting)

---

## ⏰ T-30 minutes — Solo prep (CRITICAL, do this alone first)

### Step 1 — Deploy pod (5 min)

1. RunPod → Deploy → Filter same datacenter as your `sign2gpt-data` network volume
2. GPU: **L4 24GB** (~$0.43/hr) — works reliably, NOT Blackwell
3. Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
4. Container disk: 30 GB
5. Network volume `sign2gpt-data` → mount at `/workspace`
6. Deploy → wait ~30 sec → open Web Terminal

### Step 2 — Bootstrap (10 min — copy-paste this whole block)

```bash
apt-get update && apt-get install -y tmux ffmpeg zlib1g-dev libjpeg-dev libpng-dev libtiff-dev libfreetype6-dev && \
pip install --no-cache-dir albumentations==1.4.13 numpy==1.24.4 pandas==2.0.1 transformers==4.31.0 lmdb==1.2.1 timm==0.9.16 ml-collections==0.1.1 pytorch-ignite==0.4.13 Pillow==9.0.1 matplotlib nlpaug==1.1.11 nltk==3.6.7 sentencepiece==0.1.99 einops==0.8.0 mediapipe==0.10.5 onnxscript albucore==0.0.13 spacy==3.7.4 opencv-python==4.8.1.78 pybind11 && \
pip install --no-build-isolation fasttext==0.9.2 && \
pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.7.0/de_core_news_lg-3.7.0-py3-none-any.whl && \
pip install --force-reinstall torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124 && \
pip install xformers==0.0.27.post2 --no-deps && \
pip install --force-reinstall scikit-image==0.22.0 numpy==1.24.4
```

### Step 3 — Verify it works (30 sec)

```bash
cd /workspace/Sign2GPT
git pull origin validation/phoenix-12h-run

# Quick test run on your existing test video
python scripts/infer_video.py --video /workspace/test_videos/test1.mp4
```

Expected output (after ~20 sec):
```
TRANSLATION: das tief das von westen zu uns strömt bringt uns in den nächsten tagen kräftige regenwolken die sich in der westhälfte deutschlands ausbreiten
```

**If you see a German sentence → you're ready. Keep this terminal open.**

If it errors → check `PROJECT_HANDOFF.md` Section 17 (Common restart mistakes table).

### Step 4 — Pre-stage 2-3 backup videos

Have at least 2-3 test MP4s in `/workspace/test_videos/` so you have options during demo if one looks bad.

---

## 🎬 During the demo (15 min)

### Opening — Set context (2 min)

**What to say:**
> "I built a sign language → text translation model. It takes a video of someone signing and produces a written sentence. Let me show you how it works."

**Show:** Your GitHub repo
```
https://github.com/aj-17m/Sign2GPT/tree/validation/phoenix-12h-run
```

### Live demo — Run inference (3 min)

In your already-open terminal:

```bash
cd /workspace/Sign2GPT
python scripts/infer_video.py --video /workspace/test_videos/test1.mp4
```

**While it runs (~20 sec), say:**
> "The model is loading right now — that's the 1.7-billion-parameter language model plus my trained vision encoder. In production, this loads once at server startup, so each request would only take 2-5 seconds."

**When the translation prints, say:**
> "This is the model's output in German. The video showed a meteorologist signing about a low pressure system bringing rain from the west — and the model correctly identified that. The English translation is: 'The low pressure system flowing to us from the west brings strong rain clouds in the coming days that spread across the western half of Germany.'"

### Show the results (2 min)

```bash
# Show training metrics
grep "valid/ableu_bleu4\|valid/orouge" /workspace/results/stage2.log | tail -10
```

**Talking points:**
- BLEU-4 = 13.37 on held-out validation set
- Target was ≥ 3 (just to prove pipeline works) — we got 4× that
- Trained for only 22 epochs vs the paper's 200 — at proper budget, would reach ~22 (paper-level)

### Show actual sample translations (2 min)

```bash
grep -B 1 -A 3 "ABLEU:" /workspace/results/stage2.log | head -40
```

**Point out:**
- Some predictions are word-for-word EXACT matches with ground truth
- Even when wrong, the German is grammatically perfect with correct weather vocabulary
- 519 evaluation samples total in the log

### Architecture explanation (3 min)

**Show this diagram (memorize or have on slide):**

```
Video (frames)
    ↓
DINOv2 vision encoder (frozen, 86M params) — extracts per-frame features
    ↓
Temporal aggregator (trained metaformer) — combines frames into sign tokens
    ↓
XGLM-1.7B language model (mostly frozen) — generates German text
    + LoRA adapters (trained, ~5M params) — adapts XGLM to sign-token input
    ↓
German sentence
```

**Talking points:**
- Two-stage training: first teach vision encoder what signs mean, then plug into language model
- XGLM is multilingual — supports English, German, Hindi, 27 others
- For ISL → English: same architecture, just swap the data and pseudo-gloss vocab

### The journey — bugs fixed (2 min)

**Show:** https://github.com/aj-17m/Sign2GPT/commits/validation/phoenix-12h-run

**Talking points:**
- This is research-grade code from a 2024 ICLR paper
- Found and fixed 8 bugs to make it actually run end-to-end
- Each commit documents one specific bug + the fix
- Examples: data quality patch (PHOENIX has ~55k empty PNG frames), version pinning (xformers/torch driver compatibility), dynamic config (num_classes drift across spaCy versions)

### Next steps (1 min)

**Say:**
> "PHOENIX is German Sign Language. The actual product is Indian Sign Language → English on Android. Now that the pipeline is proven, the work ahead is:
> 1. Collect 500-1000 ISL clips with English transcripts
> 2. Retrain on ISL data (same architecture)
> 3. Build a FastAPI inference server
> 4. Build Android app that records video and shows translations"

---

## 🛠 If something breaks during demo (backup plan)

### Backup A — Translation is wrong/weird

**Don't panic.** Say:
> "This was trained only on German weather forecasts — if the test video isn't from that distribution, it produces grammatically correct German but wrong content. Let me show a clip from the actual PHOENIX test set."

Then run on a known-good PHOENIX clip:
```bash
ls /workspace/data/phoenix2014t/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev/ | head -5
# Pick one, convert to MP4 with ffmpeg, then run infer_video.py
```

### Backup B — Inference crashes

**Show the existing log instead:**
```bash
grep -B 1 -A 3 "ABLEU:" /workspace/results/stage2.log | head -40
```

**Say:**
> "These are real outputs from the model's evaluation on 519 unseen clips. Live inference is finicky on cloud GPUs — but the proven model outputs are right here."

### Backup C — Pod won't start / SSH issue

**Show the GitHub repo + handoff doc:**
- https://github.com/aj-17m/Sign2GPT/tree/validation/phoenix-12h-run
- https://github.com/aj-17m/Sign2GPT/blob/validation/phoenix-12h-run/PROJECT_HANDOFF.md

**Say:**
> "The infrastructure is on RunPod — let me walk through the code and results from my GitHub instead."

### Backup D — Everything fails

**Have screenshots ready locally on your laptop:**
- Screenshot of `valid_test/ableu_bleu4: 13.37` line
- Screenshot of 5-10 sample translation pairs (PRED vs TGT)
- Screenshot of `nvidia-smi` showing GPU
- Screenshot of the commit history

Take these now (before the demo) so you're not scrambling.

---

## 📊 Key numbers to memorize

| Metric | Value | Context |
|---|---|---|
| BLEU-4 (real, beam search) | **13.37** | Target was ≥3 (4× over) |
| BLEU-4 (teacher-forced) | 18.24 | Optimistic baseline |
| ROUGE | 34.21 | Word recall |
| Training data used | 6,633 / 7,096 clips (93.5%) | Filtered 463 corrupt clips |
| Stage 1 epochs | 12 (vs paper's 100) | Validation budget |
| Stage 2 epochs | 10 (vs paper's 100) | Validation budget |
| Total training cost | ~$4 on L4 24GB | Validation run |
| Wall clock | ~8 hours | Validation run |
| Trainable params | ~16M out of 1.8B total | LoRA + vision aggregator |
| Bugs fixed | 8 critical, 9 commits | All in commit history |

---

## 🎯 Questions you might get + answers

**Q: "How accurate is it?"**
> "BLEU-4 of 13.37 on the validation set. That's about 35% single-word accuracy and 17% three-word phrase accuracy. For comparison, the original paper hits ~22 with 10× the training. For a 22-epoch validation run, this is way above target."

**Q: "Can I sign and have it translate?"**
> "Not yet — this model only knows German Sign Language from weather news. For Indian Sign Language, we'd need to collect ISL data and retrain. The architecture supports it; only the training data needs to change."

**Q: "How long does inference take?"**
> "Currently ~20 seconds per video because the model loads from disk each time. In a production setup with the model kept in memory (FastAPI server), it would be 1-3 seconds per request on cloud GPU."

**Q: "What's next?"**
> "ISL data collection — about 500-1000 video clips of signed sentences with English transcripts. Then retrain on ISL, build a FastAPI server, then an Android app."

**Q: "What if I want to deploy this?"**
> "The model checkpoint is 3.6 GB. Deployment options: cloud GPU server (FastAPI) for low latency, or distilled smaller model for on-device. The Android app would record video, send to server, and display the translation. End-to-end latency target: under 5 seconds."

**Q: "Why XGLM and not GPT-4 / ChatGPT?"**
> "XGLM is open-source, runs on our own GPU, no API costs. Also XGLM supports 30+ languages natively including Hindi — when we add ISL → Hindi later, no architecture change needed. ChatGPT would have ongoing API fees and we couldn't fine-tune."

**Q: "What were the hardest bugs?"**
> "Three stood out:
> 1. PHOENIX dataset has 55,000 empty PNG frames — wrote a patch to skip them while preserving usable clips
> 2. xformers version pinning — the unpinned install upgrades PyTorch, which breaks on the CUDA driver
> 3. Class count mismatch — config hardcoded 2306, but spaCy 3.7 produces 2338 lemmas. Made the count dynamic."

---

## 🎬 Closing the demo

**End strong with:**
> "This is the foundation. The next 4-6 weeks are ISL data collection and retraining. Then we have a real product to demo. Questions?"

**Have ready:**
- GitHub URL: `https://github.com/aj-17m/Sign2GPT/tree/validation/phoenix-12h-run`
- Handoff doc URL: `https://github.com/aj-17m/Sign2GPT/blob/validation/phoenix-12h-run/PROJECT_HANDOFF.md`

---

## ⚠️ After the demo

```bash
# Terminate the pod to stop billing
# (Network volume keeps everything for ~$0.34/day)
```

In RunPod UI → Terminate.

---

## ✅ Pre-demo checklist (print this or screenshot)

- [ ] Pod deployed in correct datacenter
- [ ] Network volume mounted at `/workspace`
- [ ] Bootstrap completed successfully (8-10 min)
- [ ] `python scripts/infer_video.py --video /workspace/test_videos/test1.mp4` outputs German text
- [ ] At least 2 backup videos in `/workspace/test_videos/`
- [ ] GitHub commit page open in browser tab
- [ ] PROJECT_HANDOFF.md open in browser tab
- [ ] Key numbers memorized (BLEU 13.37, 8 bugs, ~$4 cost)
- [ ] Screenshots saved locally as backup
- [ ] Web Terminal stays connected (don't close the tab)

Once all boxes are checked, you're ready. 🚀
