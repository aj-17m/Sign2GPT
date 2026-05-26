# PHOENIX-2014-T 12-Hour Validation Run

This branch (`validation/phoenix-12h-run`) is a single-purpose fork of
[Sign2GPT](https://github.com/ryanwongsa/Sign2GPT) configured to train
**end-to-end on PHOENIX-2014-T inside ~12 hours on a single A100 80GB**.

The point is *not* to reproduce paper BLEU. The point is to confirm the
codebase actually trains on real data before adapting it to a custom ISL
dataset.

## What this branch changes vs. upstream

| File | Change |
|---|---|
| `configs/base/base_utils.py` | Paths read from `SIGN2GPT_CKPT_PATH` / `SIGN2GPT_LMDB_PATH` env vars instead of placeholders |
| `environment_variables.py` | Wandb credentials disabled - run uses the text logger only |
| `configs/phoenix2014t/phoenix_stage1_configs/PHX_example_s1_dyn_config.py` | `max_epochs` 100 -> 12, warmup 5 -> 2 |
| `configs/phoenix2014t/phoenix_stage2_configs/PHX_example_s2_dyn_config.py` | `max_epochs` 100 -> 10, warmup 5 -> 2 |
| `scripts/phoenix2014t/image_lmdb_creator.py` | **New.** Converts PHOENIX PNG-frame directories to LMDB (upstream only ships a CSL-Daily `.mp4` converter) |
| `setup_runpod.sh` | **New.** One-shot script that runs the entire pipeline on a RunPod pod |

No model code is touched.

## Pre-flight checklist

1. Register for PHOENIX-2014-T at <https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/>. They email you a download URL within ~1 day.
2. Create a RunPod account at <https://runpod.io>. Add ~$50 credit.
3. Launch a pod:
   - Template: **PyTorch 2.1** (any recent one)
   - GPU: **1x A100 80GB SXM** (~$1.89/hr)
   - Disk: **200 GB** persistent volume mounted at `/workspace`
   - Region: pick one that actually has A100s available

## Run

In the pod terminal:

```bash
cd /workspace
git clone -b validation/phoenix-12h-run https://github.com/aj-17m/Sign2GPT.git
cd Sign2GPT
PHOENIX_URL='https://...your-download-url...' bash setup_runpod.sh
```

That single command runs all 7 phases unattended.

## Expected wall-clock budget

| Phase | Time |
|---|---|
| 1. Python environment | ~10 min |
| 2. PHOENIX download + extract | ~30 min |
| 3. LMDB conversion | ~30 min |
| 4. Pseudo-gloss vocab (incl. fastText download) | ~20 min |
| 5. Stage 1 (12 epochs) | ~4 hours |
| 6. Stage 2 (10 epochs) | ~7 hours |
| **Total** | **~12 hours** |

## Success criteria

After the run, check `/workspace/results/stage{1,2}.log`:

- **Stage 1:** `valid/class_f1_score` should rise from ~0.05 to **>= 0.25**.
- **Stage 2:** `valid/ableu` (real beam-search BLEU, not teacher-forced) should reach **>= 3.0**.
- **Sample outputs** in stage 2 logs should look like coherent German weather sentences, not random tokens.

If all three are true, the pipeline works. We then move on to the custom ISL adaptation. If any fail, debug *before* spending more GPU time.

## Restoring full training

To do a real reproduction later (5-7 days on A100), edit the two config
files and set `max_epochs = 100` and `warmup_epochs = 5`.
