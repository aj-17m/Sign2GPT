"""
Run the trained stage 2 model on an arbitrary video file (MP4, MOV, etc.).

Builds the model directly (no Trainer machinery) and replicates the exact
preprocessing pipeline the dataloader uses on PHOENIX clips, so the inputs
match the training distribution as closely as possible.

IMPORTANT: this model was trained on German Sign Language (DGS) weather
forecasts from PHOENIX-2014T. It will produce:
  - Decent German text for similar DGS clips (studio, weather, single signer)
  - Garbled German for other DGS topics or different signers
  - Random German for ISL / ASL / other sign languages

Usage:
    python scripts/infer_video.py --video /workspace/test_videos/my_clip.mp4
    python scripts/infer_video.py --video /workspace/test_videos/my_clip.mp4 --num_beams 4
"""

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")


REPO_ROOT = "/workspace/Sign2GPT"
CKPT_PATH = "/workspace/checkpoints/phoenix_stage2_configs/PHX_example_s2_dyn_config/best_result_checkpoint_10_18.2415.pt"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def read_and_resample_video(video_path, target_fps=25, max_frames=256, resize=(256, 256)):
    """Read a video file, sample to target_fps, resize each frame to `resize`.

    Returns a list of numpy arrays (HWC uint8 RGB).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {video_path}: {total} frames at {src_fps:.1f} fps")

    # Read all frames first (simpler than seeking)
    raw_frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        raw_frames.append(bgr)
    cap.release()

    # Subsample to target_fps
    if src_fps > target_fps + 1:
        step = src_fps / target_fps
        indices = np.arange(0, len(raw_frames), step).astype(int)
        raw_frames = [raw_frames[i] for i in indices if i < len(raw_frames)]
        print(f"[video] Subsampled {src_fps:.1f}fps -> {target_fps}fps: {len(raw_frames)} frames")

    # Cap total frame count (training used max_seq_len=256)
    if len(raw_frames) > max_frames:
        indices = np.linspace(0, len(raw_frames) - 1, max_frames).astype(int)
        raw_frames = [raw_frames[i] for i in indices]
        print(f"[video] Capped to {max_frames} frames (uniform subsample)")

    # BGR -> RGB, resize
    out = []
    for f in raw_frames:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, resize, interpolation=cv2.INTER_LINEAR)
        out.append(f)

    print(f"[video] Final: {len(out)} frames at {resize[0]}x{resize[1]} RGB")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file (MP4, MOV, ...)")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--target_fps", type=int, default=25, help="Resample input to this fps")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"[err] Video not found: {args.video}")
        sys.exit(1)

    # ---- Config + tokenizer ----
    from configs.phoenix2014t.phoenix_stage2_configs.PHX_example_s2_dyn_config import get_config
    from augmentation.video.base_video_aug import Transformation
    from transformers import AutoTokenizer

    print("[infer] Loading config + tokenizer...")
    cfg = get_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_name)

    # ---- Model ----
    print("[infer] Building model architecture...")
    mod = importlib.import_module(cfg.model_name)

    # pretext_length is set by the trainer based on cfg.pretext; for empty
    # pretext (our case) it's 1 (the BOS/EOS token)
    pretext = cfg.pretext if "pretext" in cfg else ""
    pretext_tokens = tokenizer(pretext)["input_ids"]
    pretext_length = len(pretext_tokens) if pretext else 1
    print(f"[infer] pretext='{pretext}', pretext_length={pretext_length}")

    model_params = dict(cfg.model_params)
    model_params["pretext_length"] = pretext_length
    model = mod.Model(**model_params)

    # ---- Load checkpoint ----
    print(f"[infer] Loading checkpoint from {CKPT_PATH} (~10 sec)...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[infer] WARNING: {len(missing)} missing keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"[infer] WARNING: {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")

    model.eval().cuda()

    # ---- Preprocess video ----
    print(f"[infer] Reading video...")
    raw_frames = read_and_resample_video(
        args.video, target_fps=args.target_fps, max_frames=cfg.aug_params.get("max_seq_len", 256)
    )

    print("[infer] Applying validation transform...")
    transform = Transformation(**cfg.aug_params)
    frames_tensor = transform.aug_video(raw_frames, isValid=True)
    print(f"[infer] Tensor shape: {tuple(frames_tensor.shape)}")  # (N, C, H, W)

    # ---- Build model input (matches trainer.prep_batch format for batch=1) ----
    frame_features = [frames_tensor.cuda()]  # list of one tensor (batch dim)
    frame_mask = torch.ones(1, frames_tensor.shape[0], dtype=torch.bool).cuda()
    text_ids = torch.tensor([pretext_tokens]).cuda()
    text_mask = torch.ones(1, len(pretext_tokens), dtype=torch.bool).cuda()
    max_len = torch.tensor(cfg.aug_params.get("max_seq_len", 256)).cuda()

    # ---- Generate ----
    print(f"[infer] Generating translation (beams={args.num_beams})...")
    with torch.inference_mode(True), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            text_ids=text_ids,
            text_mask=text_mask,
            frame_features=frame_features,
            frame_mask=frame_mask,
            max_len=max_len,
            generate=True,
            gen_params={
                "max_length": args.max_length,
                "num_beams": args.num_beams,
                "eos_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            },
        )

    output_ids = out["output_ids"]
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 70)
    print(f"VIDEO:        {args.video}")
    print(f"TRANSLATION:  {pred}")
    print("=" * 70)
    print()
    print("Note: model trained on German Sign Language weather forecasts.")
    print("      Quality depends on how similar your video is to PHOENIX studio clips.")


if __name__ == "__main__":
    main()
