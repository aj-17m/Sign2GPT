"""
Quick inference script — load the trained stage 2 checkpoint and translate
a single PHOENIX clip from the LMDB store.

Usage from /workspace/Sign2GPT:
    python scripts/infer_clip.py --clip 01April_2010_Thursday_heute-6694
    python scripts/infer_clip.py --clip 01April_2010_Thursday_heute-6694 --num_beams 4

To list available clips:
    python scripts/infer_clip.py --list train | head -20
    python scripts/infer_clip.py --list dev | head -20
    python scripts/infer_clip.py --list test | head -20

To pick a random clip:
    python scripts/infer_clip.py --random dev
"""

import argparse
import importlib
import os
import pickle
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths (adjust if you mount the repo somewhere else)
# ---------------------------------------------------------------------------
REPO_ROOT = "/workspace/Sign2GPT"
CKPT_PATH = "/workspace/checkpoints/phoenix_stage2_configs/PHX_example_s2_dyn_config/best_result_checkpoint_10_18.2415.pt"
LMDB_ROOT = "/workspace/lmdb/phoenix2014t/lmdb_videos"
CSV_DIR = f"{REPO_ROOT}/data/phoenix2014t"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.phoenix2014t.phoenix_stage2_configs.PHX_example_s2_dyn_config import (
    get_config,
)
from dataloaders.data_utils.lmdb_utils import LMDBUtility
from augmentation.video.base_video_aug import build_transform
from transformers import AutoTokenizer


def find_clip_split(clip_name: str) -> str:
    """Return which split a clip belongs to by checking the CSVs."""
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(f"{CSV_DIR}/PHOENIX-2014-T.{split}.corpus.csv", sep="|")
        if clip_name in set(df["name"]):
            return split
    raise ValueError(f"Clip {clip_name} not found in any CSV split.")


def list_clips(split: str):
    df = pd.read_csv(f"{CSV_DIR}/PHOENIX-2014-T.{split}.corpus.csv", sep="|")
    for name in df["name"]:
        if os.path.isdir(f"{LMDB_ROOT}/{name}"):
            print(name)


def get_ground_truth(clip_name: str, split: str) -> str:
    df = pd.read_csv(f"{CSV_DIR}/PHOENIX-2014-T.{split}.corpus.csv", sep="|")
    row = df[df["name"] == clip_name].iloc[0]
    return row["translation"]


def load_clip_frames(clip_name: str, transform):
    """Load and preprocess all frames of a clip into a model-ready tensor."""
    lmdb_path = f"{LMDB_ROOT}/{clip_name}"
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f"No LMDB for {clip_name} (likely rejected during conversion)")

    util = LMDBUtility(lmdb_path)
    num_frames = util.details["num_frames"]

    # Match the validation-time frame sampling logic from
    # dataloaders/phoenix_video_dataset.py
    selection = np.arange(0, num_frames, transform.stride).astype(int)
    if len(selection) > transform.max_seq_len:
        idx = np.linspace(0, len(selection) - 1, transform.max_seq_len).astype(int)
        selection = selection[idx]

    frames = util.get_frames(selection)
    frames = transform.aug_video(frames, isValid=True)
    return frames


def load_model(cfg):
    print("[infer] Loading model architecture...")
    mod = importlib.import_module(cfg.model_name)
    model = mod.Model(**dict(cfg.model_params))

    print(f"[infer] Loading checkpoint from {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # Checkpoint may be a dict {"model": state_dict, "epoch": N, ...} or just the state_dict
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[infer] WARNING: {len(missing)} missing keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"[infer] WARNING: {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")

    model.eval().cuda()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", help="Specific PHOENIX clip name to translate")
    parser.add_argument("--random", choices=["train", "dev", "test"], help="Pick a random clip from a split")
    parser.add_argument("--list", choices=["train", "dev", "test"], help="List all clips in a split")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    if args.list:
        list_clips(args.list)
        return

    # Pick the clip name
    if args.random:
        df = pd.read_csv(f"{CSV_DIR}/PHOENIX-2014-T.{args.random}.corpus.csv", sep="|")
        available = [n for n in df["name"] if os.path.isdir(f"{LMDB_ROOT}/{n}")]
        clip_name = random.choice(available)
        split = args.random
        print(f"[infer] Randomly selected: {clip_name} (from {split} split)")
    elif args.clip:
        clip_name = args.clip
        split = find_clip_split(clip_name)
        print(f"[infer] Clip {clip_name} found in {split} split")
    else:
        parser.error("Must pass --clip, --random, or --list")

    # Load config + model + tokenizer
    cfg = get_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_name)
    model = load_model(cfg)

    # Build the validation-time augmentation/transform
    transform = build_transform(cfg.aug_params)

    # Load + preprocess the clip
    print(f"[infer] Loading frames for {clip_name}...")
    frames = load_clip_frames(clip_name, transform)
    if isinstance(frames, list):
        frames = torch.tensor(np.stack(frames)).float()
    print(f"[infer] Loaded {frames.shape[0]} frames, shape={tuple(frames.shape)}")

    # Move to GPU
    frames_batch = frames.unsqueeze(0).cuda()  # (1, T, C, H, W)
    frame_mask = torch.ones(1, frames.shape[0], dtype=torch.bool).cuda()

    # Generate
    print(f"[infer] Generating translation (beams={args.num_beams})...")
    with torch.inference_mode(True), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # The model expects: frames, frame_mask, plus generation params
        # We approximate the trainer's test_step call
        try:
            out = model(
                frames=frames_batch,
                frame_mask=frame_mask,
                gen_params={
                    "max_length": args.max_length,
                    "num_beams": args.num_beams,
                    "eos_token_id": tokenizer.eos_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                },
                generate=True,
            )
            output_ids = out["output_ids"] if isinstance(out, dict) else out
        except Exception as e:
            print(f"[infer] Generation failed: {e}")
            print("[infer] This script is a thin wrapper - the trainer's exact")
            print("       calling convention may need a more careful port.")
            print("       Check the test_step method in trainer/complete_translation_trainer.py")
            raise

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    gt = get_ground_truth(clip_name, split)

    print("\n" + "=" * 70)
    print(f"Clip:       {clip_name}")
    print(f"Split:      {split}")
    print(f"PREDICTION: {pred}")
    print(f"GROUND TRUTH: {gt}")
    print("=" * 70)


if __name__ == "__main__":
    main()
