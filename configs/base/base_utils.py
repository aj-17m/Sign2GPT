import os
from pathlib import Path


# Paths are taken from environment variables so the same code works on
# RunPod (/workspace/...) and on a local Windows machine (D:/Sign2GPT/...).
#
# On RunPod, setup_runpod.sh exports:
#   SIGN2GPT_CKPT_PATH=/workspace/checkpoints
#   SIGN2GPT_LMDB_PATH=/workspace/lmdb
#
# On Windows local dev, set them in PowerShell:
#   $env:SIGN2GPT_CKPT_PATH = "D:/Sign2GPT/checkpoints"
#   $env:SIGN2GPT_LMDB_PATH = "D:/Sign2GPT/lmdb"


def get_checkpoint_path(base_name, name):
    ckpt_path = os.environ.get(
        "SIGN2GPT_CKPT_PATH",
        "/workspace/checkpoints",
    )
    return ckpt_path


def get_lmdb_path():
    lmdb_path = os.environ.get(
        "SIGN2GPT_LMDB_PATH",
        "/workspace/lmdb",
    )
    return lmdb_path
