#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_runpod.sh - Sign2GPT PHOENIX-2014-T 12-hour validation run on RunPod
# ---------------------------------------------------------------------------
# Usage:
#   1. Launch a RunPod A100 80GB pod with the PyTorch 2.1 template.
#   2. SSH into the pod (or use the web terminal).
#   3. Run:
#        cd /workspace
#        git clone -b validation/phoenix-12h-run https://github.com/aj-17m/Sign2GPT.git
#        cd Sign2GPT
#        bash setup_runpod.sh
#
# The script is idempotent - if a stage already completed, it skips.
# Total wall-clock target: ~12 hours (1h setup + 4h stage 1 + 7h stage 2).
# ---------------------------------------------------------------------------

set -euo pipefail

# ----- Paths (RunPod convention: persistent storage under /workspace) -----
export SIGN2GPT_ROOT=${SIGN2GPT_ROOT:-/workspace/Sign2GPT}
export SIGN2GPT_DATA=${SIGN2GPT_DATA:-/workspace/data}
export SIGN2GPT_CKPT_PATH=${SIGN2GPT_CKPT_PATH:-/workspace/checkpoints}
export SIGN2GPT_LMDB_PATH=${SIGN2GPT_LMDB_PATH:-/workspace/lmdb}
export SIGN2GPT_RESULTS=${SIGN2GPT_RESULTS:-/workspace/results}

PHOENIX_RAW="${SIGN2GPT_DATA}/phoenix2014t/PHOENIX-2014-T-release-v3"
PHOENIX_FRAMES="${PHOENIX_RAW}/PHOENIX-2014-T/features/fullFrame-210x260px"
PHOENIX_ANN="${PHOENIX_RAW}/PHOENIX-2014-T/annotations/manual"
PHX_PKL="${SIGN2GPT_ROOT}/data/phoenix2014t/processed_words.phx_pkl"

mkdir -p "${SIGN2GPT_DATA}" "${SIGN2GPT_CKPT_PATH}" "${SIGN2GPT_LMDB_PATH}" "${SIGN2GPT_RESULTS}"

log() { echo -e "\n\033[1;36m[setup_runpod] $*\033[0m"; }

# ===========================================================================
# Phase 1 - Python environment
# ===========================================================================
log "Phase 1/7: Python environment"

# RunPod's PyTorch template ships with conda already installed. We install
# pip deps directly into the base env to avoid the time cost of resolving
# a fresh conda env.
pip install --no-cache-dir \
    albumentations==1.4.13 \
    numpy==1.24.4 \
    pandas==2.0.1 \
    transformers==4.31.0 \
    lmdb==1.2.1 \
    timm==0.9.16 \
    requests \
    ml-collections==0.1.1 \
    pytorch-ignite==0.4.13 \
    Pillow==9.0.1 \
    matplotlib \
    nlpaug==1.1.11 \
    nltk==3.6.7 \
    fasttext==0.9.2 \
    sentencepiece==0.1.99 \
    einops==0.8.0 \
    mediapipe==0.10.5 \
    onnxscript \
    albucore==0.0.13 \
    spacy==3.7.4 \
    opencv-python==4.8.1.78

# xformers (required by models/metaformer/emb/sine_pos.py for positional
# embeddings - imports `xformers.components.positional_embedding`).
# Not in the upstream requirements list; missing it causes training to crash
# at model init with `ModuleNotFoundError: No module named 'xformers'`.
# Letting pip pick the version that matches the installed torch/CUDA.
pip install --no-cache-dir xformers

# spaCy German model (used by pseudo_gloss_de.py)
# Note: `python -m spacy download` builds a malformed URL on some RunPod pod
# templates (compatibility.json lookup leaves the version field empty,
# producing /-de_core_news_lg/-de_core_news_lg.tar.gz). Install the wheel
# from the direct URL to avoid the lookup entirely.
pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.7.0/de_core_news_lg-3.7.0-py3-none-any.whl

# ===========================================================================
# Phase 2 - PHOENIX-2014-T dataset download
# ===========================================================================
log "Phase 2/7: PHOENIX-2014-T download"

if [ ! -d "${PHOENIX_FRAMES}/train" ]; then
    if [ -z "${PHOENIX_URL:-}" ]; then
        echo "Set PHOENIX_URL to the tar.gz URL you got after registering at"
        echo "  https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/"
        echo "Then re-run:  PHOENIX_URL='https://...' bash setup_runpod.sh"
        exit 1
    fi
    mkdir -p "${SIGN2GPT_DATA}/phoenix2014t"
    cd "${SIGN2GPT_DATA}/phoenix2014t"
    log "Downloading PHOENIX-2014-T (~35 GB)..."
    wget -q --show-progress -O phoenix.tar.gz "${PHOENIX_URL}"
    log "Extracting..."
    tar -xzf phoenix.tar.gz
    rm phoenix.tar.gz
else
    log "PHOENIX frames already present, skipping download"
fi

# Sign2GPT's configs expect the corpus CSVs at
# `${SIGN2GPT_ROOT}/data/phoenix2014t/PHOENIX-2014-T.{train,dev,test}.corpus.csv`.
# PHOENIX ships them under `annotations/manual/`, so symlink them.
mkdir -p "${SIGN2GPT_ROOT}/data/phoenix2014t"
for split in train dev test; do
    src="${PHOENIX_ANN}/PHOENIX-2014-T.${split}.corpus.csv"
    dst="${SIGN2GPT_ROOT}/data/phoenix2014t/PHOENIX-2014-T.${split}.corpus.csv"
    if [ ! -e "${dst}" ]; then
        ln -s "${src}" "${dst}"
    fi
done

# ===========================================================================
# Phase 3 - LMDB conversion
# ===========================================================================
log "Phase 3/7: LMDB conversion"

# Always invoke the converter. A wrapper-level "skip if non-empty" check
# incorrectly skips when a prior run was interrupted mid-conversion (e.g.,
# tmux died after only ~80 of ~8k clips were done). The image_lmdb_creator
# itself has per-clip idempotency (see convert_clip line ~67), so re-running
# is cheap: already-converted clips return "skipped" without re-reading the
# PNGs.
cd "${SIGN2GPT_ROOT}"
python scripts/phoenix2014t/image_lmdb_creator.py \
    --frames_root "${PHOENIX_FRAMES}" \
    --lmdb_root "${SIGN2GPT_LMDB_PATH}/phoenix2014t/lmdb_videos" \
    --csv_dir "${SIGN2GPT_ROOT}/data/phoenix2014t" \
    --all_splits

# ===========================================================================
# Phase 4 - Pseudo-gloss vocabulary
# ===========================================================================
log "Phase 4/7: pseudo-gloss vocabulary"

if [ ! -f "${PHX_PKL}" ]; then
    cd "${SIGN2GPT_ROOT}"
    python scripts/pseudo_gloss_de.py
else
    log "Pseudo-gloss pkl already exists, skipping"
fi

# ===========================================================================
# Phase 5 - Stage 1 pretraining (~4 hours on A100 80GB)
# ===========================================================================
log "Phase 5/7: stage 1 pretraining (target ~4h)"

cd "${SIGN2GPT_ROOT}"
STAGE1_LOG="${SIGN2GPT_RESULTS}/stage1.log"
python main.py \
    --config=configs/phoenix2014t/phoenix_stage1_configs/PHX_example_s1_dyn_config.py \
    2>&1 | tee "${STAGE1_LOG}"

# ===========================================================================
# Phase 6 - Stage 2 translation training (~7 hours on A100 80GB)
# ===========================================================================
log "Phase 6/7: stage 2 translation training (target ~7h)"

STAGE2_LOG="${SIGN2GPT_RESULTS}/stage2.log"
python main.py \
    --config=configs/phoenix2014t/phoenix_stage2_configs/PHX_example_s2_dyn_config.py \
    2>&1 | tee "${STAGE2_LOG}"

# ===========================================================================
# Phase 7 - Summary
# ===========================================================================
log "Phase 7/7: summary"

echo "=================================================================="
echo "  Validation run finished."
echo "  Stage 1 log: ${STAGE1_LOG}"
echo "  Stage 2 log: ${STAGE2_LOG}"
echo "  Checkpoints: ${SIGN2GPT_CKPT_PATH}"
echo
echo "  Sanity-check the run by extracting metrics:"
echo "    grep 'valid/class_f1_score' ${STAGE1_LOG} | tail -5"
echo "    grep 'valid/obleu'          ${STAGE2_LOG} | tail -5"
echo "    grep 'valid/ableu'          ${STAGE2_LOG} | tail -5"
echo
echo "  Pass conditions (pipeline validated):"
echo "    - stage 1 valid/class_f1_score >= 0.25"
echo "    - stage 2 valid/ableu          >= 3.0"
echo "=================================================================="
