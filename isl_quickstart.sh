#!/usr/bin/env bash
# ============================================================================
# ISL Training Quickstart — one command to go from fresh pod to training.
#
# Run on a RunPod GPU pod with the ISL network volume (p014akuq8i) mounted
# at /workspace. Just paste this single line in the pod's Web Terminal:
#
#   curl -sSL https://raw.githubusercontent.com/aj-17m/Sign2GPT/validation/phoenix-12h-run/isl_quickstart.sh | bash
#
# What it does:
#   1. Verifies GPU + network volume are present
#   2. Installs ALL system + Python dependencies
#   3. Clones the Sign2GPT repo into /workspace if missing (or pulls latest)
#   4. Counts submissions and warns if too few for useful training
#   5. Launches run_isl_training.py inside a tmux session so it survives
#      browser tab closes / disconnects
#
# Idempotent: safe to re-run. Steps 2 and 3 skip if already done.
# ============================================================================

set -e  # exit immediately on any error

# ---- Pretty output helpers ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

banner() { echo -e "\n${BLUE}${BOLD}═════ $* ═════${NC}"; }
ok()     { echo -e "${GREEN}✓${NC} $*"; }
warn()   { echo -e "${YELLOW}⚠${NC} $*"; }
err()    { echo -e "${RED}✗ $*${NC}"; exit 1; }

banner "ISL Training Quickstart"
echo "This will set up everything and start training."
echo ""

# ============================================================================
# Step 0 — Sanity checks
# ============================================================================

banner "Step 0/4 — Sanity checks"

if ! command -v nvidia-smi > /dev/null 2>&1; then
    err "No NVIDIA GPU detected. Deploy a GPU pod (A100 PCIe or L4 recommended)."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "GPU detected: $GPU_NAME"

# Reject Blackwell GPUs (compute 12.x) - they don't work with torch 2.4
COMPUTE_CAP=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0))" 2>/dev/null || echo "(0,0)")
if echo "$COMPUTE_CAP" | grep -q "12,"; then
    err "Blackwell GPU detected ($COMPUTE_CAP). Use A100, RTX 4090, L4, or older. Torch 2.4 doesn't support Blackwell."
fi
ok "Compute capability OK: $COMPUTE_CAP"

if [ ! -d "/workspace" ]; then
    err "/workspace not mounted. Mount the ISL network volume (p014akuq8i) at /workspace."
fi
ok "/workspace is mounted"

if [ ! -d "/workspace/submissions" ]; then
    warn "No /workspace/submissions yet — has anyone uploaded a video?"
    echo "    Upload at least 1 video via the web app first, then re-run this script."
    err "Aborting: nothing to train on."
fi

N_VIDEOS=$(ls /workspace/submissions/ 2>/dev/null | wc -l)
ok "Found $N_VIDEOS uploaded video(s) in /workspace/submissions/"

if [ "$N_VIDEOS" -lt 10 ]; then
    warn "Only $N_VIDEOS videos uploaded — training will produce poor results."
    warn "Recommend at least 50+ for any useful signal, 500+ for a working demo."
    read -p "Continue anyway? (y/N) " confirm
    [ "$confirm" = "y" ] || err "Aborted by user."
fi

# ============================================================================
# Step 1 — Install all dependencies (idempotent)
# ============================================================================

banner "Step 1/4 — Install dependencies (~5-10 minutes)"

if python3 -c "import torch, xformers, ml_collections, pandas, lmdb, ignite, transformers, albumentations, spacy" > /dev/null 2>&1; then
    ok "All Python deps already installed, skipping."
else
    echo "Installing system packages..."
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        tmux ffmpeg sqlite3 \
        zlib1g-dev libjpeg-dev libpng-dev libtiff-dev libfreetype6-dev \
        > /dev/null 2>&1
    ok "System packages installed"

    echo "Installing Python packages (this takes ~5-7 minutes)..."
    pip install --no-cache-dir -q \
        albumentations==1.4.13 numpy==1.24.4 pandas==2.0.1 \
        transformers==4.31.0 lmdb==1.2.1 timm==0.9.16 \
        ml-collections==0.1.1 pytorch-ignite==0.4.13 Pillow==9.0.1 \
        matplotlib nlpaug==1.1.11 nltk==3.6.7 sentencepiece==0.1.99 \
        einops==0.8.0 mediapipe==0.10.5 onnxscript albucore==0.0.13 \
        spacy==3.7.4 opencv-python==4.8.1.78 pybind11 \
        > /dev/null 2>&1
    ok "Main Python packages installed"

    echo "Installing fasttext..."
    pip install -q --no-build-isolation fasttext==0.9.2 > /dev/null 2>&1
    ok "fasttext installed"

    echo "Pinning torch to 2.4.0 (some deps may have upgraded it)..."
    pip install -q --force-reinstall \
        torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
        > /dev/null 2>&1
    pip install -q xformers==0.0.27.post2 --no-deps > /dev/null 2>&1
    ok "torch + xformers pinned"

    echo "Fixing scikit-image ABI..."
    pip install -q --force-reinstall scikit-image==0.22.0 numpy==1.24.4 > /dev/null 2>&1
    ok "scikit-image fixed"
fi

# Quick verify
python3 -c "import torch; assert torch.cuda.is_available()" \
    && ok "CUDA verified" \
    || err "CUDA not working after install"

# ============================================================================
# Step 2 — Get the repo
# ============================================================================

banner "Step 2/4 — Set up code"

if [ ! -d "/workspace/Sign2GPT" ]; then
    cd /workspace
    git clone -b validation/phoenix-12h-run https://github.com/aj-17m/Sign2GPT.git 2>&1 | tail -3
    ok "Repo cloned to /workspace/Sign2GPT"
else
    cd /workspace/Sign2GPT
    git pull origin validation/phoenix-12h-run 2>&1 | tail -3
    ok "Repo updated"
fi

cd /workspace/Sign2GPT

# ============================================================================
# Step 3 — Install English spaCy model (needed for pseudo-gloss)
# ============================================================================

banner "Step 3/4 — Install English language model"

if python3 -c "import spacy; spacy.load('en_core_web_lg')" > /dev/null 2>&1; then
    ok "en_core_web_lg already installed"
else
    echo "Downloading en_core_web_lg (~500 MB, takes 1-2 min)..."
    pip install -q https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl > /dev/null 2>&1
    ok "en_core_web_lg installed"
fi

# ============================================================================
# Step 4 — Launch training in tmux
# ============================================================================

banner "Step 4/4 — Launch training"

# Kill any old session
tmux kill-session -t isl_train 2>/dev/null || true

# Decide what to run based on flags
SCRIPT_ARGS="${ISL_TRAIN_ARGS:-}"
if [ -n "$SCRIPT_ARGS" ]; then
    echo "Custom args: $SCRIPT_ARGS"
fi

# Create new tmux session that runs the training in the background
tmux new-session -d -s isl_train \
    "cd /workspace/Sign2GPT && python run_isl_training.py $SCRIPT_ARGS 2>&1 | tee /workspace/isl_training_$(date +%Y%m%d_%H%M%S).log"

# Give it a couple seconds to actually start
sleep 3

if tmux has-session -t isl_train 2>/dev/null; then
    ok "Training started in tmux session 'isl_train'"
    echo ""
    echo -e "${GREEN}${BOLD}🎉 Training is running in the background.${NC}"
    echo ""
    echo "  ${BOLD}Watch live:${NC}      tmux attach -t isl_train"
    echo "  ${BOLD}Detach again:${NC}    Ctrl+B then D"
    echo "  ${BOLD}Check log:${NC}       tail -f /workspace/isl_training_*.log"
    echo "  ${BOLD}Kill training:${NC}   tmux kill-session -t isl_train"
    echo ""
    echo "Estimated time:"
    if [ "$N_VIDEOS" -lt 100 ]; then
        echo "  ~30-60 minutes (small dataset, quick run)"
    elif [ "$N_VIDEOS" -lt 500 ]; then
        echo "  ~3-5 hours"
    elif [ "$N_VIDEOS" -lt 1000 ]; then
        echo "  ~6-10 hours"
    else
        echo "  ~12+ hours"
    fi
    echo ""
    echo -e "${YELLOW}Remember to TERMINATE the pod when training finishes${NC}"
    echo -e "${YELLOW}to stop GPU billing.${NC}"
else
    err "Training session failed to start. Check 'tmux ls'."
fi
