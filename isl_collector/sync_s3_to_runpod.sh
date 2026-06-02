#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sync_s3_to_runpod.sh
#
# Run this on a RunPod GPU pod when you're ready to train the ISL model.
# It pulls all approved videos from your S3 bucket down to /workspace,
# ready for the ISL_TRAINING_GUIDE.md Phase 4 onwards.
#
# Requires: aws CLI (or use rclone), S3 credentials.
#
# Usage:
#     # First time on a new pod, set credentials (one-time per pod):
#     export AWS_ACCESS_KEY_ID=<your-key>
#     export AWS_SECRET_ACCESS_KEY=<your-secret>
#     export AWS_DEFAULT_REGION=us-east-1
#
#     # If using Cloudflare R2 instead of AWS S3, also set:
#     # export S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
#
#     # Then run the sync:
#     bash sync_s3_to_runpod.sh isl-videos /workspace/data/isl/raw_videos
#
# The script syncs all videos. If you only want APPROVED ones, first
# go to your admin dashboard and click "Download ZIP" — that's filtered
# to approved-only and includes the annotations CSV.
# ---------------------------------------------------------------------------

set -euo pipefail

BUCKET="${1:-isl-videos}"
DEST="${2:-/workspace/data/isl/raw_videos}"

echo "[sync] Syncing s3://$BUCKET to $DEST"
mkdir -p "$DEST"

# Install awscli if missing
if ! command -v aws >/dev/null 2>&1; then
    echo "[sync] Installing awscli..."
    pip install -q awscli
fi

# Compose flags
EXTRA=()
if [ -n "${S3_ENDPOINT_URL:-}" ]; then
    EXTRA+=("--endpoint-url" "$S3_ENDPOINT_URL")
fi

# Sync only the submissions/ prefix (skip _state/ which has the DB backup)
aws s3 sync \
    "s3://$BUCKET/submissions/" \
    "$DEST/" \
    "${EXTRA[@]}" \
    --exclude "_state/*"

echo "[sync] Done. Files in $DEST:"
ls -lh "$DEST" | head -20
echo "..."
echo "[sync] Total count:"
find "$DEST" -name "*.mp4" | wc -l
