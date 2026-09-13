#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

BASE_RUNTIME="${DISK_ENGRAM_BASE_IMAGE:-$IMAGE}"
DISK_IMAGE="${DISK_ENGRAM_IMAGE:-deepseek-v41-exl3:disk-engram}"

if ! docker image inspect "$BASE_RUNTIME" >/dev/null 2>&1; then
  echo "ERROR: locked base runtime '$BASE_RUNTIME' is not present." >&2
  echo "Run: bash scripts/build_runtime.sh" >&2
  exit 2
fi

echo "=== Build disk-backed Engram qualification runtime ==="
echo "Base runtime: $BASE_RUNTIME"
echo "Output image: $DISK_IMAGE"

docker build \
  -f "$RECIPE_ROOT/Dockerfile.disk-engram" \
  --build-arg BASE_RUNTIME="$BASE_RUNTIME" \
  -t "$DISK_IMAGE" \
  "$RECIPE_ROOT"

docker image inspect "$DISK_IMAGE" >/dev/null

echo
echo "Built: $DISK_IMAGE"
echo "Use on every Spark:"
echo "  IMAGE=$DISK_IMAGE VLLM_ENGRAM_DISK_BACKED=1 VLLM_ENGRAM_MODEL_DIR=/models/<snapshot> ... bash scripts/start_cluster.sh head|worker"
