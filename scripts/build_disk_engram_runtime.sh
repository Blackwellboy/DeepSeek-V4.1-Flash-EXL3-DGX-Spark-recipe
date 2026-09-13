#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

BASE_RUNTIME="${DISK_ENGRAM_BASE_IMAGE:-$IMAGE}"
DISK_IMAGE="${DISK_ENGRAM_IMAGE:-deepseek-v41-exl3:disk-engram}"
LOCKED_PLUGIN="$(python3 "$LOCK_TOOL" get vllm_exl3.commit)"
LOCKED_EXLLAMA="$(python3 "$LOCK_TOOL" get exllamav3.commit)"

if ! docker image inspect "$BASE_RUNTIME" >/dev/null 2>&1; then
  echo "ERROR: locked base runtime '$BASE_RUNTIME' is not present." >&2
  echo "Run: bash scripts/build_runtime.sh" >&2
  exit 2
fi

# Do not derive a disk runtime from a stale/custom tag merely because its local
# image name matches. The baseline Dockerfile retains both source checkouts so
# their exact commits can be verified without trusting image labels.
docker run --rm --entrypoint bash "$BASE_RUNTIME" -lc '
  set -euo pipefail
  test -d /opt/vllm-exl3/.git
  test -d /opt/exllamav3/.git
  plugin="$(git -C /opt/vllm-exl3 rev-parse HEAD)"
  exllama="$(git -C /opt/exllamav3 rev-parse HEAD)"
  [[ "$plugin" == "$1" ]] || { echo "ERROR: base vllm-exl3 $plugin != locked $1" >&2; exit 2; }
  [[ "$exllama" == "$2" ]] || { echo "ERROR: base ExLlamaV3 $exllama != locked $2" >&2; exit 2; }
' bash "$LOCKED_PLUGIN" "$LOCKED_EXLLAMA"

echo "=== Build disk-backed Engram qualification runtime ==="
echo "Base runtime:       $BASE_RUNTIME"
echo "Locked vllm-exl3:   $LOCKED_PLUGIN"
echo "Locked ExLlamaV3:   $LOCKED_EXLLAMA"
echo "Output image:       $DISK_IMAGE"

docker build \
  -f "$RECIPE_ROOT/Dockerfile.disk-engram" \
  --build-arg BASE_RUNTIME="$BASE_RUNTIME" \
  -t "$DISK_IMAGE" \
  "$RECIPE_ROOT"

docker image inspect "$DISK_IMAGE" >/dev/null

echo
echo "Built: $DISK_IMAGE"
echo "Use the dedicated wrapper on every Spark, for example:"
echo "  IMAGE=$DISK_IMAGE MODEL_DIR=/large/models MODEL=/models/<snapshot> \\\"
echo "  VLLM_ENGRAM_MODEL_DIR=/models/<snapshot> HEAD_IP=<head> NODE_IP=<this-node> \\\"
echo "    bash scripts/start_disk_engram_cluster.sh head|worker"
