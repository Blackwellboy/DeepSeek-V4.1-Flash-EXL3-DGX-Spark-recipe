#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

TP="${1:-}"
if [[ "$TP" != "2" && "$TP" != "4" ]]; then
  echo "Usage: $0 2|4" >&2
  exit 2
fi

MODEL="$(resolve_model_for_tp "$TP")"
MODEL_REVISION_RESOLVED="$(resolve_model_revision_for_tp "$TP")"
export MODEL
require_env MODEL

ARGS=(
  python /recipe/scripts/preflight.py
  --model "$MODEL"
  --tp "$TP"
  --pack-reserve-gib "${PACK_RESERVE_GIB:-32}"
)
if [[ -n "$MODEL_REVISION_RESOLVED" ]]; then
  ARGS+=( --revision "$MODEL_REVISION_RESOLVED" )
fi

echo "=== Static/runtime preflight ==="
docker exec -i \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$CONTAINER_NAME" \
  "${ARGS[@]}"

if is_true "${SKIP_NCCL_COLLECTIVE:-0}"; then
  echo "WARNING: SKIP_NCCL_COLLECTIVE=1; distributed GPU transport was not qualified." >&2
  exit 0
fi

echo
echo "=== Cross-node NCCL collective preflight ==="
COLLECTIVE_ARGS=(
  python /recipe/scripts/cluster_collective.py
  --tp "$TP"
  --megabytes "${COLLECTIVE_MEGABYTES:-16}"
  --warmup "${COLLECTIVE_WARMUP:-2}"
  --iterations "${COLLECTIVE_ITERATIONS:-5}"
  --master-port "${COLLECTIVE_MASTER_PORT:-29557}"
)
if is_true "${COLLECTIVE_JSON:-0}"; then
  COLLECTIVE_ARGS+=( --json )
fi

docker exec -i "$CONTAINER_NAME" "${COLLECTIVE_ARGS[@]}"
