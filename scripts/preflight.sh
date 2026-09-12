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

docker exec -i \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$CONTAINER_NAME" \
  "${ARGS[@]}"
