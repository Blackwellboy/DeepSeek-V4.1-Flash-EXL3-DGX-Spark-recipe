#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

MODEL_RESOLVED="$(resolve_model_for_tp 2)"

# The published TP2 pack is tensor-level mixed-K and has also seen reports of
# non-materialized/invalid shard headers. Do not launch it directly from a repo
# id until a compatible repack is published. Materialize + validate first.
if [[ "$MODEL_RESOLVED" != /models/* ]]; then
  if ! is_true "${TP2_ALLOW_UNVALIDATED:-0}"; then
    cat >&2 <<EOF
ERROR: TP2 launch is fail-closed for remote/unvalidated checkpoints.

Current published repo: $MODEL_RESOLVED
Known blockers under investigation:
  - shard materialization/integrity must be verified;
  - tensor-level mixed K2-K8 is incompatible with the pinned layer-uniform loader;
  - TP2 requires streamed/nonresident Engram and measured disk/RAM headroom.

Materialize onto a large local/external mount first:
  bash scripts/materialize_tp2.sh /large/model/path/DSV4.1-Flash-SAGE-EXL3-TP2

Then mount its parent as MODEL_DIR on BOTH Sparks and set:
  MODEL=/models/DSV4.1-Flash-SAGE-EXL3-TP2

Run:
  python3 scripts/check_tp2_pack.py /large/model/path/DSV4.1-Flash-SAGE-EXL3-TP2

TP2_ALLOW_UNVALIDATED=1 is a loader-development bypass only; it is not a deployment recommendation.
EOF
    exit 2
  fi
  echo "WARNING: TP2_ALLOW_UNVALIDATED=1; bypassing TP2 snapshot safety gate." >&2
else
  # Validate the mounted local snapshot inside the existing head container when available.
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Validating mounted TP2 snapshot before launch: $MODEL_RESOLVED"
    docker exec -i "$CONTAINER_NAME" \
      python /recipe/scripts/check_tp2_pack.py "$MODEL_RESOLVED" \
      --reserve-gib "${TP2_POST_DOWNLOAD_RESERVE_GIB:-32}"
  fi
fi

export MODEL="$MODEL_RESOLVED"
exec "$SCRIPT_DIR/serve.sh" 2
