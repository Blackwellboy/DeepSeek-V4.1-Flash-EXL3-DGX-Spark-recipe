#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

MODEL_RESOLVED="$(resolve_model_for_tp 4)"
LOCK_STATUS="$(python3 "$LOCK_TOOL" get models.tp4.status)"

if [[ "$MODEL_RESOLVED" != /models/* ]]; then
  if [[ "$LOCK_STATUS" != "deployable" ]] && ! is_true "${TP4_ALLOW_UNVALIDATED:-0}"; then
    cat >&2 <<EOF
ERROR: TP4 direct remote launch is fail-closed because the locked checkpoint status is:
  $LOCK_STATUS

The current vllm-exl3 routed allocation requires one physical K width per
RoutedExperts transformer layer. Materialize and validate the exact locked TP4
snapshot before attempting a production/benchmark load:

  bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-TP4

Then set on EVERY Spark:
  MODEL_DIR=/large/models
  MODEL=/models/DSV4.1-Flash-EXL3-TP4

TP4_ALLOW_UNVALIDATED=1 is a loader-development bypass only.
EOF
    exit 2
  fi
  if is_true "${TP4_ALLOW_UNVALIDATED:-0}"; then
    echo "WARNING: TP4_ALLOW_UNVALIDATED=1; bypassing TP4 physical-pack safety gate." >&2
  fi
else
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Validating mounted TP4 snapshot before launch: $MODEL_RESOLVED"
    docker exec -i "$CONTAINER_NAME" \
      python /recipe/scripts/validate_pack.py "$MODEL_RESOLVED" \
      --topology tp4 --reserve-gib "${PACK_RESERVE_GIB:-32}"
  fi
fi

export MODEL="$MODEL_RESOLVED"
exec "$SCRIPT_DIR/serve.sh" 4
