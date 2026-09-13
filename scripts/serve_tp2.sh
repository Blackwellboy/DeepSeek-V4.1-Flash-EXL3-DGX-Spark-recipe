#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

MODEL_RESOLVED="$(resolve_model_for_tp 2)"

# The published TP2 pack is representable by the pinned mixed-K loader, but its
# canonical config predates the explicit mixed-format delegation fields required
# by vllm-exl3. The exact locked snapshot is therefore validated against a
# checked-in immutable metadata attestation; canonical model files are never edited.
# A local checkpoint may be mounted either directly at /models or beneath it.
if [[ "$MODEL_RESOLVED" != "/models" && "$MODEL_RESOLVED" != /models/* ]]; then
  if ! is_true "${TP2_ALLOW_UNVALIDATED:-0}"; then
    cat >&2 <<EOF
ERROR: TP2 launch is fail-closed for remote/unvalidated checkpoints.

Current published repo: $MODEL_RESOLVED
Current status:
  - per-expert/per-projection mixed K2-K8 is supported by the pinned loader;
  - the canonical snapshot must pass strict physical + metadata-attestation validation;
  - TP2 requires nonresident disk-backed Engram and measured UMA headroom;
  - EP2 is the correctness baseline; pure MoE TP2 is an explicit A/B candidate.

Materialize onto a large local/external mount first:
  bash scripts/materialize_tp2.sh /large/model/path/DSV4.1-Flash-SAGE-EXL3-TP2

Then either mount the exact checkpoint directory as MODEL_DIR and use MODEL=/models,
or mount its parent and use MODEL=/models/DSV4.1-Flash-SAGE-EXL3-TP2.

Run on the host path first if desired:
  python3 scripts/check_tp2_pack.py /large/model/path/DSV4.1-Flash-SAGE-EXL3-TP2 --strict-locked-snapshot

TP2_ALLOW_UNVALIDATED=1 is a loader-development bypass only; it is not a deployment recommendation.
EOF
    exit 2
  fi
  echo "WARNING: TP2_ALLOW_UNVALIDATED=1; bypassing TP2 snapshot safety gate." >&2
else
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Validating mounted TP2 snapshot + immutable metadata attestation: $MODEL_RESOLVED"
    docker exec -i "$CONTAINER_NAME" \
      python /recipe/scripts/check_tp2_pack.py "$MODEL_RESOLVED" \
      --reserve-gib "${TP2_POST_DOWNLOAD_RESERVE_GIB:-32}" \
      --strict-locked-snapshot

    # The canonical snapshot's files remain untouched. If its old config lacks
    # delegation metadata, derive a runtime-only vLLM HF override from the exact
    # attestation that just passed. A changed shard/config makes this command fail.
    HF_OVERRIDES_JSON="$(docker exec -i "$CONTAINER_NAME" \
      python /recipe/scripts/check_tp2_pack.py "$MODEL_RESOLVED" \
      --reserve-gib "${TP2_POST_DOWNLOAD_RESERVE_GIB:-32}" \
      --strict-locked-snapshot --print-hf-overrides)"
    if [[ -n "$HF_OVERRIDES_JSON" && "$HF_OVERRIDES_JSON" != "{}" ]]; then
      export HF_OVERRIDES_JSON
      echo "TP2 runtime metadata source: locked snapshot attestation (--hf-overrides only; model files unchanged)"
    fi
  fi
fi

export MODEL="$MODEL_RESOLVED"
exec "$SCRIPT_DIR/serve.sh" 2
