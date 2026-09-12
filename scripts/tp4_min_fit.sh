#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY=0
if [[ "${1:-}" == "--check" || "${1:-}" == "--dry-run" ]]; then
  DRY=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--check|--dry-run]" >&2
  exit 2
fi

# Resident-Engram minimum-fit gate. Keep this intentionally narrow: one request,
# 8K context, text only, eager, no DSpark, no native ABI-3 experiment. The goal
# is to answer one binary question: does the resident TP4 checkpoint fit at all
# once large-context/batch pressure is removed?
exec env \
  MAX_MODEL_LEN=8192 \
  GPU_MEMORY_UTILIZATION="${MIN_FIT_GPU_MEMORY_UTILIZATION:-0.75}" \
  MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=1024 \
  TEXT_ONLY=1 \
  DSPARK=0 \
  EAGER=1 \
  NATIVE_MOE=0 \
  DRY_RUN="$DRY" \
  "$SCRIPT_DIR/serve_tp4.sh"
