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

# 4x GB10 hardware evidence has already established that resident Engram reaches
# the unified-memory cliff at the corrected 8K/seq1 minimum-fit settings. Keep
# this launcher only as an explicit regression/reproduction tool; the normal TP4
# qualification path is scripts/tp4_disk_engram_min_fit.sh.
if [[ "$DRY" == "0" && "${ALLOW_RESIDENT_ENGRAM_RETEST:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL (known 4x DGX Spark / GB10 result)

The resident 8K/seq1 test has already driven a Spark to the near-full UMA cliff
(~121 GiB used class) and is no longer a useful default qualification step.

Use:
  bash scripts/tp4_disk_engram_min_fit.sh --check
  bash scripts/tp4_disk_engram_min_fit.sh

To intentionally reproduce the resident regression only:
  ALLOW_RESIDENT_ENGRAM_RETEST=1 bash scripts/tp4_min_fit.sh
EOF
  exit 2
fi

if [[ "$DRY" == "1" ]]; then
  echo "NOTE: resident Engram is a known TP4 GB10 capacity failure; this is command inspection only." >&2
fi

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
