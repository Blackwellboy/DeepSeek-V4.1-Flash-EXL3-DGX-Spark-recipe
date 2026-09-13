#!/usr/bin/env bash
# EXPERIMENTAL qualification path: TP4 min-fit with disk-backed Engram.
# This is an explicit experimental variable, not the resident-Engram baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE="${DISK_ENGRAM_PROFILE:-$RECIPE_ROOT/profiles/tp4-disk-engram.env}"

DRY=0
if [[ "${1:-}" == "--check" || "${1:-}" == "--dry-run" ]]; then
  DRY=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--check|--dry-run]" >&2
  exit 2
fi

if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: disk-Engram profile not found: $PROFILE" >&2
  exit 2
fi

# Load the explicit experimental profile. Explicit shell-prefix overrides still
# win once serve_tp4 sources lib.sh (do not re-implement .env precedence here).
set -a
# shellcheck disable=SC1090
source "$PROFILE"
set +a

if [[ "${VLLM_ENGRAM_DISK_BACKED:-}" != "1" ]]; then
  cat >&2 <<EOF
ERROR: fail-closed: VLLM_ENGRAM_DISK_BACKED must be 1 for this launcher.
Disk-backed Engram is an explicit experimental variable. Use scripts/tp4_min_fit.sh
for the resident-Engram capacity gate, or set VLLM_ENGRAM_DISK_BACKED=1.
EOF
  exit 2
fi

cat >&2 <<EOF
========================================================================
EXPERIMENTAL: TP4 disk-backed Engram min-fit qualification
Profile: $PROFILE
VLLM_ENGRAM_DISK_BACKED=${VLLM_ENGRAM_DISK_BACKED}
Native V4.1 MoE: ${NATIVE_MOE:-0}  DSpark: ${DSPARK:-0}  Eager: ${EAGER:-1}
Max model len: ${MAX_MODEL_LEN:-?}  Max num seqs: ${MAX_NUM_SEQS:-?}

Requirements (not auto-installed by this wrapper):
  1. Runtime/image must provide EngramConfig.disk_backed (see overlays/disk-engram/).
  2. Apply the overlay to the image-pinned V4.1 site-packages (volume mount or rebuild).
  3. Ray workers should inherit VLLM_ENGRAM_DISK_BACKED=1 and a node-local model dir.
  4. Arm scripts/watch_oom_guard.sh (WARN=24 GiB, ABORT=16 GiB) before load.
  5. Mixed-K loadability still depends on a separate vllm-exl3 PR (not this branch).
========================================================================
EOF

exec env \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}" \
  TEXT_ONLY="${TEXT_ONLY:-1}" \
  DSPARK="${DSPARK:-0}" \
  EAGER="${EAGER:-1}" \
  NATIVE_MOE="${NATIVE_MOE:-0}" \
  VLLM_ENGRAM_DISK_BACKED=1 \
  EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}" \
  DRY_RUN="$DRY" \
  "$SCRIPT_DIR/serve_tp4.sh"
