#!/usr/bin/env bash
# EXPERIMENTAL qualification path: TP2 min-fit with disk-backed Engram.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE="${DISK_ENGRAM_PROFILE:-$RECIPE_ROOT/profiles/tp2-disk-engram.env}"

declare -A _CALLER_ENV=()
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
  _CALLER_ENV["$key"]="$value"
done < <(env)

# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

DRY=0
if [[ "${1:-}" == "--check" || "${1:-}" == "--dry-run" ]]; then
  DRY=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--check|--dry-run]" >&2
  exit 2
fi

[[ -f "$PROFILE" ]] || { echo "ERROR: TP2 disk-Engram profile not found: $PROFILE" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$PROFILE"
set +a
for key in "${!_CALLER_ENV[@]}"; do
  printf -v "$key" '%s' "${_CALLER_ENV[$key]}"
  export "$key"
done
unset _CALLER_ENV key value

MODEL_RESOLVED="$(resolve_model_for_tp 2)"
if [[ "$MODEL_RESOLVED" != /* ]]; then
  cat >&2 <<EOF
ERROR: TP2 disk-backed Engram requires a materialized local checkpoint path.
Resolved MODEL: $MODEL_RESOLVED
Set MODEL=/models/<TP2 snapshot> after materializing the locked checkpoint on both Sparks.
EOF
  exit 2
fi

export VLLM_ENGRAM_DISK_BACKED=1
export VLLM_ENGRAM_MODEL_DIR="${VLLM_ENGRAM_MODEL_DIR:-$MODEL_RESOLVED}"
export VLLM_EXL3_MODEL_DIR="$VLLM_ENGRAM_MODEL_DIR"

# Keep the first capacity/correctness boot controlled. MOE_PARALLEL_MODE may be
# ep (baseline) or tp (experimental A/B); all other major variables stay fixed.
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=1024
TEXT_ONLY=1
DSPARK=0
EAGER=1
NATIVE_MOE=0
MOE_PARALLEL_MODE="${MOE_PARALLEL_MODE:-ep}"
export MAX_MODEL_LEN GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
export TEXT_ONLY DSPARK EAGER NATIVE_MOE MOE_PARALLEL_MODE

if [[ "${EXTRA_VLLM_ARGS:-}" != *"disk_backed"* ]]; then
  EXTRA_VLLM_ARGS='--engram-config {"cpu_offload":false,"disk_backed":true}'
fi
export EXTRA_VLLM_ARGS

if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "ERROR: head container '$CONTAINER_NAME' is not running." >&2
  echo "Start both nodes with DISK_ENGRAM_PROFILE=profiles/tp2-disk-engram.env scripts/start_disk_engram_cluster.sh." >&2
  exit 2
fi

if ! docker exec "$CONTAINER_NAME" test -f "$VLLM_ENGRAM_MODEL_DIR/model.safetensors.index.json"; then
  echo "ERROR: TP2 snapshot is not materialized at $VLLM_ENGRAM_MODEL_DIR inside the head container." >&2
  exit 2
fi

cat >&2 <<EOF
========================================================================
EXPERIMENTAL: TP2 disk-backed Engram min-fit qualification
Model:                    $MODEL_RESOLVED
MoE parallel mode:        $MOE_PARALLEL_MODE
VLLM_ENGRAM_DISK_BACKED:  1
Eager / DSpark / native:  1 / 0 / 0
Context / seq / batch:    8192 / 1 / 1024

EP2 is the correctness baseline: 192 whole experts/rank at 5120x2304.
Pure MoE TP2 is an A/B candidate: 384 experts/rank at 5120x1152; 1152 is
already 128-aligned, so no 576->640-style padding is required.
========================================================================
EOF

echo "Running two-node disk-Engram preflight..." >&2
docker exec "$CONTAINER_NAME" \
  python /recipe/scripts/check_disk_engram_cluster.py 2 "$VLLM_ENGRAM_MODEL_DIR"
echo "DISK_ENGRAM_CLUSTER_PREFLIGHT=PASS" >&2

if [[ "$DRY" == "0" ]]; then
  if is_true "${SKIP_OOM_GUARD_CHECK:-0}"; then
    echo "WARNING: SKIP_OOM_GUARD_CHECK=1; proceeding without verified host guards." >&2
  else
    export OOM_GUARD_CONTAINER_NAME="${OOM_GUARD_CONTAINER_NAME:-$CONTAINER_NAME}"
    "$SCRIPT_DIR/check_oom_guards.sh" 2
  fi
fi

exec env \
  MODEL="$MODEL_RESOLVED" \
  MAX_MODEL_LEN=8192 \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=1024 \
  TEXT_ONLY=1 DSPARK=0 EAGER=1 NATIVE_MOE=0 \
  MOE_PARALLEL_MODE="$MOE_PARALLEL_MODE" \
  VLLM_ENGRAM_DISK_BACKED=1 \
  VLLM_ENGRAM_MODEL_DIR="$VLLM_ENGRAM_MODEL_DIR" \
  VLLM_EXL3_MODEL_DIR="$VLLM_ENGRAM_MODEL_DIR" \
  EXTRA_VLLM_ARGS="$EXTRA_VLLM_ARGS" \
  DRY_RUN="$DRY" \
  "$SCRIPT_DIR/serve_tp2.sh"
