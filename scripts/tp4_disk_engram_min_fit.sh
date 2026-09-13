#!/usr/bin/env bash
# EXPERIMENTAL qualification path: TP4 min-fit with disk-backed Engram.
# This is an explicit experimental variable, not the resident-Engram baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE="${DISK_ENGRAM_PROFILE:-$RECIPE_ROOT/profiles/tp4-disk-engram.env}"

# Capture the caller before lib.sh/.env and the qualification profile are read.
# Precedence is: explicit caller env > disk profile > .env > recipe defaults.
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

if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: disk-Engram profile not found: $PROFILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$PROFILE"
set +a

# Restore explicit shell-prefix values so the disk profile cannot reintroduce
# the .env precedence bug fixed in scripts/lib.sh.
for key in "${!_CALLER_ENV[@]}"; do
  printf -v "$key" '%s' "${_CALLER_ENV[$key]}"
  export "$key"
done
unset _CALLER_ENV key value

MODEL_RESOLVED="$(resolve_model_for_tp 4)"
if [[ "$MODEL_RESOLVED" != /* ]]; then
  cat >&2 <<EOF
ERROR: disk-backed Engram requires a materialized local checkpoint path.
Resolved MODEL is not an absolute in-container path:
  $MODEL_RESOLVED
Materialize the locked TP4 snapshot first and set MODEL=/models/<snapshot>.
EOF
  exit 2
fi

export VLLM_ENGRAM_DISK_BACKED=1
export VLLM_ENGRAM_MODEL_DIR="${VLLM_ENGRAM_MODEL_DIR:-$MODEL_RESOLVED}"
export VLLM_EXL3_MODEL_DIR="$VLLM_ENGRAM_MODEL_DIR"

# This is the correctness/capacity qualification path. Heterogeneous mixed-K
# currently dispatches through the LinearEXL3 Python loop and is not yet
# CUDA-graph qualified, so the min-fit gate intentionally locks eager mode.
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=1024
TEXT_ONLY=1
DSPARK=0
EAGER=1
NATIVE_MOE=0
export MAX_MODEL_LEN GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
export TEXT_ONLY DSPARK EAGER NATIVE_MOE

if [[ "${EXTRA_VLLM_ARGS:-}" != *"disk_backed"* ]]; then
  EXTRA_VLLM_ARGS='--engram-config {"cpu_offload":false,"disk_backed":true}'
fi
export EXTRA_VLLM_ARGS

if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "ERROR: head container '$CONTAINER_NAME' is not running." >&2
  echo "Start every node with scripts/start_disk_engram_cluster.sh first." >&2
  exit 2
fi

if ! docker exec "$CONTAINER_NAME" test -f \
  "$VLLM_ENGRAM_MODEL_DIR/model.safetensors.index.json"; then
  echo "ERROR: disk Engram model directory is not materialized in the head container:" >&2
  echo "  $VLLM_ENGRAM_MODEL_DIR/model.safetensors.index.json" >&2
  exit 2
fi

cat >&2 <<EOF
========================================================================
EXPERIMENTAL: TP4 disk-backed Engram min-fit qualification
Profile:                  $PROFILE
Model:                    $MODEL_RESOLVED
Engram model dir:         $VLLM_ENGRAM_MODEL_DIR
VLLM_ENGRAM_DISK_BACKED:  1
Native V4.1 MoE:          0
DSpark:                   0
Eager:                    1 (required for mixed-K qualification)
Max model len:            8192
Max num seqs:             1

Fail-closed gates before model load:
  1. every Ray GPU node has disk-Engram env enabled;
  2. every node has the exact local checkpoint/index at the same container path;
  3. every node imports EngramConfig.disk_backed + the weight-loader skip;
  4. every node has the mixed-K-capable vllm-exl3 runtime;
  5. node-local backing is not a network filesystem;
  6. every Spark host has a live exact-container UMA guard (real load only).
========================================================================
EOF

echo "Running all-node disk-Engram preflight..." >&2
docker exec "$CONTAINER_NAME" \
  python /recipe/scripts/check_disk_engram_cluster.py 4 "$VLLM_ENGRAM_MODEL_DIR"
echo "DISK_ENGRAM_CLUSTER_PREFLIGHT=PASS" >&2

if [[ "$DRY" == "0" ]]; then
  if is_true "${SKIP_OOM_GUARD_CHECK:-0}"; then
    echo "WARNING: SKIP_OOM_GUARD_CHECK=1; real load is proceeding without verified host guards." >&2
  else
    export OOM_GUARD_CONTAINER_NAME="${OOM_GUARD_CONTAINER_NAME:-$CONTAINER_NAME}"
    echo "Verifying four host-side UMA guards..." >&2
    "$SCRIPT_DIR/check_oom_guards.sh" 4
  fi
else
  echo "DRY_RUN: host OOM guards are not required for command/preflight inspection." >&2
fi

exec env \
  MODEL="$MODEL_RESOLVED" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  TEXT_ONLY=1 \
  DSPARK=0 \
  EAGER=1 \
  NATIVE_MOE=0 \
  VLLM_ENGRAM_DISK_BACKED=1 \
  VLLM_ENGRAM_MODEL_DIR="$VLLM_ENGRAM_MODEL_DIR" \
  VLLM_EXL3_MODEL_DIR="$VLLM_ENGRAM_MODEL_DIR" \
  EXTRA_VLLM_ARGS="$EXTRA_VLLM_ARGS" \
  DRY_RUN="$DRY" \
  "$SCRIPT_DIR/serve_tp4.sh"
