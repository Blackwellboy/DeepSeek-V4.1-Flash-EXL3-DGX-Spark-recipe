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
export MODEL
require_env MODEL

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "ERROR: head container '$CONTAINER_NAME' is not running on this host." >&2
  exit 2
fi

GPU_COUNT="$(docker exec "$CONTAINER_NAME" python -c 'import ray; ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR"); print(int(ray.cluster_resources().get("GPU", 0)))' 2>/dev/null | tail -n1)"
if [[ ! "$GPU_COUNT" =~ ^[0-9]+$ ]] || (( GPU_COUNT < TP )); then
  echo "ERROR: Ray reports ${GPU_COUNT:-unknown} GPUs; TP$TP requires at least $TP." >&2
  docker exec "$CONTAINER_NAME" ray status || true
  exit 2
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# GB10 CPU and GPU share one 128 GB physical memory pool. Keep the public
# first-boot default conservative; increase only from measured headroom.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
TEXT_ONLY="${TEXT_ONLY:-1}"
DSPARK="${DSPARK:-0}"
EAGER="${EAGER:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Topology default: TP4 starts from the ExLlamaV3 control. TP2 starts from the
# ABI-3 native candidate because 192 local experts exceed the current
# ExLlamaV3 fused-MoE 128-expert envelope.
if [[ -z "${NATIVE_MOE:-}" ]]; then
  if [[ "$TP" == "2" ]]; then NATIVE_MOE=1; else NATIVE_MOE=0; fi
fi

EXEC_ENV=(
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e VLLM_USE_RUST_FRONTEND=1
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
)

if is_true "$NATIVE_MOE"; then
  EXEC_ENV+=(
    -e VLLM_EXL3_V41_NATIVE_MOE=1
    -e VLLM_EXL3_MOE_KERNEL=native
  )
  BACKEND_LABEL="native-abi3"
else
  EXEC_ENV+=( -e VLLM_EXL3_V41_NATIVE_MOE=0 )
  if [[ "$TP" == "4" ]]; then
    EXEC_ENV+=( -e VLLM_EXL3_MOE_KERNEL=exllamav3 )
    BACKEND_LABEL="exllamav3-control"
  else
    EXEC_ENV+=( -e VLLM_EXL3_MOE_KERNEL=auto )
    BACKEND_LABEL="auto-fallback"
  fi
fi

ARGS=(
  vllm serve "$MODEL"
  --quantization exl3
  --tokenizer-mode deepseek_v41
  --tensor-parallel-size "$TP"
  --enable-expert-parallel
  --enable-ep-weight-filter
  --distributed-executor-backend ray
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v41
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

if [[ -n "${MODEL_REVISION:-}" ]]; then
  ARGS+=( --revision "$MODEL_REVISION" )
fi

if is_true "$TEXT_ONLY"; then
  ARGS+=( --language-model-only )
else
  ARGS+=( --mm-encoder-tp-mode data )
fi

if is_true "$EAGER"; then
  ARGS+=( --enforce-eager )
fi

if is_true "$DSPARK"; then
  ARGS+=(
    --speculative-config
    '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
  )
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  echo "WARNING: EXTRA_VLLM_ARGS is shell-split; use only trusted local values." >&2
  # shellcheck disable=SC2206
  EXTRA=( $EXTRA_VLLM_ARGS )
  ARGS+=( "${EXTRA[@]}" )
fi

cat <<EOF
=== DeepSeek V4.1 EXL3 launch ===
Topology:               TP$TP + EP$TP
Ray GPUs:               $GPU_COUNT
Model:                  $MODEL
Served name:            $SERVED_MODEL_NAME
EXL3 backend variant:   $BACKEND_LABEL
Native V4.1 MoE:        $NATIVE_MOE
DSpark:                 $DSPARK
Eager:                  $EAGER
Text only:              $TEXT_ONLY
Max model len:          $MAX_MODEL_LEN
Max num seqs:           $MAX_NUM_SEQS
Max batched tokens:     $MAX_NUM_BATCHED_TOKENS
GPU memory utilization: $GPU_MEMORY_UTILIZATION
Dry run:                $DRY_RUN
EOF

echo
printf 'Command:'
printf ' %q' "${ARGS[@]}"
echo

if is_true "$DRY_RUN"; then
  echo "DRY_RUN=1: command verified; exiting before model load."
  exit 0
fi

exec docker exec -i "${EXEC_ENV[@]}" "$CONTAINER_NAME" "${ARGS[@]}"
