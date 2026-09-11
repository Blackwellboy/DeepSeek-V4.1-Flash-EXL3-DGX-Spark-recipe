#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-vcruz305/DSV4.1-Flash-EXL3-4.75bpw}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v41-exl3-uva}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-300}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "${VLLM_WEIGHT_OFFLOADING_DISABLE_UVA:-0}" == "1" ]]; then
  echo "ERROR: VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1 conflicts with this qualification path." >&2
  exit 2
fi

if ! is_true "$SKIP_PREFLIGHT"; then
  echo "=== sm_120 / UVA capability preflight ==="
  python "$SCRIPT_DIR/preflight_sm120_uva.py"
  echo
fi

# Fail closed if vLLM leaves any of the six large packed expert payloads resident,
# only partially offloads them, or falls back to ordinary CPU tensors.
export VLLM_EXL3_REQUIRE_UVA_EXPERTS=1

# TP1 has all 384 routed experts local. Start from auto/loop-capable behavior;
# do not force the <=128-expert fused path or the still-unqualified V4.1 native path.
export VLLM_EXL3_MOE_KERNEL="${VLLM_EXL3_MOE_KERNEL:-auto}"
export VLLM_EXL3_V41_NATIVE_MOE="${VLLM_EXL3_V41_NATIVE_MOE:-0}"

ARGS=(
  vllm serve "$MODEL"
  --quantization exl3
  --tokenizer-mode deepseek_v41
  --tensor-parallel-size 1
  --language-model-only
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --offload-backend uva
  --cpu-offload-gb "$CPU_OFFLOAD_GB"
  --cpu-offload-params
    w13_trellis w13_suh w13_svh
    w2_trellis w2_suh w2_svh
  --engram-config '{"cpu_offload":true}'
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --enforce-eager
)

if [[ -n "${MODEL_REVISION:-}" ]]; then
  ARGS+=( --revision "$MODEL_REVISION" )
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  echo "WARNING: EXTRA_VLLM_ARGS is shell-split; use only trusted local values." >&2
  # shellcheck disable=SC2206
  EXTRA=( $EXTRA_VLLM_ARGS )
  ARGS+=( "${EXTRA[@]}" )
fi

cat <<EOF
=== DeepSeek V4.1 EXL3 experimental sm_120 UVA launch ===
Model:                    $MODEL
Served name:              $SERVED_MODEL_NAME
Topology:                 TP1, one GPU
Expert execution:         GPU EXL3 kernels over vLLM UVA-mapped host parameters
Expert offload budget:    ${CPU_OFFLOAD_GB} GiB
Engram placement:         pinned CPU / UVA
Max model len:            $MAX_MODEL_LEN
Max sequences:            $MAX_NUM_SEQS
Max batched tokens:       $MAX_NUM_BATCHED_TOKENS
GPU memory utilization:   $GPU_MEMORY_UTILIZATION
EXL3 backend preference:  $VLLM_EXL3_MOE_KERNEL
Native V4.1 MoE:          $VLLM_EXL3_V41_NATIVE_MOE
Eager:                    yes
DSpark:                   no (first-boot qualification)
UVA placement guard:      required

WARNING: this path may require roughly 400-500 GiB-class pinned/mapped host state
once the EXL3 expert payload and retained Engram tables are both accounted for.
A 1 TiB machine has raw capacity, but the driver/OS may still reject that amount
of pinned memory. Treat the first failure as qualification evidence.
EOF

echo
printf 'Command:'
printf ' %q' "${ARGS[@]}"
echo

echo
exec "${ARGS[@]}"
