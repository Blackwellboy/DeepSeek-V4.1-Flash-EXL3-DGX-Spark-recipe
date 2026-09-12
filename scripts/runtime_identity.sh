#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

LOCK_BASE_IMAGE="$(python3 "$LOCK_TOOL" get base_image.ref)"
LOCK_MAX_MODEL_LEN="$(python3 "$LOCK_TOOL" get first_boot.max_model_len)"
LOCK_GPU_MEMORY_UTILIZATION="$(python3 "$LOCK_TOOL" get first_boot.gpu_memory_utilization)"
LOCK_MAX_NUM_SEQS="$(python3 "$LOCK_TOOL" get first_boot.max_num_seqs)"
LOCK_MAX_NUM_BATCHED_TOKENS="$(python3 "$LOCK_TOOL" get first_boot.max_num_batched_tokens)"
LOCK_TEXT_ONLY="$(python3 "$LOCK_TOOL" get first_boot.text_only)"
LOCK_DSPARK="$(python3 "$LOCK_TOOL" get first_boot.dspark)"
LOCK_EAGER="$(python3 "$LOCK_TOOL" get first_boot.eager)"
LOCK_NATIVE_MOE="$(python3 "$LOCK_TOOL" get first_boot.native_moe)"

printf '%s\n' '=== Runtime lock ==='
python3 "$LOCK_TOOL" dump

printf '%s\n' '=== Host/container image identity ==='
docker image inspect "$IMAGE" --format 'Derived image ID={{.Id}} RepoDigests={{json .RepoDigests}}'
docker image inspect "$LOCK_BASE_IMAGE" --format 'Base image ID={{.Id}} RepoDigests={{json .RepoDigests}}' 2>/dev/null || true

printf '%s\n' '=== Ray cluster ==='
docker exec "$CONTAINER_NAME" ray status || true

printf '%s\n' '=== GPU ==='
docker exec "$CONTAINER_NAME" nvidia-smi || true

printf '%s\n' '=== Runtime ==='
docker exec "$CONTAINER_NAME" python - <<'PY'
import json
import subprocess
import torch
import vllm
import vllm_exl3
import vllm_exl3_c

vllm_exl3.register()

def head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "vllm": getattr(vllm, "__version__", None),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "vllm_exl3_git": head("/opt/vllm-exl3"),
    "exllamav3_git": head("/opt/exllamav3"),
    "native_abi": int(getattr(vllm_exl3_c, "P2B_MOE_ABI_VERSION", 0)),
    "diagnostics": vllm_exl3.runtime_diagnostics(),
}, indent=2, sort_keys=True, default=str))
PY

printf '%s\n' '=== Recipe settings ==='
printf 'MODEL=%s\n' "${MODEL:-<lock-selected>}"
printf 'MODEL_REVISION=%s\n' "${MODEL_REVISION:-<lock-selected>}"
printf 'MAX_MODEL_LEN=%s\n' "${MAX_MODEL_LEN:-$LOCK_MAX_MODEL_LEN}"
printf 'GPU_MEMORY_UTILIZATION=%s\n' "${GPU_MEMORY_UTILIZATION:-$LOCK_GPU_MEMORY_UTILIZATION}"
printf 'MAX_NUM_SEQS=%s\n' "${MAX_NUM_SEQS:-$LOCK_MAX_NUM_SEQS}"
printf 'MAX_NUM_BATCHED_TOKENS=%s\n' "${MAX_NUM_BATCHED_TOKENS:-$LOCK_MAX_NUM_BATCHED_TOKENS}"
printf 'TEXT_ONLY=%s DSPARK=%s EAGER=%s NATIVE_MOE=%s\n' \
  "${TEXT_ONLY:-$LOCK_TEXT_ONLY}" "${DSPARK:-$LOCK_DSPARK}" "${EAGER:-$LOCK_EAGER}" "${NATIVE_MOE:-$LOCK_NATIVE_MOE}"
