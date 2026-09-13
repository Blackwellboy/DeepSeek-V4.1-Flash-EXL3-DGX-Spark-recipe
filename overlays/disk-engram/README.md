# overlays/disk-engram (experimental)

Runtime overlay that adds **disk-backed Engram** to an image-pinned DeepSeek V4.1 vLLM install. This is **not** part of the recipe baseline image build.

## Contents

| Path | Role |
|---|---|
| `vllm/config/engram.py` | Adds `EngramConfig.disk_backed` (+ GB10 UMA note on `cpu_offload`) |
| `vllm/models/deepseek_v4_1/common/engram_disk.py` | Node-local NVMe backing, bounded row staging, skip helpers |
| `vllm/models/deepseek_v4_1/common/engram.py` | Wires disk-backed path into ParallelEngramEmbedding |
| `vllm/model_executor/model_loader/weight_utils.py` | Skips full `engram.embed.{weight,scale}` materialization; `posix_fadvise` DONTNEED after consume |

Do **not** place `vllm-exl3` arena / mixed-K plugin code here. That belongs in a separate `vllm-exl3` PR.

## Provenance

- Adapted from the dedicated V4.1 image line (`vllm/vllm-openai:deepseekv41-flash-0909` / image-pinned Spark runtime).
- New disk-backed storage path reuses vLLM Engram dequant semantics (fp8_e4m3fn + ue8m0 per-32 scales) from the pinned runtime; it is **not** a copy of TonoKen3 harness code.
- `weight_utils.py` is Apache-2.0 vLLM code with a small Engram skip + fadvise addition for disk-backed mode.
- License: Apache-2.0 (vLLM). See repo [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

## Apply (qualification only)

Mount or copy onto the container site-packages that match the **locked** V4.1 image ABI, for example:

```bash
SITE=/usr/local/lib/python3.12/dist-packages
OV=overlays/disk-engram

docker run ... \
  -v "$PWD/$OV/vllm/config/engram.py:$SITE/vllm/config/engram.py:ro" \
  -v "$PWD/$OV/vllm/models/deepseek_v4_1/common/engram.py:$SITE/vllm/models/deepseek_v4_1/common/engram.py:ro" \
  -v "$PWD/$OV/vllm/models/deepseek_v4_1/common/engram_disk.py:$SITE/vllm/models/deepseek_v4_1/common/engram_disk.py:ro" \
  -v "$PWD/$OV/vllm/model_executor/model_loader/weight_utils.py:$SITE/vllm/model_executor/model_loader/weight_utils.py:ro" \
  -e VLLM_ENGRAM_DISK_BACKED=1 \
  -e VLLM_ENGRAM_MODEL_DIR=/models \
  ...
```

Re-qualify after any base-image or vLLM pin advance. Overlay drift against a newer site-packages is a fail-closed event.

## Enable

```bash
export VLLM_ENGRAM_DISK_BACKED=1
# and/or
# --engram-config {"cpu_offload":false,"disk_backed":true}
bash scripts/tp4_disk_engram_min_fit.sh
```
