# DeepSeek-V4.1 EXL3 compatibility matrix

Updated: 2026-09-11.

This document is deliberately fail-closed. A checkpoint being structurally valid EXL3 does not mean every EXL3 runtime can execute the DeepSeek-V4.1 model graph.

## Current runtime boundaries

| Path | V4.1 model graph | EXL3 routed experts | Engram / CED | CPU-MoE | Status |
|---|---|---|---|---|---|
| Dedicated DeepSeek V4.1 vLLM image + `vllm-exl3` | vLLM | `vllm-exl3` | vLLM | No host-resident EXL3 CPU executor in `vllm-exl3` | Primary recipe path; hardware qualification ongoing |
| Stock/pinned ExLlamaV3 standalone | No `DeepseekV41ForCausalLM` architecture in the pinned upstream registry | EXL3 execution exists for supported architectures | No V4.1 CED/Engram model graph | Experimental CPU-MoE exists for eligible ExLlamaV3 models | **Cannot load V4.1 end-to-end today** |
| SAGE conversion-only V4.1 port | Conversion/inventory only | Encodes EXL3 | Forward intentionally absent | Not an inference path | Conversion smoke tooling only |

`vllm-exl3` intentionally does not reimplement the DeepSeek-V4.1 architecture. vLLM owns attention, CED/source relationships, Engram, routing and DSpark while the plugin supplies EXL3 storage/execution integration.

The SAGE conversion-only port is also intentionally not an inference loader. Its `forward()` is disabled so a conversion smoke test cannot be mistaken for a qualified V4.1 runtime.

## ExLlamaV3 CPU-MoE constraints

The pinned upstream ExLlamaV3 revision has an experimental host-resident CPU-MoE path. Its current eligibility contract requires:

- `mul1` codebook experts;
- K <= 8;
- uniform per-expert bias presence for the layer (all or none);
- an architecture that ExLlamaV3 itself can instantiate.

Therefore a V4.1 pack can still be incompatible with CPU-MoE even after a V4.1 model port lands. In particular, an MCG-only pack is not currently eligible for ExLlamaV3's CPU-MoE executor.

Run the metadata preflight against a local checkpoint before promising compatibility:

```bash
python scripts/check_checkpoint_compat.py /path/to/DSV4.1-Flash-EXL3-4.75bpw
```

A `CONDITIONAL` result means only that the metadata checker did not find a known blocker. It is not a runtime pass.

## NVIDIA architecture targets

The default Docker recipe remains DGX Spark / GB10 and builds extensions for:

```text
TORCH_CUDA_ARCH_LIST=12.1a
```

The build target is now configurable. For a CUDA 12.8 environment whose installed PyTorch/toolchain reports an `sm_120` GPU, build with:

```bash
TORCH_CUDA_ARCH_LIST=12.0 ./scripts/build_runtime.sh
```

Do not blindly copy the Spark `12.1a` target to an `sm_120` host. Verify the resulting PyTorch build reports `sm_120` before benchmarking:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(torch.cuda.get_arch_list())
PY
```

Architecture compilation support does not imply all vLLM/FlashInfer/CUTLASS kernels are qualified for that GPU. Capture actual backend dispatch and fallbacks.

## Small-GPU + large-host-memory topology

A host such as 16 GB `sm_120` Blackwell + 1 TiB DDR5 + fast NVMe/Optane is a fundamentally different target from TP4+EP4 DGX Spark:

1. attention/dense/CED stay GPU-resident or use the runtime's normal placement;
2. routed experts remain packed in host RAM and execute on CPU;
3. Engram tables remain mmap-backed/nonresident when possible;
4. GPU memory is reserved for the non-expert graph, caches and workspaces.

The current public recipe does **not** implement that topology. Supporting it requires either:

- a forward-correct DeepSeek-V4.1 ExLlamaV3 architecture so the existing ExLlamaV3 CPU-MoE machinery can be reused, plus an eligible EXL3 codebook; or
- a new host-resident CPU expert backend integrated into the vLLM path.

The first route is the smaller integration surface and should be qualified before duplicating CPU-MoE machinery in `vllm-exl3`.

## Quality interpretation

DeepSeek-V4.1's routed experts are already distributed in a low-precision source format. Re-encoding that source into a higher nominal EXL3 bpw cannot recreate information already discarded by the source quantization. The useful quality question is therefore **additional transcode loss** relative to the original released checkpoint.

A proper comparison uses the same prompts/corpus and reports original-vs-EXL3 perplexity/output parity alongside throughput. Do not claim that a higher EXL3 bpw makes the source intrinsically more accurate than the released source representation.

## Upstream references

- ExLlamaV3: https://github.com/turboderp-org/exllamav3
- vLLM: https://github.com/vllm-project/vllm
- NVIDIA NeMo Automodel DeepSeek-V4.1 implementation: https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/components/models/deepseek_v41
- `vllm-exl3`: https://github.com/vcruz305/vllm-exl3
- SAGE-EXL3 V4.1 conversion work: `vcruz305/SAGE-EXL3` (private development repository at the time of writing)
