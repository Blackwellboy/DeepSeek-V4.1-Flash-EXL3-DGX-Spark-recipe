# DeepSeek-V4.1 EXL3 compatibility matrix

Updated: 2026-09-11.

This document is deliberately fail-closed. A checkpoint being structurally valid EXL3 does not mean every EXL3 runtime can execute the DeepSeek-V4.1 model graph.

## Current runtime boundaries

| Path | V4.1 model graph | EXL3 routed experts | Engram / CED | Host-memory behavior | Status |
|---|---|---|---|---|---|
| Dedicated DeepSeek V4.1 vLLM image + `vllm-exl3` | vLLM | `vllm-exl3` | vLLM | Spark-oriented resident/unified-memory path | Primary TP4/TP2 recipe; hardware qualification ongoing |
| **Current vLLM + selective UVA + `vllm-exl3`** | vLLM | GPU EXL3 kernels over vLLM mapped host parameters | vLLM V4.1 Engram supports CPU/UVA placement | **Pinned host RAM, GPU compute over UVA** | **Preferred first experiment for one 16 GB `sm_120` GPU + 1 TiB host; not yet end-to-end qualified** |
| Stock/pinned ExLlamaV3 standalone | No `DeepseekV41ForCausalLM` architecture in the pinned upstream registry | EXL3 execution exists for supported architectures | No V4.1 CED/Engram model graph | Experimental CPU-compute MoE exists for eligible ExLlamaV3 models | **Cannot load V4.1 end-to-end today** |
| SAGE conversion-only V4.1 port | Conversion/inventory only | Encodes EXL3 | Forward intentionally absent | Not an inference path | Conversion smoke tooling only |

`vllm-exl3` intentionally does not reimplement the DeepSeek-V4.1 architecture. vLLM owns attention, CED/source relationships, Engram, routing and DSpark while the plugin supplies EXL3 storage/execution integration.

The SAGE conversion-only port is also intentionally not an inference loader. Its `forward()` is disabled so a conversion smoke test cannot be mistaken for a qualified V4.1 runtime.

## Preferred small-GPU experiment: current vLLM UVA

Current upstream vLLM now exposes two pieces that make the Lna-Lab topology testable without first finishing a standalone ExLlamaV3 V4.1 graph port:

- `EngramConfig.cpu_offload` for DeepSeek-V4.1 Engram, backed by pinned host memory and UVA;
- selective model-weight UVA offload through `--offload-backend uva`, `--cpu-offload-gb`, and `--cpu-offload-params`.

`vllm-exl3` now has an opt-in placement guard that can require the six large packed EXL3 expert payloads to be represented as vLLM UVA-mapped accelerator views before its EXL3 handles/pointer tables are accepted:

```text
w13_trellis  w13_suh  w13_svh
w2_trellis   w2_suh   w2_svh
```

This is **not CPU-MoE compute**. The expert math remains on the GPU and reads packed parameters over UVA. It is attractive because the V4.1 graph, CED and Engram stay in the already-working vLLM implementation.

Use [`SM120_UVA.md`](SM120_UVA.md) and `scripts/serve_sm120_uva.sh`. The design reference is current vLLM commit `988d9b6777d077f843cd2164a222ac8535d36ed2`; do not assume the older Spark container already carries these newer offload interfaces.

A successful UVA placement check is not a quality or throughput pass. The path still needs real `sm_120` testing for output parity, pinned-memory limits, PCIe behavior and backend dispatch.

## ExLlamaV3 CPU-MoE constraints

The pinned upstream ExLlamaV3 revision has an experimental host-resident **CPU-compute** MoE path. Its current eligibility contract requires:

- `mul1` codebook experts;
- K <= 8;
- uniform per-expert bias presence for the layer (all or none);
- an architecture that ExLlamaV3 itself can instantiate.

Therefore a V4.1 pack can still be incompatible with ExLlamaV3 CPU-MoE even after a standalone V4.1 model port lands. In particular, an MCG-only pack is not currently eligible for that CPU executor.

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

A host such as 16 GB `sm_120` Blackwell + 1 TiB DDR5 + fast NVMe/Optane is fundamentally different from TP4+EP4 DGX Spark.

There are now two explicit development routes:

1. **vLLM UVA route — test first:** keep the V4.1 graph in current vLLM, keep Engram in pinned host memory via vLLM's Engram offload, keep packed EXL3 expert payloads in pinned host memory via selective UVA, and execute experts with the GPU EXL3 kernels over mapped pointers.
2. **Standalone ExLlamaV3 CPU-compute route — fallback/second path:** complete a forward-correct DeepSeek-V4.1 ExLlamaV3 architecture, then reuse ExLlamaV3 CPU-MoE if the exact pack satisfies its codebook/K/bias constraints.

The UVA route is now the smaller integration surface because it avoids reimplementing CED/Engram solely to reach host memory. If UVA is too PCIe-bound or cannot pin the required hundreds of GiB, the standalone CPU-compute route remains valuable.

## Quality interpretation

DeepSeek-V4.1's routed experts are already distributed in a low-precision source format. Re-encoding that source into a higher nominal EXL3 bpw cannot recreate information already discarded by the source quantization. The useful quality question is therefore **additional transcode loss** relative to the original released checkpoint.

A proper comparison uses the same prompts/corpus and reports original-vs-EXL3 perplexity/output parity alongside throughput. Do not claim that a higher EXL3 bpw makes the source intrinsically more accurate than the released source representation.

## Upstream references

- ExLlamaV3: https://github.com/turboderp-org/exllamav3
- vLLM: https://github.com/vllm-project/vllm
- NVIDIA NeMo Automodel DeepSeek-V4.1 implementation: https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/components/models/deepseek_v41
- `vllm-exl3`: https://github.com/vcruz305/vllm-exl3
- SAGE-EXL3 V4.1 conversion/forward-planning work: `vcruz305/SAGE-EXL3` (private development repository at the time of writing)
