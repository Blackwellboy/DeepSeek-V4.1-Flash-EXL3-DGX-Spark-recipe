# DeepSeek-V4.1 EXL3 compatibility matrix

Updated: 2026-09-12.

A structurally valid EXL3 checkpoint is not automatically a qualified DeepSeek-V4.1 deployment. Keep **model-graph ownership**, **EXL3 format compatibility**, **Engram placement**, and **hardware qualification** separate.

## Current runtime paths

| Path | V4.1 graph | Routed EXL3 | Engram | Status |
|---|---|---|---|---|
| **DGX Spark TP4 + disk Engram + pinned `vllm-exl3`** | vLLM | exact per-expert K3–K8; heterogeneous layers use LinearEXL3 loop | node-local disk, bounded staging | **Primary TP4 qualification path; full `/v1/models` + smoke still pending** |
| DGX Spark resident Engram | vLLM | supported mixed-K loader | resident/shared UMA | **Known capacity failure at corrected 8K/seq1 min-fit; regression-only** |
| DGX Spark TP2 + nonresident Engram | vLLM | exact per-expert mixed-K supported | nonresident required | Loader-format compatible; capacity qualification pending |
| `sm_120` + large host RAM + UVA | current vLLM | GPU EXL3 kernels over mapped host parameters | pinned host/UVA | Separate large-host experiment; not the Spark capacity solution |
| Standalone pinned ExLlamaV3 | no complete V4.1 graph in pinned upstream registry | EXL3 kernels available | no V4.1 Engram graph | Cannot serve V4.1 end-to-end by itself |

`vllm-exl3` intentionally does not replace the DeepSeek-V4.1 graph. vLLM owns attention/CED/Engram/routing/DSpark; the plugin integrates packed routed-expert storage and execution.

## Mixed-K contract

Pinned `vllm-exl3` supports physical K2–K8 configuration and **heterogeneous K inside one routed layer**:

```text
expert A: w1=K3 w3=K4 w2=K5
expert B: w1=K8 w3=K8 w2=K7
```

Exact trellis shapes are retained per expert/projection. Uniform-K layers use the fused path when available; heterogeneous layers use the correctness-first `LinearEXL3` loop.

The heterogeneous path is **not CUDA-graph-qualified**. Eager mode is the first-boot contract.

## Spark Engram contract

Resident Engram on four GB10 nodes is already recorded as:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

Pinned-host/UVA placement does not create a second physical capacity pool on Spark. The primary TP4 capacity path is the explicit disk-backed derivative in `Dockerfile.disk-engram` / `docs/DISK_ENGRAM.md`.

Disk mode is not silently baked into the baseline runtime. It must pass all-node preflight and full-model serving qualification independently.

## TP/EP geometry

- TP4+EP4: 96 whole experts/rank, 5120×2304.
- TP2+EP2: 192 whole experts/rank, 5120×2304.
- Per-MoE TP/EP geometry is resolved before process-wide TP state.
- There is no 128-total-expert ExLlamaV3 ceiling.

The low-memory mixed-K arena prescan assumes linear contiguous expert placement. Mainline `vllm-exl3` disables that optimization for non-linear placement/EPLB and falls back to the authoritative loader mapping.

## Small-GPU / large-host UVA experiment

The separate `sm_120` path keeps the V4.1 graph in current vLLM and places selected packed experts/Engram in pinned host memory via UVA. That may be useful on a host with hundreds of GiB or TiB-class RAM, but it is **not** a Spark memory-capacity fix.

See `SM120_UVA.md` for that experiment.

## Quality interpretation

The released source experts are already low precision. EXL3 re-encoding cannot recover information lost in the source representation. Measure **additional transcode loss** against the original released checkpoint using fixed prompts/corpus and deterministic settings.

## Upstream references

- ExLlamaV3: https://github.com/turboderp-org/exllamav3
- vLLM: https://github.com/vllm-project/vllm
- `vllm-exl3`: https://github.com/vcruz305/vllm-exl3
- DeepSeek V4.1 model: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
