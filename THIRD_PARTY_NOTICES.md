# Third-party notices and provenance

This repository contains original recipe/orchestration code, references external projects, and contains an **explicit experimental vLLM overlay** under `overlays/disk-engram/`. It does not redistribute DeepSeek model weights.

## DeepSeek

- Project/model: `deepseek-ai/DeepSeek-V4.1-Flash`
- Role: model architecture, checkpoint formats, tokenizer/tool protocol and DSpark contract.
- License: use the license published with the model/checkpoint you download.

## vLLM

- Project: https://github.com/vllm-project/vllm
- Dedicated V4.1 image: `vllm/vllm-openai:deepseekv41-flash-0909`
- Role: DeepSeek V4.1 graph, distributed execution, OpenAI server, Engram, sparse attention/indexer and DSpark runtime.
- Upstream license: Apache-2.0.

The baseline recipe uses public vLLM interfaces without copying vLLM source. The **disk-Engram experimental overlay is an exception**: it contains copied/adapted Apache-2.0 vLLM files with their upstream SPDX/copyright headers retained.

The exact upstream source commit represented by the dedicated image tag has not been independently resolved, so this repository deliberately does not invent one. The image tag plus recipe runtime lock define the current overlay compatibility boundary; any base-image change requires requalification.

## ExLlamaV3

- Project: https://github.com/turboderp-org/exllamav3
- Pinned revision: `be57335b087e4f001c5caae061544df3c06ba01e`
- Role: EXL3 trellis format, codebooks and packed execution.
- Upstream license: MIT; verify the pinned checkout for exact notices.

## vllm-exl3

- Project: https://github.com/vcruz305/vllm-exl3
- Pinned revision: `c69c8f8dda806ab19807a83506574fd391263cad`
- License: AGPL-3.0-only plus historical/third-party notices in that repository.

### PR #9 — @fattchris

Contributed GB10/setuptools compatibility and diagnostic shape handling, including relative CUDA extension source paths and regression coverage. The diagnostic behavior hard-failed by default.

### PR #10 — @Blackwellboy

Original contributor commit:

```text
74191ec2de80c961044b499e97646510784896e4
```

GitHub records the original PR as merged. It contributed the core **per-expert/per-projection mixed-K routed EXL3 implementation**:

- exact heterogeneous K3–K8 trellis geometry inside one routed layer;
- support for `w1/w2/w3` using different K inside one expert;
- contiguous per-shape trellis arenas/direct-fill path;
- preservation of the uniform-K fused fast path;
- correctness-first `LinearEXL3` fallback for heterogeneous layers;
- synthetic mixed-K/loader/arena regression tests;
- 4× DGX Spark evidence that loading proceeded beyond the former 64-vs-48 trellis failure.

Mainline hardening after that contribution:

- marks heterogeneous mixed-K eager-first/not CUDA-graph-qualified;
- disables arena prescan for non-linear/EPLB expert placement;
- derives fused uniform-layer K from the **loaded physical trellis geometry**, so a physically uniform K5/K7 layer cannot be launched using a stale config/base K.

Those follow-up maintainer changes do not rewrite or squash @Blackwellboy's original merged commit; it remains in `main` ancestry.

## GB10 build / RoCE contribution — @fattchris

Recipe PR #1 supplied hardware-validated 4× DGX Spark findings integrated into the mainline recipe, including conditional Python alias creation, dynamic cuSPARSE include discovery, configurable plugin source and guarded RDMA/RoCE settings. Mainline integration additionally preserved locked build inputs and non-destructive container startup.

## Disk-backed Engram contribution — @Blackwellboy

Recipe PR #2 original contributor commit:

```text
65c160e565fa4f0a01e55f433ab5868207f04401
```

The contribution established the guarded node-local disk-backed Engram direction after resident TP4 Engram reached the GB10 unified-memory cliff. Its core contribution includes:

- `EngramConfig.disk_backed` experimental mode;
- nonresident Engram embedding placeholders;
- skipping full Engram embedding tensor materialization;
- bounded row staging from local safetensors backing;
- TP ownership handling;
- synthetic parity, real-slice bit-exact parity and stress evidence;
- TP4 disk-Engram qualification profile and memory watchdog.

Mainline hardening on top of that contribution adds:

- a reproducible derived `Dockerfile.disk-engram` instead of manual site-packages editing;
- all-node Ray disk-Engram preflight;
- propagation of disk mode/model path into Ray workers and the vLLM process;
- exact-container OOM guard targeting and four-host guard verification;
- caller/profile precedence protection;
- eager-first mixed-K qualification contract;
- retirement of resident Engram as the normal TP4 first gate;
- rejection of whole-shard eager/prefetch/torchao loading strategies in disk mode.

### Overlay files

The experimental overlay contains:

- `overlays/disk-engram/vllm/config/engram.py`
- `overlays/disk-engram/vllm/models/deepseek_v4_1/common/engram.py`
- `overlays/disk-engram/vllm/models/deepseek_v4_1/common/engram_disk.py`
- `overlays/disk-engram/vllm/model_executor/model_loader/weight_utils.py`

Copied/adapted vLLM files retain Apache-2.0 headers. The new disk helper uses the same license/provenance boundary documented in its source header.

## NVIDIA / CUDA / DGX Spark

DGX Spark, CUDA, NVIDIA drivers and container runtime components are NVIDIA products with their own licenses. This repository does not redistribute them.
