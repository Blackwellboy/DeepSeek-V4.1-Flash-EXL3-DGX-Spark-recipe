# Third-party notices and provenance

This repository contains original recipe/orchestration code and references external projects. It does **not** redistribute model weights or vendor the DeepSeek V4.1 model implementation.

## DeepSeek

- Project/model: `deepseek-ai/DeepSeek-V4.1-Flash`
- Role here: model architecture, tokenizer behavior, reasoning/tool protocol, DSpark training/runtime contract and checkpoint formats.
- License: use the license published with the model/checkpoint you download.

## vLLM

- Project: https://github.com/vllm-project/vllm
- Dedicated V4.1 image used by this recipe: `vllm/vllm-openai:deepseekv41-flash-0909`
- Role here: DeepSeek V4.1 architecture implementation, distributed execution, OpenAI server, sparse attention/indexer, DSpark runtime and source MXFP4/MXFP8 support.
- Upstream license: Apache-2.0.

The command-line settings in this repository are independently assembled from public vLLM interfaces and the public DeepSeek-V4.1 recipe. No vLLM source files are copied here.

## ExLlamaV3

- Project: https://github.com/turboderp-org/exllamav3
- Pinned revision: `be57335b087e4f001c5caae061544df3c06ba01e`
- Role here: EXL3 trellis format, codebooks and packed execution used by `vllm-exl3`.
- Upstream license: MIT (verify the pinned checkout for exact notices).

## vllm-exl3

- Project: https://github.com/vcruz305/vllm-exl3
- Pinned revision: `3ce1ae08f3e4a9545c58d5ac6456807c702d524a`
- Role here: vLLM quantization plugin, routed-expert EXL3 loading/execution, source-quantization delegation, DeepSeek V4.1 TP/EP planning, mixed-K capability declaration, and native ABI-3 dynamic MoE geometry.
- License at the pinned revision: AGPL-3.0-only, with additional historical/third-party notices in that repository.

That pinned revision includes PR #9 by `@fattchris`, which contributed:

- setuptools>=77-compatible relative CUDA extension source paths;
- an explicitly opt-in diagnostic shape-mismatch mode;
- coordinate-preserving overlap copies for that diagnostic path;
- regression tests preserving the default hard-fail behavior.

The runtime image builds the plugin from its own source repository so its license and attribution files remain available in `/opt/vllm-exl3`.

## GB10 recipe build / RoCE contribution

Recipe PR #1 by `@fattchris` supplied hardware-validated 4× DGX Spark findings that were integrated into the mainline recipe rather than merged verbatim after the repository had diverged. The integrated work includes:

- conditional creation of the `python` alias in the dedicated V4.1 image;
- dynamic discovery of the NVIDIA cuSPARSE header include directory;
- configurable `VLLM_EXL3_REPO` build input;
- fail-safe RDMA device passthrough and RoCE/NCCL configuration.

The mainline integration additionally preserves the locked ExLlamaV3 build argument, applies Spark-specific interface defaults only when those interfaces exist, and refuses to replace an existing runtime container without explicit opt-in.

## NVIDIA / CUDA / DGX Spark

DGX Spark, CUDA, NVIDIA drivers and container runtime components are proprietary NVIDIA products with their own licenses. This repository does not redistribute them.

## Community Spark work

Other public DeepSeek V4.1 Spark experiments may use disk-backed Engram, FlashInfer SM12x patches, indexer changes or JIT prebuilds. This baseline intentionally does not copy those patches. If a future recipe variant adopts code or a patch from another project, add the exact source commit/file and license here before merging it.
