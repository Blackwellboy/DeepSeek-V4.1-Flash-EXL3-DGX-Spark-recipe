# Hugging Face model-card correction for DSV4.1-Flash-EXL3-4.75bpw

This is the replacement runtime/compatibility text for the public `vcruz305/DSV4.1-Flash-EXL3-4.75bpw` model card.

It intentionally does **not** duplicate or replace the model repository's license/frontmatter. Keep the existing Hugging Face metadata and replace the misleading standalone-runtime claims with the body below.

---

# DeepSeek-V4.1-Flash EXL3 — 4.75 bpw

Compiled EXL3 checkpoint derived from `deepseek-ai/DeepSeek-V4.1-Flash`, published by `vcruz305`.

> **Important runtime status:** the checkpoint files are complete, but **stock/current standalone ExLlamaV3 does not currently provide a forward-correct `DeepseekV41ForCausalLM` loader with V4.1 CED/CSA2/Engram support.** Do not interpret “EXL3 checkpoint” as “loadable by every ExLlamaV3 runtime.”

Any earlier wording that suggested a generally available “recent ExLlamaV3 with DeepSeek V4.1 / CED support” was too broad. The direct standalone `Config.from_directory(...) -> Model.from_config(...) -> model.load(...)` path should not be presented as working until a public forward-qualified V4.1 ExLlamaV3 implementation exists.

## What is in the pack

- source: `deepseek-ai/DeepSeek-V4.1-Flash`;
- architecture metadata: `DeepseekV41ForCausalLM`;
- routed-expert body re-encoded to EXL3;
- approximately 4.75-bpw compiled EXL3 target as described by the release;
- V4.1-specific metadata/retained structures preserved rather than treating V4.1 as a V4 architecture alias.

DeepSeek-V4.1's source routed experts are already distributed in a low-precision representation. A higher nominal EXL3 bpw cannot recreate information discarded before the transcode. The quality target is therefore **minimal additional transcode loss versus the released source checkpoint**, not an accuracy improvement over the source.

## Runtime compatibility

| Runtime | Status |
|---|---|
| V4.1-capable vLLM + `vcruz305/vllm-exl3` | **Current integration target.** vLLM owns the V4.1 model graph/CED/Engram/routing; `vllm-exl3` supplies EXL3 routed-expert integration. Qualification remains topology-specific. |
| Current vLLM + selective UVA host offload + `vllm-exl3` | **Experimental one-small-GPU / large-host-RAM path.** Engram and packed EXL3 expert parameters can be placed in pinned host memory while GPU EXL3 kernels execute over mapped UVA pointers. Placement/correctness/performance still require hardware qualification. |
| Stock/current standalone ExLlamaV3 | **Not currently end-to-end loadable.** Upstream has `DeepseekV4ForCausalLM`; V4.1 requires its own CED/CSA2/Engram forward graph. |
| SAGE-EXL3 V4.1 conversion port | Conversion/source-loading and architecture-planning work only. Its conversion-only class deliberately does not claim a working V4.1 forward path. |
| ExLlamaV3 CPU-MoE | Separate fallback path. It requires a forward-correct V4.1 ExLlamaV3 architecture plus the exact pack satisfying that backend's current format constraints. |

### Why DeepSeek V4 cannot be used as an alias

DeepSeek-V4.1 uses a different compression/source topology from V4. The released V4.1 graph uses compression ratios `0`, `2` and `1`, explicit KV/index/candidate source relationships, and Engram layers. The existing V4 ExLlamaV3 class is therefore not a safe compatibility alias.

## Current recommended serving/validation paths

### DGX Spark / GB10

Use the public recipe:

- `https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe`

TP4+EP4 is the preferred correctness-first Spark target; TP2 remains a separate experimental target.

### One `sm_120` GPU + large host RAM

Current vLLM exposes a promising shorter path than waiting for a complete standalone V4.1 ExLlamaV3 port:

- V4.1 Engram CPU/UVA placement;
- selective generic parameter UVA offload;
- `vllm-exl3` fail-closed validation for the six large packed EXL3 expert payloads before normal EXL3 handle/pointer construction.

This route keeps expert **compute on the GPU** while packed expert parameters live in pinned host RAM and are read over UVA/PCIe. It is not CPU-MoE compute.

Use:

- `docs/SM120_UVA.md`
- `scripts/preflight_sm120_uva.py`
- `scripts/serve_sm120_uva.sh`

from the public recipe above.

## Standalone ExLlamaV3 CPU-MoE note

Current upstream ExLlamaV3 has an experimental CPU-MoE executor for supported architectures. Its current eligibility contract includes:

- `mul1` experts;
- K <= 8;
- uniform per-expert bias presence;
- a model architecture ExLlamaV3 can instantiate.

Do **not** infer this checkpoint's codebook or per-tensor K from the `4.75bpw` repository name. Inspect the actual `config.json` and safetensors index first. The public recipe includes a fail-closed metadata checker for this purpose.

## Recommended quality validation

For any new runtime/topology, compare this pack against the released source checkpoint on the same prompts/corpus and report:

- PPL/NLL delta;
- deterministic token/logit agreement where available;
- prefill and decode throughput separately;
- speculative acceptance/throughput separately;
- GPU VRAM and host RAM/pinned-memory usage;
- Engram placement/storage behavior;
- actual EXL3 backend dispatch;
- exact checkpoint and runtime revisions.

A bit-exact source/reference implementation is especially valuable because it can distinguish a runtime/model-port error from EXL3 transcode loss.

## Related repositories

- `https://github.com/vcruz305/vllm-exl3`
- `https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe`

---

Until the Hugging Face model card itself is updated, this document is the public correction/superseding compatibility note for the runtime claims above.
