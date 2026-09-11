# Lna-Lab / TonoKen3 interoperability validation

This protocol captures the compatibility gap reported for `vcruz305/DSV4.1-Flash-EXL3-4.75bpw` on a large-host-memory system with `sm_120` Blackwell GPUs.

The goal is not to force the existing TP4 recipe onto that machine. The goal is to establish the smallest correct path to compare the EXL3 transcode against a bit-exact original DeepSeek-V4.1-Flash baseline.

## Reported reference topology

- 10 x RTX PRO 2000 Blackwell, 16 GB each, `sm_120`;
- 1 TiB DDR5;
- Optane/NVMe scratch;
- original V4.1 routed experts resident/mmap'd in host RAM and computed on CPU;
- attention/dense path on a GPU;
- Engram tables mmap'd from local storage;
- a bit-exact source/reference implementation is available for comparison.

This topology is not equivalent to the TP4+EP4 DGX Spark target.

## Gate 0: identify the actual pack

On the downloaded checkpoint run:

```bash
python scripts/check_checkpoint_compat.py /models/DSV4.1-Flash-EXL3-4.75bpw --json | tee compat.json
```

Preserve `config.json`, `model.safetensors.index.json`, the Hugging Face revision and file hashes with the result.

Do not assume the codebook from the average `4.75bpw` name. CPU-MoE eligibility depends on the actual EXL3 codebook and per-tensor K metadata.

## Gate 1: architecture ownership

There are two distinct paths.

### vLLM path

Use the dedicated DeepSeek-V4.1 vLLM model implementation for CED/CSA2/Engram and layer `vllm-exl3` onto it. This is the current recipe architecture, but it does not provide host-resident CPU execution for EXL3 experts.

### ExLlamaV3 path

Use ExLlamaV3's existing CPU-MoE machinery, but only after a **forward-correct** `DeepseekV41ForCausalLM` architecture exists. The SAGE repository currently contains a conversion-only V4.1 class whose forward deliberately raises; that class is not sufficient for inference.

A compatibility alias to `DeepseekV4ForCausalLM` is forbidden. V4.1 uses different compression/source relationships and Engram semantics.

## Gate 2: CPU-MoE format eligibility

Current upstream ExLlamaV3 CPU-MoE is conditional on:

- `mul1` codebook;
- K <= 8;
- uniform per-expert bias presence;
- a loadable ExLlamaV3 architecture.

If the 4.75bpw release is MCG, stop here for the upstream CPU-MoE path. Choose one of these explicit follow-ups rather than silently converting formats:

1. produce and quality-test a CPU-offload-oriented `mul1` sibling release; or
2. add an MCG CPU expert kernel/backend to ExLlamaV3 and qualify it independently.

Do not label either path equivalent until the original-vs-EXL3 quality comparison passes.

## Gate 3: V4.1 forward port

Minimum text-only forward scope before asking Lna-Lab to run the pack:

- exact 40-layer compression schedule (`0`, `2`, `1` semantics);
- KV source layers 2/8/14/20;
- index source layers 2/8/14/20/24/28/32/36;
- candidate source layer 20, top-k blocks and block-size behavior;
- Engram at layers 1 and 14;
- mHC residual behavior;
- routed/shared expert behavior;
- correct source/compiled tensor namespace;
- tokenizer and text generation path.

DSpark and vision should be staged after base text parity. A base-text loader that produces tokens but skips Engram/CED is not a valid V4.1 implementation.

## Gate 4: first Lna-Lab run

Start with the smallest comparison that can falsify the port quickly:

1. text-only;
2. batch 1;
3. short fixed prompts;
4. deterministic/greedy decoding;
5. DSpark disabled;
6. compare token-by-token or logits where the harness exposes them;
7. capture CPU RAM, GPU VRAM, storage reads and actual expert backend.

Only after base parity is acceptable:

1. enable the source-equivalent speculative path;
2. test 48K prefill;
3. increase context in steps toward 512K;
4. benchmark steady-state decode separately from prefill;
5. test restart/reload and long-running stability.

## Quality A/B

The source experts are already low precision. The EXL3 question is whether re-encoding adds material error.

For each corpus/prompt set record:

| Metric | Original released checkpoint | EXL3 4.75bpw | Delta |
|---|---:|---:|---:|
| PPL / NLL | | | |
| greedy token agreement | | | |
| base tok/s | | | |
| speculative tok/s | | | |
| prefill ms/token | | | |
| peak GPU VRAM | | | |
| peak host RAM | | | |
| storage read behavior | | | |

Use the same tokenizer, prompts, context, sampling policy and reference kernels for both sides.

## Evidence to return with a result

Please include:

```text
checkpoint HF revision / hashes
ExLlamaV3 or vLLM revision
vllm-exl3 revision if used
Python / Torch / CUDA versions
GPU model + compute capability
CPU model / NUMA topology / RAM channels
CPU-MoE flags and worker/thread settings
Engram placement
DSpark state
context / batch / prefill settings
actual backend dispatch
PPL or parity receipt
throughput receipt
```

A failed gate is useful evidence. Report the first failing gate rather than patching around it invisibly.
