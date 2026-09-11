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

## New preferred first experiment: keep the V4.1 graph in current vLLM

Current upstream vLLM now gives us a shorter path than porting the entire V4.1 graph into standalone ExLlamaV3 before the first host-memory test:

- V4.1 Engram has an explicit CPU/UVA placement mode;
- generic model weights can be selectively put in pinned host memory and exposed as mapped accelerator views;
- `vllm-exl3` can build its packed expert handles/pointer tables after that placement.

The new experimental shape is:

```text
V4.1 CED / attention / routing        vLLM on GPU
Engram                                vLLM pinned-host UVA
EXL3 routed-expert packed weights     vLLM pinned-host UVA
EXL3 expert math                      GPU kernels reading mapped host memory
```

This is **not** the same as your existing CPU-expert harness. The purpose of the first run is to establish whether the EXL3 GPU kernels can execute correctly and usefully over UVA on your `sm_120` machine. If it is correct but too PCIe-bound, the standalone ExLlamaV3 + CPU-MoE route remains the next target.

See [`SM120_UVA.md`](SM120_UVA.md) for the exact experiment.

## Gate 0: identify the actual pack

On the downloaded checkpoint run:

```bash
python scripts/check_checkpoint_compat.py /models/DSV4.1-Flash-EXL3-4.75bpw --json | tee compat.json
```

Preserve `config.json`, `model.safetensors.index.json`, the Hugging Face revision and file hashes with the result.

Do not assume the codebook from the average `4.75bpw` name. The codebook matters for the separate ExLlamaV3 CPU-compute route, even though the vLLM UVA experiment uses the existing GPU EXL3 kernels.

## Gate 1: current-vLLM / sm_120 / UVA capability preflight

Use a current V4.1-capable vLLM environment, not the older Spark-pinned container, and install a `vllm-exl3` revision containing the UVA guard.

Run:

```bash
python scripts/preflight_sm120_uva.py | tee sm120-uva-preflight.json
```

The preflight must prove:

- compute capability `(12, 0)`;
- vLLM reports UVA available;
- `EngramConfig.cpu_offload` exists;
- selective UVA weight offload exists;
- the current V4.1 NVIDIA model class imports;
- the `vllm-exl3` UVA placement guard is installed.

A pass here only proves runtime capability. It does not allocate the model.

## Gate 2: first UVA model load

Start conservatively:

```bash
MODEL=/models/DSV4.1-Flash-EXL3-4.75bpw \
CPU_OFFLOAD_GB=300 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.80 \
./scripts/serve_sm120_uva.sh
```

The launcher is text-only, TP1, eager, batch 1, DSpark off and requires all six large EXL3 expert payloads to be vLLM UVA-mapped:

```text
w13_trellis  w13_suh  w13_svh
w2_trellis   w2_suh   w2_svh
```

Engram is separately requested in vLLM's CPU/UVA mode.

Expect very large pinned-memory pressure. The model can require roughly 400–500 GiB-class mapped host state once expert payload, Engram and runtime overhead are included. A 1 TiB machine has raw capacity, but CUDA/OS pinning limits still have to accept it.

If the first load fails, please send the **first** traceback plus both preflight JSON files before changing flags.

## Gate 3: short correctness comparison

If model load succeeds:

1. keep DSpark disabled;
2. use batch 1;
3. use short fixed prompts;
4. use deterministic/greedy decoding;
5. compare token-by-token and, where your harness permits, logits/NLL against the bit-exact source implementation;
6. capture actual EXL3 backend dispatch, GPU VRAM, host RSS/pinned memory, PCIe traffic and storage activity.

Do not benchmark long context before this gate passes.

## Gate 4: performance and scale-up

Only after base parity is acceptable:

1. record base decode tok/s;
2. enable the source-equivalent DSpark path and record acceptance/output agreement;
3. test 48K prefill;
4. increase context to 128K and then toward 512K from measured memory/cache headroom;
5. benchmark steady-state decode separately from prefill;
6. test restart/reload and long-running stability.

## Fallback path: standalone ExLlamaV3 + CPU-MoE

If the vLLM UVA path is too PCIe-bound, cannot pin enough memory, or exposes a packed-kernel incompatibility, the next route is the one originally identified in your report:

1. complete a **forward-correct** standalone `DeepseekV41ForCausalLM` implementation in ExLlamaV3;
2. keep Engram mmap/host-backed;
3. reuse ExLlamaV3 CPU-MoE for the routed experts if the exact pack satisfies its format constraints.

A compatibility alias to `DeepseekV4ForCausalLM` is forbidden. V4.1 has different compression/source relationships and Engram semantics.

Current upstream ExLlamaV3 CPU-MoE is conditional on:

- `mul1` codebook;
- K <= 8;
- uniform per-expert bias presence;
- a loadable ExLlamaV3 architecture.

If the 4.75bpw release is not eligible, choose one explicit follow-up rather than silently converting formats:

- produce and quality-test a CPU-offload-oriented sibling release; or
- add the missing CPU expert backend for the actual codebook and qualify it independently.

SAGE-EXL3 now contains a fail-closed V4.1 architecture contract plus an auditable 40-layer CED source/ownership plan, but its conversion-only V4.1 class still deliberately has no forward implementation. That distinction is intentional.

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
| peak host RAM / pinned memory | | | |
| PCIe traffic | | | |
| storage read behavior | | | |

Use the same tokenizer, prompts, context and sampling policy for both sides.

## Evidence to return with a result

Please include:

```text
checkpoint HF revision / hashes
vLLM revision
vllm-exl3 revision
ExLlamaV3 revision used by the EXL3 kernels
Python / Torch / CUDA versions
GPU model + compute capability
CPU model / NUMA topology / RAM channels
UVA/offload flags and budget
Engram placement
DSpark state
context / batch / prefill settings
actual EXL3 backend dispatch
PPL / NLL or parity receipt
throughput receipt
host pinned-memory / PCIe receipt
```

A failed gate is useful evidence. Report the first failing gate rather than patching around it invisibly.
