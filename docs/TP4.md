# TP4 + EP4: four DGX Sparks

This is the preferred first qualification path for DeepSeek-V4.1-Flash EXL3.

**Published TP4 checkpoint:** [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw)

## Geometry

- 4 Spark nodes / 4 GB10 GPUs total
- vLLM tensor parallel size: 4
- expert parallel: enabled
- 384 main routed experts -> **96 whole experts per rank**
- expert hidden/intermediate: **5120 x 2304** on every rank
- top-k routing: 6

Both dimensions are multiples of 128 and 96 local experts are below the current ExLlamaV3 fused-MoE 128-expert envelope.

## First gate: resident-Engram minimum fit

Before using the normal 65K defaults, prove the resident model can fit at all with large-context and batch pressure removed.

First verify the exact command **without loading the model**:

```bash
bash scripts/tp4_min_fit.sh --check
```

The printed launch must show all of these:

```text
Max model len:          8192
Max num seqs:           1
Max batched tokens:     1024
Native V4.1 MoE:        0
DSpark:                 0
Eager:                  1
Text only:              1
```

Then monitor all four Ray nodes from a second terminal:

```bash
bash scripts/watch_cluster_memory.sh
```

And run the actual min-fit load:

```bash
bash scripts/tp4_min_fit.sh
```

The harness intentionally fixes **8K / seq1 / text-only / eager / DSpark off / native off**. `MIN_FIT_GPU_MEMORY_UTILIZATION` may be overridden if needed, but do not turn this into a memory-knob sweep.

### Decision rule

If the true 8K/seq1 run loads and serves, resident TP4 is technically viable and the larger default context/batch needs separate tuning.

If the true 8K/seq1 run still drives one or more Sparks into the roughly full-system-memory cliff during Engram load and a node becomes unresponsive, record:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

At that point stop tuning context/batch/memory-utilization. The next engineering task is a clearly identified **non-resident / disk-backed Engram** variant.

## Baseline after min-fit passes

Start with:

```bash
NATIVE_MOE=0 DSPARK=0 EAGER=1 TEXT_ONLY=1 bash scripts/serve_tp4.sh
```

This requests the ExLlamaV3 expert backend and isolates EXL3 checkpoint/load correctness from the new ABI-3 native kernel.

Required evidence before moving on:

- all four Ray GPU resources visible;
- 96 EXL3 expert packs owned per main-stack rank;
- dense/attention/shared weights delegated to the V4.1 source MXFP8 path;
- DSpark source blocks are not mistaken for main EXL3 experts;
- finite logits/output and repeatable deterministic smoke prompt;
- per-rank unified-memory high-water mark captured;
- no persistent BF16 reconstruction of the entire routed expert bank.

## Native p2b A/B

Once the control is correct:

```bash
NATIVE_MOE=1 DSPARK=0 EAGER=1 TEXT_ONLY=1 bash scripts/serve_tp4.sh
```

This requires native MoE ABI >= 3 and enables the 5120 x 2304 candidate geometry. Hold the model revision, prompt, sampling, context and batch fixed.

Compare:

- output parity / deterministic token stream;
- actual EXL3 backend dispatch;
- decode tok/s;
- TTFT and prefill tok/s;
- peak unified memory;
- GPU power/clocks;
- stability over repeated requests.

Do not make a performance claim from the kernel microbenchmark alone.

## DSpark-5

After the expert backend is stable:

```bash
DSPARK=1 NATIVE_MOE=0 EAGER=1 bash scripts/serve_tp4.sh
```

Then repeat with `NATIVE_MOE=1` if desired. The recipe uses the trained 5-token DSpark block and begins with adaptive verification disabled on Spark.

Record proposed tokens, accepted tokens, mean accepted length and output speed. DSpark experts remain in their source format.

## CUDA graphs

Only after eager mode works:

```bash
EAGER=0 bash scripts/serve_tp4.sh
```

Warm/compile all required runtime kernels before interpreting graph performance. Keep `VLLM_USE_BREAKABLE_CUDAGRAPH=1` in the recipe environment.

## Context progression

Only after resident minimum-fit and the 65K baseline are stable:

1. 65,536
2. 131,072
3. 300,000
4. longer contexts only with measured memory headroom

V4.1 advertises 1M context, but model weights, Engram, graphs, workspaces and KV compete for the same Spark unified memory. Do not infer a practical context ceiling from the model config alone.

## Engram

The resident baseline is a qualification experiment, not an assumption. Keep any disk/node-local Engram variant separately identified so EXL3 memory savings and Engram I/O costs remain measurable.
