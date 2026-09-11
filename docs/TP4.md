# TP4 + EP4: four DGX Sparks

This is the preferred first qualification path for DeepSeek-V4.1-Flash EXL3.

## Geometry

- 4 Spark nodes / 4 GB10 GPUs total
- vLLM tensor parallel size: 4
- expert parallel: enabled
- 384 main routed experts -> **96 whole experts per rank**
- expert hidden/intermediate: **5120 x 2304** on every rank
- top-k routing: 6

Both dimensions are multiples of 128 and 96 local experts are below the current ExLlamaV3 fused-MoE 128-expert envelope.

## Baseline

Start with:

```bash
NATIVE_MOE=0 DSPARK=0 EAGER=1 TEXT_ONLY=1 ./scripts/serve_tp4.sh
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
NATIVE_MOE=1 DSPARK=0 EAGER=1 TEXT_ONLY=1 ./scripts/serve_tp4.sh
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
DSPARK=1 NATIVE_MOE=0 EAGER=1 ./scripts/serve_tp4.sh
```

Then repeat with `NATIVE_MOE=1` if desired. The recipe uses the trained 5-token DSpark block and begins with adaptive verification disabled on Spark.

Record proposed tokens, accepted tokens, mean accepted length and output speed. DSpark experts remain in their source format.

## CUDA graphs

Only after eager mode works:

```bash
EAGER=0 ./scripts/serve_tp4.sh
```

Warm/compile all required runtime kernels before interpreting graph performance. Keep `VLLM_USE_BREAKABLE_CUDAGRAPH=1` in the recipe environment.

## Context progression

Recommended qualification gates:

1. 65,536
2. 131,072
3. 300,000
4. longer contexts only with measured memory headroom

V4.1 advertises 1M context, but model weights, Engram, graphs, workspaces and KV compete for the same Spark unified memory. Do not infer a practical context ceiling from the model config alone.

## Engram

Try the clean resident-Engram baseline first. If it OOMs, record the exact high-water mark before introducing a disk/node-local Engram variant. Keep that variant separately identified so EXL3 savings are measurable.
