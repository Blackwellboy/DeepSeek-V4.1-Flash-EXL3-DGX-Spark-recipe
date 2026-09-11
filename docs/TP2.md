# TP2 + EP2: two DGX Sparks

TP2 is the aggressive target. Treat it as a separate checkpoint/memory problem rather than a smaller copy of TP4.

## Geometry

- 2 Spark nodes / 2 GB10 GPUs total
- vLLM tensor parallel size: 2
- expert parallel: enabled
- 384 main routed experts -> **192 whole experts per rank**
- expert hidden/intermediate: **5120 x 2304** on every rank
- top-k routing: 6

The dimensions are 128-aligned, but 192 local experts exceed the current ExLlamaV3 fused-MoE 128-expert envelope.

## Checkpoint requirement

A pack that is comfortable on TP4 may still be too large on TP2. Before serving, measure the actual checkpoint and estimate per-rank residency including:

- EXL3 main routed experts;
- source-format DSpark experts;
- source MXFP8 dense/attention/router weights;
- embedding/lm head;
- Engram tables or their chosen backing strategy;
- quantization metadata/scales;
- runtime workspaces and CUDA graphs;
- KV cache.

Do not advertise a two-Spark fit from weight size alone.

## First boot

The TP2 wrapper defaults to `NATIVE_MOE=1` because ABI-3 p2b can represent 5120 x 2304 and does not use the ExLlamaV3 128-local-expert fused ceiling.

```bash
DSPARK=0 EAGER=1 TEXT_ONLY=1 ./scripts/serve_tp2.sh
```

This is still **experimental** until the full V4.1 geometry passes GB10 numerical parity and real-checkpoint serving.

For a conservative fallback/control attempt:

```bash
NATIVE_MOE=0 DSPARK=0 EAGER=1 ./scripts/serve_tp2.sh
```

With native disabled, the plugin may fall back to a slower applicable path. That is useful for correctness comparison but not a performance target.

## Qualification gates

Before calling TP2 usable, retain evidence for:

1. checkpoint preflight and ABI 3;
2. Ray sees exactly/at least two GPUs;
3. each rank owns 192 main experts;
4. deterministic output parity against a known-good V4.1 baseline;
5. no OOM during load and first generation;
6. repeated decode stability;
7. per-rank unified-memory high-water mark;
8. actual EXL3 backend dispatch;
9. 64K context stability;
10. DSpark only after the non-speculative baseline is stable.

## Engram warning

Source V4.1 Engram is enormous relative to a two-Spark memory budget. A successful TP2 recipe may require a separately documented disk/node-local Engram strategy. Do not hide that modification inside the EXL3 recipe; make it an explicit variant and report its latency/prefill impact.

## Performance reporting

Report both per-request decode speed and aggregate throughput. A two-node result is only comparable to TP4 when the checkpoint revision, K schedule, context, prompt, output length, DSpark policy, graph mode and sampling are identical.
