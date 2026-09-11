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

## The 4.75-bpw TP4 pack is not the recommended TP2 pack

If the full checkpoint is approximately **422 GiB** on disk because it contains about **189 GiB of Engram/PLE** plus about **233 GiB of non-PLE model body**, putting Engram/PLE on NVMe changes the TP2 capacity math to the model body only:

- TP4 + disk PLE: ~233 / 4 = **58.25 GiB model body per Spark**
- TP2 + disk PLE: ~233 / 2 = **116.5 GiB model body per Spark**

A GB10 has 128 GB coherent unified system memory and commonly exposes roughly 121-122 GiB to CUDA. CPU, GPU, the OS, page cache and runtime all share that physical pool. Therefore ~116.5 GiB/rank leaves effectively no usable space for vLLM workspaces, sparse-attention/indexer state, CUDA graphs, KV cache, Ray, DSpark or ordinary OS headroom.

The same 4.75-bpw pack can remain a diagnostic/bring-up attempt, but it should **not** be advertised as the normal TP2 recipe.

## Recommended TP2 quantization target

Generate a separate TP2 EXL3 checkpoint and optimize to a **byte budget**, not only a nominal bpw.

Recommended first target for the non-PLE body:

- **goal: 170-175 GiB total body**
- per Spark under TP2: **85-87.5 GiB/rank**
- stretch ceiling for early experiments: about **180 GiB body / 90 GiB per rank**

That leaves materially more unified-memory headroom for the runtime while still keeping Engram/PLE on NVMe.

If size scaled perfectly linearly from the current 233 GiB / 4.75-bpw body, 170-175 GiB would correspond very roughly to **3.47-3.57 bpw**, and 180 GiB to about **3.67 bpw**. Treat those only as planning estimates: dense MXFP8 weights, embeddings, scales and other fixed-format tensors do not shrink linearly with the routed-expert EXL3 K schedule. SAGE should therefore target the final body byte size directly and spend precision on the most sensitive layers.

## CPU offload on DGX Spark

Do **not** rely on `--cpu-offload-gb` as the TP2 capacity solution.

On a normal discrete-GPU server, vLLM CPU offload can increase effective GPU capacity because GPU VRAM and host RAM are separate physical pools. DGX Spark is different: Grace CPU and Blackwell GPU share the same 128 GB LPDDR5x coherent unified memory. vLLM's UVA offloader places parameters in pinned CPU memory and exposes accelerator views, but those pages still consume the same physical Spark memory pool.

CPU/UVA offload may still be useful as an experimental access-policy or allocator test, but it does **not** turn a 116.5 GiB/rank checkpoint into a comfortably sized TP2 deployment. It is disabled in this recipe by default.

The useful capacity offload for V4.1 on Spark is **NVMe-backed data that is not kept resident**, especially the ~189 GiB Engram/PLE tables. Any additional NVMe/layer streaming variant must be documented and benchmarked separately because it can materially change latency and prefill throughput.

## Checkpoint requirement

Before serving, measure actual per-rank residency including:

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

Start conservative:

```bash
GPU_MEMORY_UTILIZATION=0.75 DSPARK=0 EAGER=1 TEXT_ONLY=1 ./scripts/serve_tp2.sh
```

This is still **experimental** until the full V4.1 geometry passes GB10 numerical parity and real-checkpoint serving.

For a conservative fallback/control attempt:

```bash
NATIVE_MOE=0 GPU_MEMORY_UTILIZATION=0.75 DSPARK=0 EAGER=1 ./scripts/serve_tp2.sh
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

Source V4.1 Engram is enormous relative to a two-Spark memory budget. A successful TP2 recipe requires the Engram/PLE tables to remain non-resident or use another explicitly documented backing strategy. Do not hide that modification inside the EXL3 recipe; report its latency/prefill impact.

## Performance reporting

Report both per-request decode speed and aggregate throughput. A two-node result is only comparable to TP4 when the checkpoint revision, K schedule, context, prompt, output length, DSpark policy, graph mode and sampling are identical.
