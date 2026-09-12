# TP2 + EP2: two DGX Sparks

TP2 is the aggressive target. Treat it as a separate checkpoint, storage, loader and memory problem rather than a smaller copy of TP4.

**Current SAGE source artifact:** [`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw)

> **Deployment status:** the currently published 3.30-bpw snapshot is not yet the recommended direct vLLM TP2 deployment pack. A lab campaign reported invalid/unmaterialized shard headers and tensor-level mixed K2-K8. Current `vllm-exl3` accepts K2-K8, but routed-MoE allocation still requires **one K per transformer layer**.

`serve_tp2.sh` therefore fails closed on a remote/unvalidated model by default.

## Geometry

- 2 Spark nodes / 2 GB10 GPUs total
- vLLM tensor parallel size: 2
- expert parallel: enabled
- 384 main routed experts -> **192 whole experts per rank**
- expert hidden/intermediate: **5120 × 2304**
- top-k routing: 6

**192 local experts is not an ExLlamaV3 fused-MoE expert-count limit.** The correctness-first TP2 backend is ExLlamaV3 (`NATIVE_MOE=0`). The custom native p2b path remains a separate K2-K4 experiment.

## Gate 0: host and storage doctor

Run on both Sparks before downloading weights:

```bash
bash scripts/doctor.sh 2
```

The doctor checks Docker/NVIDIA visibility, host architecture, filesystem headroom, RDMA discovery and the locked runtime contract.

## Gate 1: materialize intentionally

Do not use a plain Git checkout as proof that Hugging Face large objects are present.

The locked TP2 artifact is quarantined, so downloading it for recovery/repack work is explicit:

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 2 \
  /large/models/DSV4.1-Flash-SAGE-EXL3-TP2
```

The generic materializer:

- uses `hf download`, `huggingface-cli`, or `huggingface_hub.snapshot_download`;
- uses the locked model revision when one is available;
- defaults to a conservative **550 GiB** pre-download free-space gate;
- keeps `HF_HOME`, Hub cache and Xet cache on the same large filesystem;
- runs the physical pack validator after download.

`scripts/materialize_tp2.sh` remains as a backwards-compatible wrapper around this generic path.

If a Spark root filesystem is short by ~60 GB, move the snapshot **and** Hugging Face/Xet caches to a larger NVMe/shared mount. Freeing only enough space to finish a download does not solve loader or unified-memory compatibility.

## Gate 2: validate shards and K geometry

Run:

```bash
python3 scripts/validate_pack.py \
  /large/models/DSV4.1-Flash-SAGE-EXL3-TP2 \
  --topology tp2 \
  --reserve-gib 32
```

The validator checks:

- indexed shard count and missing shards;
- Git LFS/Xet pointer stubs;
- HTML/XML accidentally saved as model files;
- truncated/invalid safetensors headers;
- index entries missing from their declared shard;
- unindexed tensors present in shard headers;
- out-of-range `data_offsets`;
- known dtype/shape byte counts against declared tensor byte ranges;
- EXL3 trellis K histogram inferred from physical geometry;
- routed layers containing more than one K;
- per-shard header SHA256 values;
- materialized bytes and filesystem reserve.

Require:

```text
DEPLOYABLE_CURRENT_LOADER=YES
```

If the reported bad shards are pointer stubs, proper materialization can fix that blocker without requantizing. If they remain genuinely truncated/invalid after correct materialization, those affected files need to be rebuilt/re-uploaded.

## Gate 3: SAGE compatibility repack

Current vLLM supports mixed precision **between transformer layers**, but one `RoutedExperts` layer allocates one trellis K width for all routed expert projections.

The fastest compatibility path is not global uniform K. Preserve K2-K8 variation across the 40 transformer layers while coalescing all routed tensors within each layer to one K:

```bash
# in vcruz305/SAGE-EXL3
python tools/coalesce_v41_vllm_recipe.py \
  /path/to/exact-tp2-sage-recipe.yaml \
  --out recipes/recipe-tp2-vllm-layer-uniform.yaml \
  --min-k 2 \
  --max-k 8
```

Changing K requires a real re-encode. Use the encode bank to reuse exact `(tensor key, K)` variants and encode only missing deltas. A metadata edit is not enough.

## Memory contract: streamed/nonresident Engram

TP2 should not be qualified with resident Engram as if it were a smaller TP4 run. The SAGE TP2 direction is **streamed/nonresident Engram/PLE**.

The 4.75-bpw TP4 body is not the normal TP2 body. At roughly 233 GiB excluding the large Engram/PLE component, a two-way split would be about **116.5 GiB body per Spark** before runtime/KV/workspaces, which is not a practical GB10 operating point.

The corrected TP2 repack should be judged by actual final body bytes, not nominal bpw alone. A useful target remains roughly 170–175 GiB of non-Engram body if quality allows.

CPU/UVA offload does not create a second physical memory pool on DGX Spark. Grace CPU and Blackwell GPU share the same LPDDR5x.

## Safe TP2 deployment sequence

After a corrected layer-uniform snapshot exists:

1. materialize the exact snapshot on sufficiently large storage on both Sparks;
2. require `validate_pack.py` to pass;
3. build the exact `runtime.lock.json` runtime;
4. run `preflight.sh 2`;
5. start with ExLlamaV3, text-only, eager, seq1, DSpark off;
6. prove 8K;
7. then 32K;
8. then 64K;
9. attempt 128K only with measured unified-memory headroom.

A first boot can use:

```bash
MODEL_DIR=/large/models \
MODEL=/models/DSV4.1-Flash-SAGE-EXL3-TP2-vllm \
bash scripts/serve_tp2.sh
```

The locked/default profile already resolves to:

```text
8K / seq1 / 1024 batched tokens / text-only / eager / DSpark off / native off
```

`TP2_ALLOW_UNVALIDATED=1` exists only for loader-development experiments. Never use it for deployment or benchmark claims.

## Qualification evidence

Before calling TP2 usable, retain evidence for:

1. fully materialized/structurally valid safetensors;
2. layer-uniform routed K2-K8 layout;
3. checkpoint/repack provenance and exact runtime lock;
4. both Ray GPUs visible;
5. 192 main experts per rank;
6. nonresident Engram behavior identified;
7. deterministic output parity against a known-good V4.1 baseline;
8. no OOM during load and first generation;
9. repeated decode stability;
10. per-rank unified-memory high-water mark;
11. actual EXL3 backend dispatch;
12. context progression, with 128K unverified until measured;
13. DSpark only after the non-speculative baseline is stable.

## Performance reporting

Report per-request decode speed and aggregate throughput. Compare TP2 and TP4 only when checkpoint identity, K schedule, Engram strategy, context, prompt, output length, DSpark policy, graph mode and sampling are recorded.
