# TP2 + EP2: two DGX Sparks

TP2 is the aggressive DeepSeek-V4.1-Flash EXL3 target.

**Current artifact:** `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`  
**Locked snapshot:** 31 shards, immutable revision in `runtime.lock.json`.

## What changed

The current pinned `vllm-exl3` can represent **exact per-expert/per-projection K2–K8 trellis shapes**. TP2 therefore no longer needs a layer-uniform compatibility repack merely because the SAGE artifact contains tensor-level mixed K.

The remaining blocker is **capacity qualification**, not mixed-K representation.

## Geometry

- 2 Spark nodes / 2 GB10 GPUs
- TP2 + EP2
- 384 routed experts -> **192 whole experts/rank**
- expert matrix: **5120 × 2304**
- top-k: 6

192 local experts is not an ExLlamaV3 expert-count ceiling.

## Gate 0: host and remote metadata

```bash
bash scripts/doctor.sh 2
python3 scripts/probe_remote_pack.py --tp 2
```

The remote probe checks the immutable HF snapshot using metadata/header range reads only. Mixed K within one routed layer is now a supported layout.

## Gate 1: materialize the exact snapshot

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 2 \
  /large/models/DSV4.1-Flash-SAGE-EXL3-TP2
```

Keep the snapshot and Hugging Face/Xet caches on a sufficiently large filesystem.

## Gate 2: physical pack validation

```bash
python3 scripts/validate_pack.py \
  /large/models/DSV4.1-Flash-SAGE-EXL3-TP2 \
  --topology tp2 \
  --reserve-gib 32
```

The validator still fails closed on:

- missing/pointer/truncated shards;
- index/header disagreement;
- invalid offsets;
- dtype/shape byte-count mismatch;
- K outside the pinned K2–K8 capability;
- invalid source-format V4.1 metadata.

It records heterogeneous K instead of rejecting it.

Require:

```text
DEPLOYABLE_CURRENT_LOADER=YES
```

This means loader-format compatible only.

## Gate 3: memory contract

TP2 should be treated as a **nonresident-Engram** deployment. Resident or pinned-host Engram does not create new physical capacity on DGX Spark because Grace CPU and Blackwell GPU share the same LPDDR5x pool.

Do not infer TP2 viability from nominal `3.30 bpw` alone. Measure:

- actual non-Engram bytes owned by each rank;
- loader/materialization transients;
- vLLM/ExLlamaV3 workspaces;
- KV/cache allocations;
- node-local Engram staging;
- MemAvailable high-water mark.

If the full two-Spark body plus runtime does not leave safe headroom, a **smaller quant/reallocation** is still required—but that would be for capacity, not because the loader cannot represent mixed K.

## First boot

Keep the correctness profile:

```text
8K
seq1
1024 batched tokens
text-only
eager
DSpark off
native MoE off
```

Then qualify context only from measured headroom:

1. 8K
2. 32K
3. 64K
4. 128K only if memory receipts support it

Heterogeneous mixed-K is not CUDA-graph-qualified yet, so do not use graph-mode results as the first correctness evidence.

## Qualification evidence

Before calling TP2 deployable, retain:

1. exact HF revision and complete shard validation;
2. mixed-K histogram/layout receipt;
3. exact runtime lock;
4. two Ray GPU nodes and NCCL collective pass;
5. 192 owned experts/rank;
6. nonresident Engram path and staging budget;
7. successful full load;
8. `/v1/models` ready;
9. deterministic smoke/parity evidence;
10. per-node memory high-water mark;
11. actual EXL3 dispatch;
12. repeated decode stability;
13. DSpark only after the base path is stable.

`TP2_ALLOW_UNVALIDATED=1` remains a loader-development bypass only. Do not use it for benchmark/deployment claims.
