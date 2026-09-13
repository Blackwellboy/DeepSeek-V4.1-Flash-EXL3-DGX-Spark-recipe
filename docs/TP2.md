# TP2 on two DGX Sparks

TP2 is the aggressive DeepSeek-V4.1-Flash EXL3 target.

**Current artifact:** `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`  
**Locked snapshot:** 31 shards, immutable revision in `runtime.lock.json`.

## Current status

The pinned `vllm-exl3` can represent **exact per-expert/per-projection K2–K8 trellis shapes**. TP2 no longer needs a layer-uniform repack merely because the SAGE artifact contains tensor-level mixed K.

The remaining blocker is **two-Spark capacity and runtime qualification**, not mixed-K representation.

## Two MoE topology candidates

### Baseline: TP2 + EP2

- 384 routed experts -> **192 whole experts/rank**
- expert matrix per rank: **5120 × 2304**
- top-k: 6

This is the correctness-first baseline.

### Experimental A/B: pure MoE TP2 (EP1)

- each rank participates in all 384 experts;
- local expert matrix: **5120 × 1152**;
- 1152 is exactly `9 × 128`, so no extra width padding is needed.

This candidate is motivated by SGLang's observation that MoE TP can reduce inter-rank waiting versus EP when expert routing is uneven. Their published TP4 experiment needed 576->640 padding; our TP2 local width is already 128-aligned, so that specific padding penalty does not apply. This is still an A/B hypothesis, not a performance claim for Spark.

Select with:

```bash
MOE_PARALLEL_MODE=ep  # baseline
MOE_PARALLEL_MODE=tp  # experimental pure MoE TP2
```

Pure MoE TP4 remains blocked by default because this recipe has not implemented/qualified 576->640 EXL3 padding.

## Gate 0: host and remote metadata

```bash
bash scripts/doctor.sh 2
python3 scripts/probe_remote_pack.py --tp 2
```

The remote probe checks the immutable HF snapshot using metadata/header range reads only. Mixed K within one routed layer is a supported layout.

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

The validator still fails closed on missing/pointer/truncated shards, index/header disagreement, invalid offsets, dtype/shape byte-count mismatch, K outside the pinned K2–K8 capability, and invalid source-format V4.1 metadata.

Require:

```text
DEPLOYABLE_CURRENT_LOADER=YES
```

This means loader-format compatible only.

## Gate 3: nonresident Engram

TP2 should be treated as a **disk-backed/nonresident-Engram** deployment. Resident or pinned-host Engram does not create new physical capacity on DGX Spark because Grace CPU and Blackwell GPU share the same LPDDR5x pool.

Build the derived runtime once:

```bash
bash scripts/build_disk_engram_runtime.sh
```

Start both Spark containers with the TP2 profile, setting the same in-container model path on each host:

```bash
export DISK_ENGRAM_PROFILE="$PWD/profiles/tp2-disk-engram.env"
export MODEL=/models/DSV4.1-Flash-SAGE-EXL3-TP2
export VLLM_ENGRAM_MODEL_DIR="$MODEL"

bash scripts/start_disk_engram_cluster.sh head
# on Spark 2: bash scripts/start_disk_engram_cluster.sh worker
```

Then use the guarded min-fit path:

```bash
MOE_PARALLEL_MODE=ep bash scripts/tp2_disk_engram_min_fit.sh --check
MOE_PARALLEL_MODE=ep bash scripts/tp2_disk_engram_min_fit.sh
```

Only after the EP2 baseline succeeds should the same model/profile be tested with:

```bash
MOE_PARALLEL_MODE=tp bash scripts/tp2_disk_engram_min_fit.sh
```

Change one variable at a time.

## Gate 4: V4.1 cache accounting

Do **not** use the older V4 43-layer MLA helper for V4.1 context claims. V4.1 has four shared persistent global-cache producers (layers 2, 8, 14, 20), with the first three pooled 2:1. The logical global KV+index floor is 890 bytes/original token, but actual vLLM allocation may be higher because of backend layout, local SWA, padding, metadata and workspaces.

Generate a lower-bound receipt inside the pinned runtime:

```bash
docker exec dsv41-exl3 python /recipe/scripts/v41_context_receipt.py \
  --context 8192 \
  --model-resident-gib <MEASURED_MODEL_GIB>
```

A receipt without `--backend-bytes-per-token` is **not** a capacity qualification. Once actual cache allocation is measured, rerun with:

```bash
--backend-bytes-per-token <MEASURED_BYTES_PER_TOKEN>
```

## First boot

Keep the correctness profile fixed:

```text
8K
seq1
1024 batched tokens
text-only
eager
DSpark off
native MoE off
disk-backed Engram
```

Heterogeneous mixed-K is not CUDA-graph-qualified yet, so do not use graph-mode results as the first correctness evidence.

After a successful boot, capture actual dispatch evidence:

```bash
bash scripts/kernel_dispatch_receipt.sh > kernel-dispatch.txt
```

Then qualify context only from measured headroom:

1. 8K
2. 32K
3. 64K
4. 128K only if memory receipts support it

## Qualification evidence

Before calling TP2 deployable, retain:

1. exact HF revision and complete shard validation;
2. mixed-K histogram/layout receipt;
3. exact runtime lock;
4. two Ray GPU nodes and NCCL collective pass;
5. topology identity (`EP2` or pure `MoE-TP2`);
6. nonresident Engram path and staging budget;
7. successful full load;
8. `/v1/models` ready;
9. deterministic smoke/parity evidence;
10. per-node memory high-water mark;
11. actual EXL3/kernel dispatch receipt;
12. repeated decode stability;
13. EP2-vs-TP2 A/B only after both are individually correct;
14. DSpark only after the base path is stable.

`TP2_ALLOW_UNVALIDATED=1` remains a loader-development bypass only. Do not use it for benchmark/deployment claims.
