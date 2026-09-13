# Disk-backed Engram qualification

Disk-backed Engram is the current **TP4 DGX Spark capacity path**. It remains isolated from the baseline image so storage/I/O costs stay explicit and measurable.

## Why

Corrected resident testing at 8K / seq1 / eager / text-only produced:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
~121 GiB used class
~0.5 GiB MemAvailable class near the failure cliff
```

On GB10, pinned host memory and GPU memory consume the same physical LPDDR5x pool, so `cpu_offload=True` does not solve this capacity problem.

## What the disk path does

The overlay:

- does **not** allocate the full Engram embedding table as a resident Parameter;
- skips full `engram.embed.weight/scale` materialization in the weight iterator;
- keeps the safetensors backing on node-local storage;
- stages only unique rows required for the current lookup into bounded pinned-host and GPU buffers;
- reuses vLLM's FP8/ue8m0 Engram dequant semantics;
- keeps disk mode explicit through `EngramConfig.disk_backed` / `VLLM_ENGRAM_DISK_BACKED=1`.

Offline/hardware qualification reported by @Blackwellboy:

| Gate | Result |
|---|---|
| synthetic lookup parity | PASS |
| real checkpoint slice | PASS, bit-exact |
| repeated staging/reclaim stress | PASS |
| TP ownership checks | PASS |
| resident full Engram | not materialized |

Those are storage-path results, **not** a full-model serving claim.

## Mixed-K dependency

The companion mixed-K work is no longer a placeholder. `vllm-exl3` PR #10 by @Blackwellboy is merged and pinned through `runtime.lock.json`.

The loader retains exact per-expert/per-projection K3–K8 trellis geometry. Heterogeneous layers use a correctness-first `LinearEXL3` loop and stay eager-first until CUDA-graph qualification exists.

## Build

Build the locked baseline first, then the derived disk image:

```bash
bash scripts/build_runtime.sh
bash scripts/build_disk_engram_runtime.sh
```

Output image:

```text
deepseek-v41-exl3:disk-engram
```

`Dockerfile.disk-engram` applies only the explicit overlay files to the already locked Spark runtime and performs import/syntax assertions. Manual site-packages editing is no longer the recommended path.

## Materialize the TP4 snapshot

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-TP4
```

Use the same in-container path on every Spark, for example:

```text
/models/DSV4.1-Flash-EXL3-TP4
```

## Start all four Ray nodes in disk mode

Example head:

```bash
IMAGE=deepseek-v41-exl3:disk-engram \
MODEL_DIR=/large/models \
MODEL=/models/DSV4.1-Flash-EXL3-TP4 \
VLLM_ENGRAM_MODEL_DIR=/models/DSV4.1-Flash-EXL3-TP4 \
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.10 \
  bash scripts/start_disk_engram_cluster.sh head
```

Run `worker` on the other three nodes. The wrapper ensures disk-Engram and EXL3 model-path environment exists in each Ray container before vLLM workers are created.

## All-node preflight

The first-load wrapper invokes:

```text
scripts/check_disk_engram_cluster.py
```

and fails unless every selected Ray GPU node has:

- disk mode enabled;
- matching model/index path;
- overlay import available;
- Engram weight-loader skip active;
- mixed-K-capable plugin;
- non-network backing storage.

## OOM guard

Default thresholds:

```text
WARN:  24 GiB MemAvailable
ABORT: 16 GiB MemAvailable
```

The guard targets **one exact configured container name**. It no longer pattern-matches arbitrary `dsv41*` or `deepseek-v41*` containers.

Arm it on every node before a real load.

## First load

```bash
bash scripts/tp4_disk_engram_min_fit.sh --check
bash scripts/tp4_disk_engram_min_fit.sh
```

This path is locked to:

```text
8K / seq1 / 1024 batched tokens
text-only
eager
DSpark off
native MoE off
disk-backed Engram
```

The launcher preserves caller overrides over profile/`.env` values, but it intentionally does **not** allow the correctness-critical first-boot toggles above to drift.

## Release evidence still required

Before TP4 is called deployable:

1. all-node disk preflight passes;
2. NCCL collective passes;
3. full checkpoint load completes;
4. `/v1/models` is ready;
5. deterministic `scripts/smoke_test.sh` passes;
6. per-node memory receipts prove full Engram stays non-resident;
7. no UMA guard trip;
8. repeated short decode is stable.

Only then qualify larger context, DSpark and CUDA graphs separately.
