# TP4 + EP4: four DGX Sparks

This is the preferred DeepSeek-V4.1-Flash EXL3 runtime qualification topology.

**Published artifact:** `vcruz305/DSV4.1-Flash-EXL3-4.75bpw`

## Geometry

- 4 Spark nodes / 4 GB10 GPUs total
- tensor parallel size: 4
- expert parallel: enabled
- 384 routed experts -> **96 whole experts per rank**
- expert matrix: **5120 × 2304**
- top-k: 6

There is no 128-total-expert ExLlamaV3 ceiling.

## Gate 0: physical checkpoint compatibility

The published checkpoint uses heterogeneous K3–K8 **inside routed layers and even across `w1/w2/w3` of one expert**. The pinned `vllm-exl3` now stores exact per-expert/per-projection trellis shapes, so this layout no longer requires a layer-uniform repack.

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-TP4

python3 scripts/validate_pack.py \
  /large/models/DSV4.1-Flash-EXL3-TP4 \
  --topology tp4 \
  --reserve-gib 32
```

Require:

```text
DEPLOYABLE_CURRENT_LOADER=YES
```

That proves the **pack/loader contract**, not full hardware deployment.

The heterogeneous path is correctness-first `LinearEXL3` execution and is not CUDA-graph-qualified yet. First boot must stay eager.

## Gate 1: runtime identity and transport

Every Spark must use the same locked runtime and model revision.

```bash
bash scripts/doctor.sh 4
bash scripts/runtime_identity.sh
bash scripts/preflight.sh 4
bash scripts/cluster_collective.sh 4
```

The collective must pass before model load.

## Gate 2: resident Engram is already a known capacity failure

Corrected resident min-fit testing at **8K / seq1 / eager / text-only / DSpark-off / native-off** drove the GB10 unified-memory pool into the near-full cliff:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
~121 GiB used class / ~0.5 GiB MemAvailable class during Engram materialization
```

Do not use resident Engram as the normal qualification gate anymore. `scripts/tp4_min_fit.sh` is retained only for an explicit regression reproduction:

```bash
ALLOW_RESIDENT_ENGRAM_RETEST=1 bash scripts/tp4_min_fit.sh
```

On Spark, pinned-host/UVA Engram does not create physical capacity because CPU and GPU share the same memory pool.

## Gate 3: build and start the disk-Engram runtime

Build the baseline and explicit derivative:

```bash
bash scripts/build_runtime.sh
bash scripts/build_disk_engram_runtime.sh
```

Start every node with the same materialized checkpoint path. The dedicated cluster wrapper propagates disk-Engram and mixed-K prescan environment into the Ray processes before vLLM launches.

Example head:

```bash
IMAGE=deepseek-v41-exl3:disk-engram \
MODEL_DIR=/large/models \
MODEL=/models/DSV4.1-Flash-EXL3-TP4 \
VLLM_ENGRAM_MODEL_DIR=/models/DSV4.1-Flash-EXL3-TP4 \
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.10 \
  bash scripts/start_disk_engram_cluster.sh head
```

Use `worker` on the other three Sparks.

## Gate 4: all-node disk preflight and first load

Arm the OOM guard on every node. The guard is bound to one exact configured container name and uses WARN=24 GiB / ABORT=16 GiB by default.

Then:

```bash
bash scripts/tp4_disk_engram_min_fit.sh --check
bash scripts/tp4_disk_engram_min_fit.sh
```

Before load, the launcher probes all four Ray GPU nodes and requires:

- `VLLM_ENGRAM_DISK_BACKED=1`;
- the same local model/index path;
- disk-Engram overlay import;
- weight-loader Engram skip active;
- mixed-K-capable `vllm-exl3`;
- non-network backing storage.

The first load is locked to:

```text
context:              8192
max seqs:             1
max batched tokens:   1024
text only:            yes
DSpark:               off
native MoE:           off
eager:                on
Engram:               node-local disk-backed
```

## Gate 5: serving correctness

A successful load is not enough. Require:

```bash
bash scripts/smoke_test.sh
```

and exact response:

```text
EXL3 Spark OK
```

Also capture:

- `/v1/models` ready;
- 96 owned experts/rank;
- actual mixed-K dispatch;
- per-rank unified-memory high-water mark;
- full Engram non-residency evidence;
- no dense reconstruction of the routed expert bank.

## After the baseline passes

Only then qualify independently:

1. larger context (32K/64K/128K from measured headroom);
2. DSpark;
3. uniform-K fused/native A/B where eligible;
4. CUDA graphs **only after a mixed-K graph-safe path is implemented/qualified**.

Do not infer 128K viability from the disk-Engram parity tests alone.

See [`DISK_ENGRAM.md`](DISK_ENGRAM.md) for the storage path and overlay details.
