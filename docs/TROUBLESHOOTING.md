# Troubleshooting

## Start with the gates, not runtime knobs

```bash
bash scripts/doctor.sh 4
python3 scripts/probe_remote_pack.py --tp 4
bash scripts/preflight.sh 4
```

Do not debug a vLLM model load until runtime identity, checkpoint structure and distributed transport pass.

## Shell overrides look wrong

Normal recipe precedence is:

```text
explicit shell environment > .env > runtime-lock defaults
```

The disk-Engram wrapper adds its profile between caller values and `.env`:

```text
explicit caller > disk profile > .env > lock/defaults
```

Use `--check`/`DRY_RUN=1` and inspect the printed launch before counting a run as valid.

## Pointer / HTML / truncated safetensors

Those are artifact-materialization failures, not EXL3 kernel failures. Materialize the immutable HF revision with `scripts/materialize_model.sh` on a large filesystem and rerun `validate_pack.py`.

The validator also rejects index/header disagreement, invalid `data_offsets` and dtype/shape byte-count mismatches.

## Multiple physical K widths in one routed layer

This is **supported by the current pinned plugin**.

`vllm-exl3` stores exact per-expert/per-projection trellis shapes. K3–K8 may differ between experts and `w1`, `w2`, `w3` may differ inside one expert.

Expected behavior:

```text
uniform-K layer       -> fused path when available
heterogeneous-K layer -> LinearEXL3 python loop
```

The heterogeneous path is correctness-first and **not CUDA-graph-qualified**. Keep first boot eager.

If `validate_pack.py` reports an unsupported K, malformed tensor geometry or physical corruption, that is still a hard failure.

## Old `64 vs 48` trellis shape error

That was the layer-uniform allocation bug fixed by the per-expert mixed-K loader. The current loader should allocate the physical trellis width rather than pad/trim it.

If the same failure reappears, capture:

```bash
bash scripts/runtime_identity.sh
```

and confirm the plugin pin is the one in `runtime.lock.json`.

Do **not** revive the old partial-copy diagnostic as a production fix. Exact-shape trellis loading supersedes it.

## TP4 resident Engram hits the UMA cliff

This is a known result:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

Do not keep sweeping memory-utilization/context/batch settings to reproduce it. Use the guarded disk path:

```bash
bash scripts/build_disk_engram_runtime.sh
bash scripts/tp4_disk_engram_min_fit.sh --check
bash scripts/tp4_disk_engram_min_fit.sh
```

The resident launcher is regression-only:

```bash
ALLOW_RESIDENT_ENGRAM_RETEST=1 bash scripts/tp4_min_fit.sh
```

## Disk-Engram preflight fails on one node

The disk launcher checks every Ray GPU node. Common causes:

- the node was started with the baseline image instead of `deepseek-v41-exl3:disk-engram`;
- `VLLM_ENGRAM_DISK_BACKED=1` was not present when the Ray container started;
- `VLLM_ENGRAM_MODEL_DIR` differs across nodes;
- the local model/index is absent;
- backing storage is NFS/CIFS/another network filesystem;
- the node has a stale `vllm-exl3` revision.

Start every node through `scripts/start_disk_engram_cluster.sh` with the same in-container model path.

## Disk mode still starts materializing full Engram

Stop the run. The disk path is supposed to skip the large `engram.embed.weight/scale` tensors.

Verify the derived image and all-node preflight instead of manually mounting overlay files. The supported path is:

```bash
bash scripts/build_runtime.sh
bash scripts/build_disk_engram_runtime.sh
```

Then restart the Ray containers through the dedicated disk wrapper.

## OOM guard could stop the wrong workload

Current guard behavior is exact-name only. Set:

```text
OOM_GUARD_CONTAINER_NAME=<exact recipe container>
```

It no longer pattern-matches arbitrary DeepSeek containers. MemAvailable crossing the hard threshold is the stop condition; swap growth alone is evidence/warning.

## Ray sees fewer GPUs than expected

```bash
bash scripts/cluster_status.sh
```

Each Spark contributes one GPU. Verify `HEAD_IP`, per-node `NODE_IP`, host networking and Ray port. Then require the real collective:

```bash
bash scripts/cluster_collective.sh 4
```

## RDMA / RoCE issues

```text
ENABLE_RDMA=auto   # default
ENABLE_RDMA=1      # require RDMA
ENABLE_RDMA=0      # disable RDMA mapping/IB overrides
```

If cluster behavior is unclear, use `ENABLE_RDMA=0` first to separate generic Ray/NCCL issues from RoCE tuning.

## Existing cluster container blocks startup

That is intentional. The recipe does not delete a running container silently.

```bash
REPLACE_CONTAINER=1 ... bash scripts/start_cluster.sh head
```

Only use replacement when intentional.

## Worker cannot see the local model

Every node must mount the same host-side model root at `/models`, and the in-container `MODEL` / `VLLM_ENGRAM_MODEL_DIR` must identify the same snapshot path.

For disk Engram the all-node preflight verifies this before load.

## TP4 shows intermediate width 576

You are not using the intended expert-parallel geometry. TP4+EP4 keeps whole **5120 × 2304** experts and assigns **96 experts/rank**.

## TP2 owns 192 local experts

Expected. It is not an ExLlamaV3 total-expert ceiling. TP2 is now primarily a **capacity/nonresident-Engram qualification problem**, not a mixed-K representation problem.

## CUDA graphs or DSpark fail after eager base works

Return to:

```bash
DSPARK=0 EAGER=1 NATIVE_MOE=0
```

Heterogeneous mixed-K graph mode is not yet qualified. DSpark and graphs are separate qualification steps after base correctness.

## Native MoE produces wrong output

Return to the ExLlamaV3/mixed-K correctness path:

```bash
NATIVE_MOE=0 DSPARK=0 EAGER=1
```

Native p2b remains a separate K2–K4 experiment.

## Smoke test gets HTTP 200 but fails

Expected when output is wrong. `smoke_test.sh` verifies `/v1/models` and exact deterministic response content:

```text
EXL3 Spark OK
```

A fluent but incorrect response is not a pass.

## Docker image IDs differ after loading the same archive

Do not use a short image ID as the identity proof. Compare archive SHA256, complete RootFS layer list, locked source revisions and `runtime_identity.sh` receipts.
