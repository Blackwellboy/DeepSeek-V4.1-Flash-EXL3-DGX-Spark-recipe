# Troubleshooting

## Start here: run the doctor

Before changing runtime knobs:

```bash
bash scripts/doctor.sh 4
# or
bash scripts/doctor.sh 2
```

The recipe now treats runtime identity, disk/cache placement and physical checkpoint layout as first-class gates. Do not debug a vLLM load until those pass.

## `runtime.lock.json` and my local settings disagree

`runtime.lock.json` is authoritative for the supported recipe path. `build_runtime.sh` fails closed when a component override differs from the lock.

For an intentional experiment only:

```bash
ALLOW_RUNTIME_OVERRIDE=1 \
VLLM_EXL3_REF=<sha> \
bash scripts/build_runtime.sh
```

Record the resulting runtime identity before comparing results.

## My shell overrides were ignored

Current precedence is:

```text
explicit shell environment > .env > runtime.lock.json defaults
```

Prove the exact launch without loading weights:

```bash
DRY_RUN=1 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash scripts/serve_tp4.sh
```

Do not count a run as an 8K/seq1 test unless the printed command actually says 8192 and seq1.

## The TP4/TP2 remote model is blocked before serving

Expected when the locked artifact status is not `deployable`.

Current source/qualification artifacts may be complete on Hugging Face while still violating the vLLM routed-loader contract. The recipe therefore refuses direct remote launch by default.

Materialize the artifact deliberately, then validate it:

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 4 /large/models/tp4

python3 scripts/validate_pack.py /large/models/tp4 --topology tp4 --reserve-gib 32
```

For TP2 use topology `tp2`.

Never use `TP4_ALLOW_UNVALIDATED=1` or `TP2_ALLOW_UNVALIDATED=1` for benchmark or production claims. Those are loader-development bypasses only.

## `validate_pack.py` reports pointer/HTML/truncated shards

The local bytes are not a complete safetensors snapshot.

- `pointer`: Git LFS/Xet object was not materialized.
- `html`: an error/login/proxy page was saved as a model file.
- `truncated`: file/header ended before the declared safetensors structure.

Use `scripts/materialize_model.sh` on a sufficiently large filesystem. Do not requantize until you know the HF objects themselves are genuinely bad.

## `validate_pack.py` reports index/header disagreement

The validator now fails when:

- an index tensor is missing from its declared shard;
- a shard contains a tensor absent from the index;
- a tensor's `data_offsets` exceed the file payload;
- known dtype/shape byte counts disagree with the declared byte range.

Treat those as artifact integrity/repack problems, not loader problems.

## `validate_pack.py` reports multiple physical K widths in one routed layer

This is a **pack/runtime-format mismatch**.

Current `vllm-exl3` accepts K2-K8, but each vLLM `RoutedExperts` transformer layer still allocates one K width. Tensor-granular SAGE packs can therefore be valid EXL3 artifacts yet incompatible with this serving layout.

The short-term fix is a layer-uniform compatibility repack that preserves mixed K across transformer layers while coalescing K inside each routed layer. Changing K requires real re-encoding; metadata edits are not sufficient.

Do not solve this by enabling diagnostic shape-mismatch loading for normal inference.

## Shape mismatch hard failure

Default behavior remains a hard failure. This is intentional.

The merged plugin supports:

```text
VLLM_EXL3_ALLOW_SHAPE_MISMATCH=1
```

only as an explicit diagnostic mode. It zero-fills the destination and coordinate-copies the per-dimension overlap while preserving rank checks. Output is knowingly invalid and must not be benchmarked or presented as model inference.

## TP4 resident model drives a Spark to near-full system memory

First ensure the physical pack validation passed. Then run the narrow min-fit gate:

```bash
# terminal 1
bash scripts/watch_cluster_memory.sh

# terminal 2
bash scripts/tp4_min_fit.sh --check
bash scripts/tp4_min_fit.sh
```

If the verified 8K/seq1/text-only/eager/no-DSpark/no-native run still reaches the same near-full unified-memory cliff during Engram loading and a worker becomes unresponsive, record:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

Then stop changing context/batch/memory-utilization knobs and move to a separately identified nonresident/disk-backed Engram variant.

## Docker image IDs differ between Sparks after loading the same archive

Do not immediately assume the image payloads differ. Moby issue [#51934](https://github.com/moby/moby/issues/51934) documents `docker load` producing different image IDs across machines/storage integrations while RootFS layers can remain identical.

Capture:

```bash
bash scripts/image_fingerprint.sh /path/to/saved-image.tar
```

Compare:

1. archive SHA256;
2. complete `RootFS.Layers` list;
3. locked plugin/ExLlama revisions;
4. `runtime_identity.sh` output;
5. Docker storage driver/image-store mode.

A different short `docker images` ID alone is not sufficient evidence of different runtime contents.

## `DeepseekV41ForCausalLM` is unknown

Use the dedicated image from the runtime lock:

```text
vllm/vllm-openai:deepseekv41-flash-0909
```

Do not fix this with a stock `pip install -U vllm`; that can remove the V4.1 architecture/runtime this recipe depends on.

## `--quantization exl3` is unknown

Run:

```bash
bash scripts/runtime_identity.sh
```

The image must contain `/opt/vllm-exl3` at the locked commit and `vllm_exl3.runtime_diagnostics()` must succeed after registration.

## Native ABI is stale

The container has an old `vllm_exl3_c` extension. Rebuild the locked runtime; do not copy new Python plugin files over an old `.so`.

Expected:

```text
P2B_MOE_ABI_VERSION >= 3
```

## Ray sees fewer GPUs/nodes than expected

On the head:

```bash
bash scripts/cluster_status.sh
```

Each DGX Spark contributes one GPU. Verify `HEAD_IP`, `NODE_IP`, host networking and that every worker joined the same Ray port.

## The launcher says the cluster container already exists

This is now fail-closed by design. `start_cluster.sh` no longer destroys an existing container implicitly.

If replacement is intentional:

```bash
REPLACE_CONTAINER=1 HEAD_IP=... NODE_IP=... bash scripts/start_cluster.sh head
```

## RDMA / RoCE startup problems

Modes:

```text
ENABLE_RDMA=auto   # default; use RDMA device only when present
ENABLE_RDMA=1      # require RDMA device; missing device is a hard error
ENABLE_RDMA=0      # no RDMA mapping or IB/RoCE overrides
```

In `auto`, Spark-specific HCA/interface defaults are only applied when those host interfaces exist. Explicit environment overrides still win.

If transport is unstable, first run with `ENABLE_RDMA=0` to separate generic cluster issues from RoCE tuning. Do not change networking and model/kernel settings at the same time.

## Worker cannot find the local model

Every node must mount the same host-side `MODEL_DIR` at `/models`, and `MODEL` must point inside it, for example:

```bash
MODEL_DIR=/data/models
MODEL=/models/DeepSeek-V4.1-Flash-EXL3
```

Use the validator's per-shard header hashes and the locked model revision as lightweight cross-node identity evidence.

## TP4 shows a 576 intermediate width

You are not actually using expert parallelism. Intended TP4+EP4 keeps whole 5120 × 2304 experts and assigns 96 experts per rank.

The normal launcher includes `--enable-expert-parallel`.

## TP2 owns 192 local experts

That is expected and is **not** an ExLlamaV3 total-expert-count failure. The correctness-first backend for TP2 remains ExLlamaV3 (`NATIVE_MOE=0`). The custom native p2b path is a separate K2-K4 experiment.

## DSpark hangs or graph capture becomes unstable

Return to:

```bash
DSPARK=0 EAGER=1
```

Then add DSpark in eager mode only after the non-speculative baseline is correct.

## Native path produces bad output

Return immediately to:

```bash
NATIVE_MOE=0 DSPARK=0 EAGER=1
```

Capture checkpoint revision, prompt, runtime diagnostics and failing output. Do not interpret speed until output parity is established.

## Smoke test prints a failure despite a 200 response

That is intentional. `smoke_test.sh` now verifies both `/v1/models` and exact deterministic response content. A fluent but incorrect model is not considered a pass.
