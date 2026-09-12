# DeepSeek-V4.1-Flash EXL3 on DGX Spark

Turnkey serving recipes for **DeepSeek-V4.1-Flash EXL3** on NVIDIA DGX Spark / GB10, with separate **TP2 (2 Spark)** and **TP4 (4 Spark)** paths, plus an experimental **one-GPU `sm_120` + large-host-RAM UVA** qualification path.

> **Runtime boundary:** this repository uses the dedicated DeepSeek-V4.1 **vLLM** implementation for the V4.1 model graph. `vllm-exl3` supplies EXL3 routed-expert integration. Stock standalone ExLlamaV3 does **not** currently provide a forward-correct `DeepseekV41ForCausalLM` loader with V4.1 CED/CSA2/Engram support. See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

## Hugging Face checkpoints

| Topology | Published checkpoint | Recipe variable |
|---|---|---|
| **TP4 / 4× DGX Spark** | [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw) | `MODEL_TP4` |
| **TP2 / 2× DGX Spark** | [`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw) | `MODEL_TP2` |

Leave `MODEL=` blank in `.env` and the TP4/TP2 launchers automatically choose the matching checkpoint. Set `MODEL=` only when intentionally overriding both topology defaults with a local path or another HF repo.

The older [`vcruz305/DSV4.1-Flash-EXL3`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3) repository is a work/supersession pointer, not the preferred published runtime checkpoint.

This repository is a runtime/serving recipe. It does not contain model weights and it does not quantize the model.

> **Status:** early hardware qualification. TP4+EP4 remains the preferred correctness-first DGX Spark topology. TP2+EP2 is intentionally experimental. The ABI-3 native EXL3 MoE path remains opt-in until hardware parity/throughput receipts are complete.

## Current TP4 capacity investigation

A recent four-Spark test reproduced a near-full unified-memory cliff while using the **fat defaults** (65K context, memory util 0.75, seq4). An attempted 8K min-fit run did **not** actually test 8K because the old launcher allowed `.env` to overwrite explicit shell overrides.

That precedence bug is fixed. Explicit shell values now win:

```text
shell environment > .env > built-in defaults
```

Before another resident-Engram attempt, use the dedicated gate:

```bash
# prove the exact command without model load
bash scripts/tp4_min_fit.sh --check

# terminal 1: monitor all four Ray nodes
bash scripts/watch_cluster_memory.sh

# terminal 2: real 8K/seq1 min-fit attempt
bash scripts/tp4_min_fit.sh
```

The printed command must show **8192 / seq1 / text-only / eager / DSPARK=0 / NATIVE_MOE=0**. If that verified run still reaches the same near-full system-memory cliff during Engram loading and a worker becomes unresponsive, record:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

Then stop tuning context/batch knobs and move to a separately identified non-resident/disk-backed Engram path. See [`docs/TP4.md`](docs/TP4.md).

## Pinned Spark runtime

| Component | Pin |
|---|---|
| TP4 EXL3 checkpoint | `vcruz305/DSV4.1-Flash-EXL3-4.75bpw` |
| TP2 EXL3 checkpoint | `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw` |
| DeepSeek V4.1 vLLM image | `vllm/vllm-openai:deepseekv41-flash-0909` |
| vLLM requirement | 0.30.0+ architecture image; do **not** replace with stock pip vLLM |
| `vllm-exl3` | `8f4517e80416466fa4a3ad2eb28685021d39e95f` |
| ExLlamaV3 | `be57335b087e4f001c5caae061544df3c06ba01e` |
| Spark CUDA target | `sm_121` / `TORCH_CUDA_ARCH_LIST=12.1a` |

The base image is DeepSeek/vLLM's dedicated V4.1 image. The EXL3 plugin is installed on top of that image; the recipe never upgrades or replaces the image's vLLM package.

## Why TP + EP on Spark

DeepSeek-V4.1-Flash has 384 routed experts, hidden size 5120, expert intermediate size 2304, and top-k 6 routing.

| Recipe | Nodes | Main experts/rank | Expert shape/rank | Expected EXL3 path |
|---|---:|---:|---:|---|
| **TP4 + EP4** | 4 | **96** | **5120 × 2304** | Preferred. ExLlamaV3 expert-kernel control first; ABI-3 native p2b optional A/B. |
| **TP2 + EP2** | 2 | **192** | **5120 × 2304** | Experimental. Above the current 128-local-expert ExLlamaV3 fused envelope; ABI-3 native p2b is the intended experiment. |
| TP4 without EP | 4 | 384 | **5120 × 576** | Not recommended: 576 leaves a 64-wide tail in 128-wide EXL3 native tiles. |

With expert parallelism enabled, vLLM assigns **whole experts** to each rank. The “ExLlamaV3” wording here refers only to the EXL3 expert execution backend inside the vLLM integration.

## Required checkpoint metadata

A V4.1 EXL3 pack must preserve the original V4.1 architecture/configuration while declaring EXL3 tensors accurately. At minimum verify:

- `architectures` resolves to `DeepseekV41ForCausalLM`;
- `quantization_config.quant_method` is `exl3`;
- mixed-K metadata remains intact rather than being flattened to one base width;
- source-format non-routed weights retain the correct V4.1 block/scale metadata;
- DSpark/source tensors are not accidentally reinterpreted as main EXL3 experts;
- Engram tables remain in their declared retained representation.

Run the fail-closed metadata checker against a local checkpoint:

```bash
python scripts/check_checkpoint_compat.py /path/to/checkpoint --json
```

## DGX Spark quick start

Run the same recipe checkout and runtime image on every Spark.

### 1. Build the pinned runtime image

```bash
bash scripts/build_runtime.sh
```

### 2. Configure the model

```bash
cp .env.example .env
```

No model edit is required for the published checkpoints. With `MODEL=` blank:

- `serve_tp4.sh` / `preflight.sh 4` use `MODEL_TP4=vcruz305/DSV4.1-Flash-EXL3-4.75bpw`;
- `serve_tp2.sh` / `preflight.sh 2` use `MODEL_TP2=vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`.

For a local checkpoint, set `MODEL_DIR` on every node and set `MODEL=/models/<checkpoint-directory>`.

### 3. Start the Ray cluster

Head:

```bash
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.10 bash scripts/start_cluster.sh head
```

Each worker:

```bash
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.11 bash scripts/start_cluster.sh worker
```

For TP4 start three workers. For TP2 start one worker.

```bash
bash scripts/cluster_status.sh
```

### 4. Preflight

```bash
bash scripts/preflight.sh 4   # selects TP4 HF repo
bash scripts/preflight.sh 2   # selects TP2 HF repo
```

### 5. Serve

TP4:

```bash
bash scripts/serve_tp4.sh
```

TP2:

```bash
bash scripts/serve_tp2.sh
```

Both launchers start text-only, eager, DSpark off. TP4 defaults to the ExLlamaV3 expert-kernel control; TP2 defaults to the experimental ABI-3 native path.

### 6. Smoke test

```bash
bash scripts/smoke_test.sh
```

## Dry-run verification

Any expensive launch can be resolved without loading weights:

```bash
DRY_RUN=1 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash scripts/serve_tp4.sh
```

Always trust the **printed resolved command**, not the values you intended to pass.

## Controlled Spark A/B progression

1. TP4 resident min-fit at 8K/seq1.
2. TP4 ExLlamaV3 expert-kernel control.
3. Native ABI-3 A/B with identical checkpoint/prompts.
4. DSpark-5 in eager mode.
5. CUDA graphs after runtime kernels are warm.
6. Context: 64K → 128K → 300K only with measured headroom.
7. Vision after text serving is stable.

## TP2 is a separate quant target

TP2 is **not** just TP4 with two machines removed. The 4.75-bpw TP4 body is too tight when divided over only two Sparks. The published TP2 checkpoint is a separate SAGE target:

[`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw)

See [`docs/TP2.md`](docs/TP2.md) for the two-Spark memory/qualification boundary.

## Conservative GB10 defaults

Defaults in `.env.example` are intentionally conservative for ordinary bring-up:

- `MAX_MODEL_LEN=65536`
- `GPU_MEMORY_UTILIZATION=0.75`
- `MAX_NUM_SEQS=4`
- `MAX_NUM_BATCHED_TOKENS=4096`
- `TEXT_ONLY=1`
- `DSPARK=0`
- `EAGER=1`

The dedicated `tp4_min_fit.sh` harness overrides these to its fixed 8K/seq1 qualification geometry.

## Docker 29 cross-store image-ID trap

Moby issue [#51934](https://github.com/moby/moby/issues/51934) reports `docker load` producing different image IDs across hosts/storage integrations even when the saved image layers match. In a mixed Docker 29 fleet, do not use the short `docker images` ID alone as your identity proof.

Capture on every node:

```bash
bash scripts/image_fingerprint.sh /path/to/saved-image.tar
```

Compare archive SHA256, complete RootFS layer list, storage driver, pinned source revisions, and runtime identity.

## Runtime identity and memory monitoring

Before publishing any benchmark:

```bash
bash scripts/runtime_identity.sh
```

During capacity tests:

```bash
bash scripts/watch_cluster_memory.sh
```

DGX Spark CPU and GPU share one coherent memory pool, so system `MemAvailable` is a primary capacity signal; the watcher prints CUDA free/total alongside it for correlation.

## Engram and DGX Spark

V4.1's two Engram tables are roughly 189 GiB in the source checkpoint. Resident Engram must be treated as a measured qualification result, not an assumption. Keep any disk/node-local Engram variant separately identified so EXL3 savings and I/O costs remain measurable.

## Experimental one-GPU `sm_120` + host-RAM path

The repo also carries an experimental current-vLLM UVA path for a reported 16 GB Blackwell + very-large-host-RAM topology. This is **not** the DGX Spark TP2/TP4 path. See:

- [`docs/SM120_UVA.md`](docs/SM120_UVA.md)
- [`docs/TONOKEN3_VALIDATION.md`](docs/TONOKEN3_VALIDATION.md)
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md)

UVA keeps selected packed weights in pinned host RAM while expert math still executes on the GPU; it is not CPU-MoE compute.

## Repository layout

```text
Dockerfile.spark                    pinned V4.1 + EXL3 Spark runtime
scripts/build_runtime.sh            build the shared Spark runtime image
scripts/start_cluster.sh            Ray head/worker launcher
scripts/cluster_status.sh           cluster resource check
scripts/preflight.py                checkpoint + ABI/topology validation
scripts/preflight.sh                topology-aware preflight
scripts/serve.sh                    shared vLLM launcher
scripts/serve_tp4.sh                TP4+EP4 wrapper
scripts/serve_tp2.sh                TP2+EP2 wrapper
scripts/tp4_min_fit.sh              fixed 8K/seq1 resident-Engram gate
scripts/watch_cluster_memory.sh     all-node unified-memory monitor
scripts/image_fingerprint.sh        Docker archive/layer/store identity receipt
scripts/smoke_test.sh               OpenAI API smoke test
scripts/runtime_identity.sh         reproducibility receipt
docs/TP4.md                         four-Spark qualification path
docs/TP2.md                         two-Spark qualification path
docs/TROUBLESHOOTING.md             failure-mode playbook
```

## Upstream projects and credit

This recipe builds on DeepSeek, vLLM, Turboderp/ExLlamaV3, and `vcruz305/vllm-exl3`. See `THIRD_PARTY_NOTICES.md` for provenance and licensing boundaries.

## License

Recipe code authored in this repository is released under **AGPL-3.0-only**. Model weights, vLLM, ExLlamaV3, CUDA components and container layers keep their own licenses.
