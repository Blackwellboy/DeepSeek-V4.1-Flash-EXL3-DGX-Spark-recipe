# DeepSeek-V4.1-Flash EXL3 on DGX Spark

Turnkey serving and qualification recipes for **DeepSeek-V4.1-Flash EXL3** on NVIDIA DGX Spark / GB10, with separate **TP4 (4 Spark)** and **TP2 (2 Spark)** paths.

> **Runtime boundary:** vLLM owns the DeepSeek-V4.1 model graph (CED/CSA2, Engram, vision, DSpark, parsers). `vllm-exl3` supplies EXL3 routed-expert storage/execution. Standalone ExLlamaV3 is an expert-kernel dependency here, not the V4.1 model loader. See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

## Hugging Face checkpoints

| Topology | Checkpoint | Status |
|---|---|---|
| **TP4 / 4× DGX Spark** | [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw) | TP4 qualification checkpoint |
| **TP2 / 2× DGX Spark** | [`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw) | **Recovery/source artifact; direct vLLM deployment is quarantined** pending shard validation + layer-uniform-K repack |

The older [`vcruz305/DSV4.1-Flash-EXL3`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3) repo is a work/supersession pointer.

## Pinned Spark runtime

| Component | Pin |
|---|---|
| DeepSeek V4.1 image | `vllm/vllm-openai:deepseekv41-flash-0909` |
| `vllm-exl3` | `21fa627a3933d80de2d1030e732354d8c3cd761e` |
| ExLlamaV3 | `be57335b087e4f001c5caae061544df3c06ba01e` |
| Spark CUDA target | `sm_121` / `TORCH_CUDA_ARCH_LIST=12.1a` |

The recipe never upgrades the base image to stock pip vLLM. The pinned plugin is CI-green and reports:

```text
accepted EXL3 config K:       K2-K8
ExLlamaV3 fused MoE kernels:  K1-K8
custom native p2b:            K2-K4 only
routed allocation:            one K per transformer RoutedExperts layer
tensor-level mixed K in layer:not yet supported
```

## Corrected TP2 architecture facts

DeepSeek-V4.1 has 384 routed experts, hidden size 5120, intermediate size 2304, and top-k 6.

| Layout | Local experts | Local expert shape | Correctness-first backend |
|---|---:|---:|---|
| **TP4 + EP4** | 96 | 5120 × 2304 | ExLlamaV3 |
| **TP2 + EP2** | 192 | 5120 × 2304 | ExLlamaV3 |
| TP4 without EP | 384 | 5120 × 576 | Not recommended; 576 leaves a 128-tile tail |

**192 local experts is not an ExLlamaV3 fused-MoE ceiling.** Upstream ExLlamaV3 sizes the pointer tables from the actual expert count and has K1-K8 fused instances. The historical `>128` fallback in `vllm-exl3` refers to **tokens assigned to one expert in a batch**, not the total number of local experts.

The separate `vllm-exl3` native p2b backend remains an optional K2-K4 experiment. It is not the first-boot TP2 backend.

---

# TP2 recovery: current blockers and fixes

A two-Spark lab campaign reported:

- 29 of 31 local shard headers failing safetensors validation;
- tensor-level mixed K2-K8 incompatible with the current one-K-per-layer routed allocation;
- one Spark about 59.9 GB short on local disk before build/cache overhead;
- very tight unified-memory headroom;
- 128K context unverified.

The recipe now **fails closed** instead of attempting an expensive model load through those blockers.

## 1. Materialize the TP2 snapshot correctly

A plain Git checkout is not proof that the Hugging Face large objects are materialized. Use the supplied downloader on a large NVMe/shared filesystem:

```bash
bash scripts/materialize_tp2.sh /large/models/DSV4.1-Flash-SAGE-EXL3-TP2
```

The helper:

- uses `hf download`, `huggingface-cli`, or `huggingface_hub.snapshot_download`;
- defaults to a conservative 500 GiB free-space gate;
- puts `HF_HOME`, Hub cache, and **Xet cache on the same large filesystem** as the model;
- validates the downloaded snapshot before declaring it usable.

This directly prevents the reported “Spark 2 is 59.9 GB short” condition from simply moving into hidden Hugging Face/Xet cache traffic on the root disk.

## 2. Classify the 29/31 shard failures before rebuilding weights

```bash
python3 scripts/check_tp2_pack.py \
  /large/models/DSV4.1-Flash-SAGE-EXL3-TP2 \
  --reserve-gib 32
```

The checker reports:

- missing indexed shards;
- Git-LFS/Xet-compatible pointer stubs;
- HTML/XML accidentally saved as model files;
- truncated or invalid safetensors headers;
- physical trellis K inferred from `shape[-1] / 16`;
- K histogram;
- layers containing more than one K;
- materialized model bytes;
- free filesystem capacity;
- `DEPLOYABLE_CURRENT_LOADER=YES|NO`.

If the 29 bad files are pointer stubs, proper Hugging Face materialization can fix that blocker **without requantizing**. If they remain genuinely truncated/invalid after proper materialization, those specific shards need to be rebuilt/re-uploaded.

## 3. Repack SAGE mixed K for current vLLM

The current published 3.30-bpw artifact is tensor-granular. Current `vllm-exl3` can execute K2-K8, but one `RoutedExperts` transformer layer still allocates one K width.

The short path is **not uniform K3** and does not require throwing away SAGE's dynamic allocation. Instead, preserve mixed K **across the 40 transformer layers** while coalescing all routed tensors within each layer to one K:

```bash
# in vcruz305/SAGE-EXL3
python tools/coalesce_v41_vllm_recipe.py \
  /path/to/exact-tensor-level-tp2-recipe.yaml \
  --out recipes/recipe-tp2-vllm-layer-uniform.yaml \
  --min-k 2 \
  --max-k 8
```

The optimizer minimizes deviation from the original tensor-level SAGE K allocation while preserving the aggregate average on roughly a **0.025-K grid** across 40 equal-geometry layers.

Changing K requires a real re-encode. Use the existing SAGE encode bank to reuse exact `(tensor key, K)` variants and encode only missing deltas, then repack and run release/fidelity checks.

See [`SAGE-EXL3/docs/V41_VLLM_TP2_COMPAT.md`](https://github.com/vcruz305/SAGE-EXL3/blob/main/docs/V41_VLLM_TP2_COMPAT.md).

## 4. TP2 memory contract

TP2 must use a **streamed/nonresident Engram/PLE strategy**. Do not treat it as a resident-Engram downsizing of TP4.

The 4.75-bpw TP4 body is roughly 233 GiB excluding the ~189 GiB Engram/PLE component. Splitting that body over only two Sparks would be about 116.5 GiB/Spark before runtime/KV/workspaces, which is not a practical GB10 operating point.

The SAGE TP2 build exists to reduce the body, but the corrected layer-uniform repack must be measured after encoding. Nominal “3.30 bpw” alone is not a memory-fit proof.

`--cpu-offload-gb` is not a capacity escape hatch on DGX Spark because CPU and GPU share the same physical LPDDR5x pool.

## 5. First corrected TP2 boot

After the corrected local snapshot passes `check_tp2_pack.py`:

```bash
MODEL_DIR=/large/models \
MODEL=/models/DSV4.1-Flash-SAGE-EXL3-TP2-vllm \
MAX_MODEL_LEN=8192 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=1024 \
GPU_MEMORY_UTILIZATION=0.75 \
NATIVE_MOE=0 \
DSPARK=0 \
EAGER=1 \
TEXT_ONLY=1 \
bash scripts/serve_tp2.sh
```

Qualification order:

```text
8K -> 32K -> 64K -> 128K only if measured memory headroom remains
```

**128K is currently unverified.** Do not publish it as supported until a real run proves it.

`TP2_ALLOW_UNVALIDATED=1` exists only for loader-development experiments. Do not use it for deployment or benchmark claims.

See [`docs/TP2.md`](docs/TP2.md).

---

# TP4 qualification

TP4 continues to use the 4.75-bpw checkpoint.

A previous attempted “8K” capacity test was invalid because older recipe code let `.env` overwrite explicit shell overrides. Precedence is now:

```text
explicit shell environment > .env > built-in defaults
```

For the resident-Engram minimum-fit gate:

```bash
# resolve command only
bash scripts/tp4_min_fit.sh --check

# second terminal
bash scripts/watch_cluster_memory.sh

# actual run
bash scripts/tp4_min_fit.sh
```

The printed launch must show 8192 / seq1 / text-only / eager / DSpark off / native off.

If that verified run still reaches the same near-full unified-memory cliff during Engram loading and a worker becomes unresponsive, record:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

Then stop memory-knob tuning and move to an explicitly identified nonresident/disk-backed Engram variant.

See [`docs/TP4.md`](docs/TP4.md).

---

# Quick start

## Build the Spark runtime

```bash
bash scripts/build_runtime.sh
cp .env.example .env
```

## Start Ray

Head:

```bash
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.10 bash scripts/start_cluster.sh head
```

Worker:

```bash
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.11 bash scripts/start_cluster.sh worker
```

TP4 needs three workers. TP2 needs one worker.

```bash
bash scripts/cluster_status.sh
```

## TP4

```bash
bash scripts/preflight.sh 4
bash scripts/serve_tp4.sh
```

## TP2

The remote 3.30-bpw source artifact is intentionally blocked by `serve_tp2.sh` until a validated layer-uniform repack exists. Use the recovery workflow above.

## Smoke test

```bash
bash scripts/smoke_test.sh
```

## Dry-run command resolution

```bash
DRY_RUN=1 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash scripts/serve_tp4.sh
```

Always trust the **printed resolved command**, not the values you intended to pass.

---

# Docker 29 image-ID trap

Moby issue [#51934](https://github.com/moby/moby/issues/51934) reports `docker load` yielding different image IDs across machines/storage integrations despite matching image content/layers.

For cross-node identity, use:

```bash
bash scripts/image_fingerprint.sh /path/to/saved-image.tar
```

Compare archive SHA256, complete RootFS layer list, storage driver, source revisions, and runtime identity. Do not reject a node solely because the short `docker images` ID differs.

---

# Diagnostics

```bash
bash scripts/runtime_identity.sh
bash scripts/watch_cluster_memory.sh
```

DGX Spark CPU and GPU share one coherent memory pool, so system `MemAvailable` is a primary capacity signal. CUDA free/total is shown for correlation, not as an independent memory pool.

Useful docs:

- [`docs/TP4.md`](docs/TP4.md)
- [`docs/TP2.md`](docs/TP2.md)
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md)
- [`docs/SM120_UVA.md`](docs/SM120_UVA.md)
- [`docs/TONOKEN3_VALIDATION.md`](docs/TONOKEN3_VALIDATION.md)

## License

Recipe code authored in this repository is released under **AGPL-3.0-only**. Model weights, vLLM, ExLlamaV3, CUDA components, and container layers retain their own licenses.
