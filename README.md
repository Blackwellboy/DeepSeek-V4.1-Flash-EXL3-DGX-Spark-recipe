# DeepSeek-V4.1-Flash EXL3 on DGX Spark

Serving and qualification tooling for **DeepSeek-V4.1-Flash EXL3** on NVIDIA DGX Spark / GB10 with separate TP4 and TP2 paths.

> **Runtime boundary:** vLLM owns the DeepSeek-V4.1 model graph (CED/CSA2, Engram, vision, DSpark and parsers). `vllm-exl3` supplies EXL3 routed-expert storage/execution. ExLlamaV3 is an EXL3 kernel dependency in this recipe; it is not the model-graph owner.

## Current release status

| Topology | Hugging Face artifact | Current vLLM recipe status |
|---|---|---|
| **TP4 / 4× Spark** | [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw) | **Source/qualification artifact pending physical layer-K validation.** The published pack is SAGE tensor-granular mixed-K; current `vllm-exl3` allocates one K per routed transformer layer. |
| **TP2 / 2× Spark** | [`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw) | **Quarantined source artifact.** Requires valid materialized shards plus a layer-uniform routed-K compatibility repack. |

The recipe intentionally fails closed instead of spending hours downloading/building/loading a checkpoint whose physical layout is incompatible with the pinned loader.

The older [`vcruz305/DSV4.1-Flash-EXL3`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3) repository is a supersession pointer.

## One source of truth: `runtime.lock.json`

`runtime.lock.json` is authoritative for:

- dedicated V4.1 vLLM image;
- exact `vllm-exl3` repository and commit;
- exact ExLlamaV3 commit;
- minimum native ABI;
- CUDA architecture target;
- accepted EXL3 K ranges;
- routed allocation contract;
- canonical HF model IDs/revisions;
- safe first-boot settings.

Current plugin pin:

```text
vllm-exl3: ee8c2c171bbe0d036a3accb24a76af5a95506748
```

That history includes the GB10 setuptools>=77 build fix and opt-in coordinate-preserving diagnostic shape-mismatch mode from PR #9 by @fattchris. Shape mismatch still **hard-fails by default**. The current pin also resolves TP/EP weight geometry from `RoutedExperts.moe_config.moe_parallel_config` before any process-wide TP fallback, so EP layouts with MoE TP=1 keep whole expert matrices.

The locked runtime reports:

```text
accepted EXL3 config K:        K2-K8
ExLlamaV3 MoE kernel family:   K1-K8
custom native p2b:             K2-K4 only
routed allocation:             one K per RoutedExperts transformer layer
tensor-level mixed K in layer: not currently supported
TP/EP geometry source:          per-MoE config before process TP
```

## Architecture

DeepSeek-V4.1 has 384 routed experts, hidden size 5120, expert intermediate size 2304 and top-k 6.

| Layout | Local experts | Expert shape/rank | First correctness backend |
|---|---:|---:|---|
| TP4 + EP4 | 96 | 5120 × 2304 | ExLlamaV3 |
| TP2 + EP2 | 192 | 5120 × 2304 | ExLlamaV3 |
| TP4 without EP | 384 | 5120 × 576 | Not recommended |

**192 local experts is not an ExLlamaV3 total-expert ceiling.** The historical `>128` fallback in `vllm-exl3` is about tokens assigned to one expert in a batch, not total experts owned by a rank.

## Validation-first quick start

Do not begin with model loading. Qualify the host and artifact first.

### 1. Clone and create config

```bash
git clone https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe
cd DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe
cp .env.example .env
```

`.env.example` is the safe **8K / seq1 / text-only / eager / DSpark-off / native-off** first-boot profile. Larger contexts are explicit qualification steps, not defaults.

### 2. Host doctor

Run on every Spark:

```bash
bash scripts/doctor.sh 4
# or
bash scripts/doctor.sh 2
```

The doctor checks Docker, NVIDIA visibility, host architecture, filesystem headroom, RDMA discovery, local image state and—when a local model is supplied—the physical checkpoint contract.

### 3. Probe the remote HF layout without downloading weights

Before moving hundreds of GiB, inspect only `config.json`, the index, and safetensors header byte ranges:

```bash
python3 scripts/probe_remote_pack.py --tp 4
# or
python3 scripts/probe_remote_pack.py --tp 2
```

The probe requires HTTP `206 Partial Content` for shard reads and aborts if a server ignores `Range`, so it cannot silently become a full-shard download. It checks remote shard/index membership, offset bounds, known dtype/shape byte counts, EXL3 K geometry and within-layer mixed K.

A compatible layout reports:

```text
REMOTE_LAYOUT_COMPATIBLE=YES
```

This is a layout gate only; local integrity validation remains required.

### 4. Build the locked runtime

```bash
bash scripts/build_runtime.sh
```

Build inputs come from `runtime.lock.json`. Experimental overrides require:

```bash
ALLOW_RUNTIME_OVERRIDE=1 ... bash scripts/build_runtime.sh
```

The Dockerfile includes GB10 fixes contributed and hardware-tested by @fattchris: conditional `python` alias creation, dynamic cuSPARSE header discovery, configurable `VLLM_EXL3_REPO`, and the merged plugin fixes.

### 5. Materialize a model onto a large filesystem

TP4:

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-TP4
```

TP2:

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 2 /large/models/DSV4.1-Flash-EXL3-TP2
```

The generic materializer keeps `HF_HOME`, Hub cache and Xet cache on the same large filesystem, pins the HF revision when one is locked, then runs the physical pack validator.

Current TP4/TP2 artifacts are source/qualification artifacts rather than declared vLLM-deployable packs, so downloading them for compatibility work is deliberately explicit.

### 6. Validate a local pack

```bash
python3 scripts/validate_pack.py \
  /large/models/DSV4.1-Flash-EXL3-TP4 \
  --topology tp4 \
  --reserve-gib 32
```

or:

```bash
python3 scripts/validate_pack.py \
  /large/models/DSV4.1-Flash-EXL3-TP2 \
  --topology tp2 \
  --reserve-gib 32
```

The validator checks without loading weights:

- all indexed shards exist and are real safetensors, not LFS/Xet pointers or HTML;
- index entries exist in their declared shard;
- no unexpected shard tensor is missing from the index;
- `data_offsets` are in bounds;
- known dtype/shape byte counts equal the declared tensor byte range;
- physical EXL3 K inferred from trellis geometry;
- K values are within the locked runtime capability;
- routed transformer layers obey the loader's one-K-per-layer contract;
- model/config source-quantization metadata matches V4.1 expectations;
- filesystem reserve remains available;
- per-shard safetensors-header hashes are emitted for cheap cross-node comparison.

Require:

```text
DEPLOYABLE_CURRENT_LOADER=YES
```

before treating a pack as deployment-ready.

### 7. Start Ray

`ENABLE_RDMA=auto` is the default. It maps `/dev/infiniband` only when present. `ENABLE_RDMA=1` requires it and hard-fails if missing; `ENABLE_RDMA=0` leaves network selection to NCCL/Gloo.

Head:

```bash
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.10 bash scripts/start_cluster.sh head
```

Worker:

```bash
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.11 bash scripts/start_cluster.sh worker
```

An existing container is **not deleted automatically**. Intentional replacement requires `REPLACE_CONTAINER=1`.

TP4 needs three workers; TP2 needs one.

### 8. Preflight runtime identity **and** a real GPU collective

```bash
bash scripts/preflight.sh 4
# or
bash scripts/preflight.sh 2
```

Preflight verifies the locked plugin/ExLlama revisions and ABI on every live Ray node. For a mounted local checkpoint it also runs the physical pack validator. It then pins one Ray GPU task to each selected node and performs a real NCCL all-reduce, verifying both numerical parity and cross-node transport before any model load.

The collective can be run independently:

```bash
bash scripts/cluster_collective.sh 4
```

`SKIP_NCCL_COLLECTIVE=1` exists for targeted debugging only; a skipped collective is not a fully qualified distributed deployment.

### 9. First serve

TP4:

```bash
bash scripts/serve_tp4.sh
```

TP2:

```bash
bash scripts/serve_tp2.sh
```

Both wrappers fail closed on remote/unvalidated source artifacts unless an explicit loader-development bypass is set.

### 10. Deterministic smoke test

```bash
bash scripts/smoke_test.sh
```

The smoke test fails unless `/v1/models` exposes the expected served model **and** the deterministic response content is exactly:

```text
EXL3 Spark OK
```

Pretty-printing a fluent but wrong response no longer counts as a pass.

## TP4 qualification

The published 4.75-bpw pack is complete on HF, but its card explicitly describes **mixed K per tensor**. Current vLLM routed allocation remains one K per transformer layer. Therefore TP4 must pass the remote/local layout gates before runtime qualification.

Once a compatible TP4 pack exists, use:

```bash
bash scripts/tp4_min_fit.sh --check
bash scripts/watch_cluster_memory.sh   # second terminal
bash scripts/tp4_min_fit.sh
```

If the verified 8K/seq1 load still drives a Spark into the near-full unified-memory cliff during resident Engram load, record:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

and move to an explicitly identified nonresident/disk-backed Engram variant rather than another memory-knob sweep.

See [`docs/TP4.md`](docs/TP4.md).

## TP2 qualification

TP2 is a separate checkpoint and memory target. The 4.75-bpw TP4 body is not a practical two-Spark body. The current TP2 SAGE source artifact also uses tensor-granular K2-K8 and must be repacked to one K per routed layer for the current vLLM loader.

The intended short path is:

```text
original SAGE tensor-granular allocation
        ↓
coalesce one K per transformer layer
        ↓
reuse exact (tensor,K) encodes from the encode bank
        ↓
encode only missing deltas
        ↓
repack + fidelity/release checks
```

TP2 should use streamed/nonresident Engram and target actual final body bytes rather than trusting nominal bpw. 128K remains unverified.

See [`docs/TP2.md`](docs/TP2.md).

## Explicit profiles

Qualification settings live in `profiles/`:

- `profiles/tp4-minfit.env`
- `profiles/tp4-64k.env`
- `profiles/tp2-minfit.env`
- `profiles/tp2-32k.env`

The safe first-boot profile is also the default in `.env.example` and `runtime.lock.json`.

## Reproducibility receipts

```bash
bash scripts/runtime_identity.sh
bash scripts/image_fingerprint.sh /path/to/saved-image.tar
bash scripts/watch_cluster_memory.sh
```

Do not use the short Docker image ID alone to prove cross-node identity on Docker 29. Compare archive SHA256, complete RootFS layer list, locked source revisions and runtime identity.

## Useful docs

- [`docs/VALIDATION.md`](docs/VALIDATION.md)
- [`docs/TP4.md`](docs/TP4.md)
- [`docs/TP2.md`](docs/TP2.md)
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md)
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

## License

Recipe code authored in this repository is released under **AGPL-3.0-only**. Model weights, vLLM, ExLlamaV3, CUDA components and container layers retain their own licenses.
