# DeepSeek-V4.1-Flash EXL3 on DGX Spark

Turnkey serving recipes for **DeepSeek-V4.1-Flash EXL3** on NVIDIA DGX Spark / GB10, with separate **TP2 (2 Spark)** and **TP4 (4 Spark)** paths, plus an experimental **one-GPU `sm_120` + large-host-RAM UVA** qualification path.

> **Runtime boundary:** this repository uses the dedicated DeepSeek-V4.1 **vLLM** implementation for the V4.1 model graph. `vllm-exl3` supplies EXL3 routed-expert integration. Stock standalone ExLlamaV3 does **not** currently provide a forward-correct `DeepseekV41ForCausalLM` loader with V4.1 CED/CSA2/Engram support. See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

## Hugging Face model

**Default compiled checkpoint:** [vcruz305/DSV4.1-Flash-EXL3-4.75bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw)

The recipe uses `vcruz305/DSV4.1-Flash-EXL3-4.75bpw` as the default `MODEL` in `.env.example`. The older `vcruz305/DSV4.1-Flash-EXL3` repository is a work/supersession pointer, not the compiled 32-shard checkpoint.

This repository is a runtime/serving recipe. It does not contain model weights and it does not quantize the model. Override `MODEL` when testing a topology-specific or local EXL3 checkpoint.

> **Status:** early hardware qualification. TP4+EP4 remains the preferred correctness-first DGX Spark topology. TP2+EP2 is intentionally experimental. The one-GPU `sm_120` path is also experimental and uses **current vLLM selective UVA offload**, not the older Spark container and not CPU-MoE compute.

For the Lna-Lab/TonoKen3 small-GPU report, start with [`docs/SM120_UVA.md`](docs/SM120_UVA.md) and then follow [`docs/TONOKEN3_VALIDATION.md`](docs/TONOKEN3_VALIDATION.md).

## Pinned Spark runtime

| Component | Pin |
|---|---|
| Default EXL3 checkpoint | `vcruz305/DSV4.1-Flash-EXL3-4.75bpw` |
| DeepSeek V4.1 vLLM image | `vllm/vllm-openai:deepseekv41-flash-0909` |
| vLLM requirement | 0.30.0+ architecture image; do **not** replace with stock pip vLLM |
| `vllm-exl3` | `8f4517e80416466fa4a3ad2eb28685021d39e95f` |
| ExLlamaV3 | `be57335b087e4f001c5caae061544df3c06ba01e` |
| Spark CUDA target | `sm_121` / `TORCH_CUDA_ARCH_LIST=12.1a` |

The base image is DeepSeek/vLLM's dedicated V4.1 image. The EXL3 plugin is installed on top of that image; the recipe never upgrades or replaces the image's vLLM package.

The CUDA architecture is configurable at build time. For a CUDA 12.8 environment whose toolchain reports an `sm_120` GPU, compile extensions with the corresponding target, normally:

```bash
TORCH_CUDA_ARCH_LIST=12.0 ./scripts/build_runtime.sh
```

That only selects extension compilation targets. It is **not** a claim that every vLLM/FlashInfer/kernel path is qualified on that GPU.

## Experimental one-GPU `sm_120` + host-RAM path

Current upstream vLLM adds a shorter route for the reported 16 GB Blackwell + 1 TiB RAM topology:

- V4.1 Engram can live in pinned host memory and be accessed through UVA;
- selected model parameters can be put in pinned host memory through vLLM's `uva` offloader;
- `vllm-exl3` now has an opt-in guard that requires the six large packed EXL3 expert payloads to be vLLM UVA-mapped before its EXL3 handles/pointer tables are accepted.

The expert math still runs on the **GPU**. This is zero-copy host-memory execution over UVA/PCIe, not CPU-MoE.

The design reference is current vLLM commit `988d9b6777d077f843cd2164a222ac8535d36ed2` and `vllm-exl3` commit `1cb234b754566f1581f700a31ea5415879586a76` or newer. The older Spark image should **not** be assumed to contain those newer offload interfaces.

In a prepared current-vLLM environment:

```bash
python scripts/preflight_sm120_uva.py
./scripts/serve_sm120_uva.sh
```

The conservative launcher starts text-only, TP1, batch 1, eager, 8K context, DSpark off, Engram CPU/UVA on, and selectively offloads:

```text
w13_trellis  w13_suh  w13_svh
w2_trellis   w2_suh   w2_svh
```

See [`docs/SM120_UVA.md`](docs/SM120_UVA.md) before running it. The model may require roughly **400–500 GiB-class pinned/mapped host state**, so a 1 TiB machine has raw capacity but still needs the driver/OS to accept that page-lock pressure.

## Why TP + EP on Spark

DeepSeek-V4.1-Flash has 384 routed experts, hidden size 5120, expert intermediate size 2304, and top-k 6 routing.

| Recipe | Nodes | Main experts/rank | Expert shape/rank | Expected EXL3 path |
|---|---:|---:|---:|---|
| **TP4 + EP4** | 4 | **96** | **5120 x 2304** | Preferred. ExLlamaV3 expert-kernel control first; ABI-3 native p2b optional A/B. |
| **TP2 + EP2** | 2 | **192** | **5120 x 2304** | Experimental. Above the current 128-local-expert ExLlamaV3 fused envelope; use fallback for correctness or ABI-3 native p2b for an explicit experiment. |
| TP4 without EP | 4 | 384 | **5120 x 576** | Not recommended: 576 leaves a 64-wide tail in 128-wide EXL3 native tiles. |

With expert parallelism enabled, vLLM assigns **whole experts** to each rank. That is what makes TP4 especially attractive for EXL3 on Spark.

The “ExLlamaV3” wording in this table refers only to the EXL3 expert execution backend inside the vLLM integration. It does **not** mean standalone ExLlamaV3 owns or can currently instantiate the V4.1 model graph.

## Required checkpoint metadata

A V4.1 EXL3 pack must preserve the original V4.1 architecture/configuration while declaring the EXL3 tensors accurately. Do **not** infer the codebook or per-tensor K from the model name or average bpw.

At minimum, verify:

- `architectures` resolves to `DeepseekV41ForCausalLM`;
- `quantization_config.quant_method` is `exl3`;
- mixed-K metadata remains intact rather than being flattened to one base width;
- source-format non-routed weights retain the correct V4.1 block/scale metadata;
- DSpark/source tensors are not accidentally reinterpreted as main EXL3 experts;
- Engram tables remain in their declared retained representation;
- the actual codebook is inspected from checkpoint metadata/tensor suffixes before attempting ExLlamaV3 CPU-MoE.

Run the fail-closed metadata checker against a local checkpoint:

```bash
python scripts/check_checkpoint_compat.py /path/to/DSV4.1-Flash-EXL3-4.75bpw
```

A `CONDITIONAL` CPU-MoE result is only a metadata preflight. It is not an end-to-end runtime pass.

`vllm-exl3` exposes the delegated source block shape through the outer EXL3 config where V4.1 inspects global quantization metadata before individual dense layers select their source delegate.

## DGX Spark quick start

Run the same recipe checkout and runtime image on every Spark.

### 1. Build the pinned runtime image

```bash
./scripts/build_runtime.sh
```

The image layers the pinned EXL3 plugin and ExLlamaV3 expert-kernel dependency onto the dedicated DeepSeek V4.1 image and verifies that the native extension reports ABI 3.

### 2. Configure the model

```bash
cp .env.example .env
# MODEL already defaults to vcruz305/DSV4.1-Flash-EXL3-4.75bpw.
```

To test a different Hugging Face checkpoint, change `MODEL`. If `MODEL` is local, set `MODEL_DIR` to a host directory mounted at `/models` on every Spark and set `MODEL=/models/<checkpoint-directory>`. Every node must see identical checkpoint contents.

### 3. Start the Ray cluster

Choose one Spark as the head. All nodes use host networking.

Head:

```bash
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.10 ./scripts/start_cluster.sh head
```

Each worker:

```bash
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.11 ./scripts/start_cluster.sh worker
```

For TP4, start three workers. For TP2, start one worker.

Check the head:

```bash
./scripts/cluster_status.sh
```

### 4. Preflight the checkpoint and plugin

```bash
./scripts/preflight.sh 4   # TP4
./scripts/preflight.sh 2   # TP2
```

Preflight validates the V4.1 EXL3 metadata, prints the actual `vllm-exl3` diagnostics/ABI, and shows the expected expert layout before the expensive model load.

For a standalone metadata/CPU-offload compatibility inspection of a downloaded pack:

```bash
python scripts/check_checkpoint_compat.py /path/to/checkpoint --json
```

### 5. Serve

Preferred four-Spark control:

```bash
./scripts/serve_tp4.sh
```

Experimental two-Spark recipe:

```bash
./scripts/serve_tp2.sh
```

Both launchers default to **text-only, eager mode, DSpark off** so the first variable being tested is the model/EXL3 integration itself.

### 6. Smoke test

```bash
./scripts/smoke_test.sh
```

## Controlled Spark A/B progression

Do not turn everything on at once. Use this order:

1. **TP4+EP4 / ExLlamaV3 expert-kernel control**: `NATIVE_MOE=0 DSPARK=0 EAGER=1`.
2. **Native ABI-3 A/B**: same checkpoint/prompts, set `NATIVE_MOE=1`.
3. **DSpark-5**: set `DSPARK=1`; this recipe intentionally starts with adaptive verification disabled on Spark.
4. **CUDA graphs**: set `EAGER=0` only after all runtime/JIT kernels have warmed successfully.
5. **Context**: 64K -> 128K -> 300K -> longer only after measuring unified-memory headroom.
6. **Vision**: set `TEXT_ONLY=0` after text serving is stable.

Example native A/B on TP4:

```bash
NATIVE_MOE=1 ./scripts/serve_tp4.sh
```

The opt-in sets both `VLLM_EXL3_V41_NATIVE_MOE=1` and the EXL3 backend preference to native. An old extension cannot accidentally run the new geometry: V4.1 native eligibility requires `P2B_MOE_ABI_VERSION >= 3`.

## CPU-compute offload boundary

`vllm-exl3` still has **no host-CPU expert compute backend**. The new UVA route is different: it keeps packed weights in host RAM but executes the expert kernels on the GPU.

Current upstream ExLlamaV3 has an experimental CPU-compute MoE path, but it requires an architecture ExLlamaV3 can instantiate and currently expects `mul1`, K <= 8, plus uniform per-expert bias presence. Since upstream does not yet provide a forward-correct V4.1 architecture, that route remains blocked at the model-graph gate even if a particular pack's EXL3 metadata is otherwise eligible.

For a small-GPU host, test the current-vLLM UVA route first. If it is too PCIe-bound or cannot pin enough memory, continue the standalone ExLlamaV3 V4.1 port and CPU-MoE route.

See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md), [`docs/SM120_UVA.md`](docs/SM120_UVA.md), and [`docs/TONOKEN3_VALIDATION.md`](docs/TONOKEN3_VALIDATION.md).

## TP2 is a separate target

TP2 is **not just TP4 with two machines removed**. It has 192 whole experts per rank and only 256 GB of aggregate unified memory before runtime/KV/graph overhead. Use a checkpoint specifically sized for TP2 and expect Engram placement to be the dominant memory constraint.

The TP2 launcher defaults to the ABI-3 native candidate because the current ExLlamaV3 fused MoE path has a 128-local-expert ceiling. To force a conservative fallback test:

```bash
NATIVE_MOE=0 ./scripts/serve_tp2.sh
```

Do not report TP2 performance until output parity, per-rank memory, actual backend dispatch, and long-running stability are captured.

## Conservative GB10 defaults

The dedicated V4.1 runtime provides the `deepseek_v41` tokenizer mode, tool parser, reasoning parser, vision architecture and DSpark implementation. This recipe adds the EXL3 storage/execution layer; it does not fork those features.

Defaults here are intentionally conservative:

- `MAX_MODEL_LEN=65536`
- `GPU_MEMORY_UTILIZATION=0.75`
- `MAX_NUM_SEQS=4`
- `MAX_NUM_BATCHED_TOKENS=4096`
- `TEXT_ONLY=1`
- `DSPARK=0`
- `EAGER=1`

Override them in `.env` or per command only after the baseline works.

## Runtime identity

Capture this before publishing any benchmark:

```bash
./scripts/runtime_identity.sh
```

Keep the image digest, plugin commit, ExLlamaV3 revision, CUDA/Torch/vLLM/FlashInfer versions, native ABI, Ray topology, model revision, backend selection, context settings and Engram placement with every result.

## Engram and DGX Spark

V4.1's two Engram tables are roughly 189 GiB in the source checkpoint. EXL3 reduces the routed-expert footprint, which may make resident Engram practical on TP4, but that must be measured rather than assumed. TP2 is substantially tighter.

This baseline does not silently patch Engram to disk or claim that the dedicated upstream image is already fully Spark-qualified. If a given pack cannot hold resident Engram, record that failure and use a clearly identified disk/node-local Engram patch as a separate variant rather than mixing it into the EXL3 baseline.

## Repository layout

```text
Dockerfile.spark                    pinned V4.1 + EXL3 Spark runtime
scripts/build_runtime.sh            build the shared Spark runtime image
scripts/check_checkpoint_compat.py  fail-closed standalone metadata/CPU-MoE preflight
scripts/preflight_sm120_uva.py      current-vLLM sm_120/UVA capability preflight
scripts/serve_sm120_uva.sh          conservative one-GPU host-RAM/UVA launcher
scripts/start_cluster.sh            Ray head/worker container launcher
scripts/cluster_status.sh           cluster resource check
scripts/preflight.py                Spark checkpoint + ABI/topology validation
scripts/preflight.sh                run Spark preflight inside the head container
scripts/serve.sh                    shared Spark vLLM launcher
scripts/serve_tp4.sh                TP4+EP4 wrapper
scripts/serve_tp2.sh                TP2+EP2 wrapper
scripts/smoke_test.sh               OpenAI API smoke test
scripts/runtime_identity.sh         reproducibility receipt
configs/                            pack metadata examples
docs/COMPATIBILITY.md               runtime/format/topology boundaries
docs/SM120_UVA.md                   one-GPU Blackwell + host-RAM qualification path
docs/TONOKEN3_VALIDATION.md         Lna-Lab validation protocol
docs/TP4.md                         four-Spark qualification path
docs/TP2.md                         two-Spark qualification path
```

## Upstream projects and credit

This recipe builds on DeepSeek, vLLM, Turboderp/ExLlamaV3, and `vcruz305/vllm-exl3`. See `THIRD_PARTY_NOTICES.md` for provenance and licensing boundaries.

## License

Recipe code authored in this repository is released under **AGPL-3.0-only**. Model weights, vLLM, ExLlamaV3, CUDA components and container layers keep their own licenses.
