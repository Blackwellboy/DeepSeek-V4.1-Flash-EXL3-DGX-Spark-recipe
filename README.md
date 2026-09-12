# DeepSeek-V4.1-Flash EXL3 on DGX Spark

Turnkey serving recipes for **DeepSeek-V4.1-Flash EXL3** on NVIDIA DGX Spark / GB10, with separate **TP2 (2 Spark)** and **TP4 (4 Spark)** paths, plus an experimental **one-GPU `sm_120` + large-host-RAM UVA** qualification path.

> **Runtime boundary:** this repository uses the dedicated DeepSeek-V4.1 **vLLM** implementation for the V4.1 model graph. `vllm-exl3` supplies EXL3 routed-expert integration. Stock standalone ExLlamaV3 does **not** currently provide a forward-correct `DeepseekV41ForCausalLM` loader with V4.1 CED/CSA2/Engram support. See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

## Hugging Face checkpoints

| Topology | Published checkpoint | Status in this recipe |
|---|---|---|
| **TP4 / 4× DGX Spark** | [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw) | TP4 qualification checkpoint |
| **TP2 / 2× DGX Spark** | [`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw) | **Recovery/source artifact; direct vLLM launch is quarantined** until shard integrity + layer-uniform K compatibility are satisfied |

The older [`vcruz305/DSV4.1-Flash-EXL3`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3) repository is a work/supersession pointer.

This repository is a runtime/serving recipe. It does not contain model weights and it does not quantize the model.

> **Status:** early hardware qualification. TP4+EP4 remains the preferred correctness-first DGX Spark topology. TP2+EP2 is intentionally experimental. The ABI-3 native EXL3 MoE path remains opt-in until hardware parity/throughput receipts are complete.

## TP2 recovery status

A two-Spark lab campaign identified four independent blockers in the current TP2 deployment attempt:

1. most local shard files failed safetensors-header validation;
2. the published SAGE pack uses tensor-level mixed K2-K8 while the current routed vLLM allocation is one K per transformer layer;
3. one Spark was short roughly 60 GB of local storage before build/cache overhead;
4. unified-memory headroom was too tight to claim 128K context.

The recipe now fails closed instead of allowing another expensive launch through those conditions.

### What is fixed in code

- `scripts/check_tp2_pack.py` distinguishes materialized safetensors from Git LFS/Xet-compatible pointer stubs, HTML/XML, truncation, or invalid headers and reports the trellis K histogram/per-layer K set.
- `scripts/materialize_tp2.sh` materializes the HF snapshot with Hugging Face tooling onto a filesystem that passes a free-space gate; it deliberately avoids using plain Git clone as the weight download path.
- `scripts/serve_tp2.sh` blocks direct remote/unvalidated launch by default and validates a mounted local snapshot before serving.
- the pinned `vllm-exl3` revision accepts **K2-K8 configuration**; K2-K4 remain the native-qualified family, while K5-K8 can use generic fallback execution.
- SAGE now includes `tools/coalesce_v41_vllm_recipe.py`, which preserves mixed precision across V4.1 layers while choosing **one K inside each routed transformer layer** for the current vLLM allocation contract.

### What still requires a corrected pack

The existing tensor-mixed 3.30-bpw snapshot cannot become current-vLLM-compatible through metadata edits alone. If a tensor's selected K changes, its EXL3 trellis must be re-encoded. Reuse exact `(tensor key, K)` variants from the SAGE encode bank and generate only missing deltas, then repack/revalidate.

If `check_tp2_pack.py` shows the reported invalid shards are LFS/Xet pointer stubs, proper HF materialization can fix that part without re-quantizing. Hugging Face's Xet-backed Git repos intentionally retain Git-LFS-compatible pointer files for large objects. If the files remain genuinely invalid/truncated after proper materialization, the affected model shards must be re-uploaded/repacked.

See [`docs/TP2.md`](docs/TP2.md) and [`SAGE-EXL3/docs/V41_VLLM_TP2_COMPAT.md`](https://github.com/vcruz305/SAGE-EXL3/blob/main/docs/V41_VLLM_TP2_COMPAT.md).

## Current TP4 capacity investigation

A recent four-Spark test reproduced a near-full unified-memory cliff while using the **fat defaults** (65K context, memory util 0.75, seq4). An attempted 8K min-fit run did **not** actually test 8K because the old launcher allowed `.env` to overwrite explicit shell overrides.

That precedence bug is fixed. Explicit shell values now win:

```text
shell environment > .env > built-in defaults
```

Before another resident-Engram attempt, use the dedicated gate:

```bash
bash scripts/tp4_min_fit.sh --check
bash scripts/watch_cluster_memory.sh   # second terminal
bash scripts/tp4_min_fit.sh
```

If the verified 8K/seq1 run still reaches the same near-full system-memory cliff during Engram loading and a worker becomes unresponsive, record:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

Then stop tuning context/batch knobs and move to a separately identified non-resident/disk-backed Engram path. See [`docs/TP4.md`](docs/TP4.md).

## Pinned Spark runtime

| Component | Pin |
|---|---|
| TP4 EXL3 checkpoint | `vcruz305/DSV4.1-Flash-EXL3-4.75bpw` |
| TP2 SAGE source artifact | `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw` |
| DeepSeek V4.1 vLLM image | `vllm/vllm-openai:deepseekv41-flash-0909` |
| vLLM requirement | 0.30.0+ architecture image; do **not** replace with stock pip vLLM |
| `vllm-exl3` | `85358661df07efa6c3f48a4572cff92642fa08bb` |
| ExLlamaV3 | `be57335b087e4f001c5caae061544df3c06ba01e` |
| Spark CUDA target | `sm_121` / `TORCH_CUDA_ARCH_LIST=12.1a` |

The pinned plugin's CI passed with K7/K8 config coverage. Its current mixed-K contract is intentionally explicit:

```text
accepted config K: K2-K8
native-qualified K: K2-K4
routed allocation: one K per RoutedExperts transformer layer
tensor-level mixed K inside one routed layer: not yet supported
```

## Why TP + EP on Spark

DeepSeek-V4.1-Flash has 384 routed experts, hidden size 5120, expert intermediate size 2304, and top-k 6 routing.

| Recipe | Nodes | Main experts/rank | Expert shape/rank | Expected EXL3 path |
|---|---:|---:|---:|---|
| **TP4 + EP4** | 4 | **96** | **5120 × 2304** | Preferred. ExLlamaV3 expert-kernel control first; ABI-3 native p2b optional A/B. |
| **TP2 + EP2** | 2 | **192** | **5120 × 2304** | Experimental. Requires corrected layer-uniform-K TP2 pack + nonresident/streamed Engram. |
| TP4 without EP | 4 | 384 | **5120 × 576** | Not recommended: 576 leaves a 64-wide tail in 128-wide EXL3 native tiles. |

With expert parallelism enabled, vLLM assigns **whole experts** to each rank. The “ExLlamaV3” wording here refers only to the EXL3 expert execution backend inside the vLLM integration.

## DGX Spark quick start

Run the same recipe checkout and runtime image on every Spark.

### TP4

```bash
bash scripts/build_runtime.sh
cp .env.example .env
# start Ray on 4 Sparks, then:
bash scripts/preflight.sh 4
bash scripts/serve_tp4.sh
```

### TP2 recovery / preparation

Do **not** start with `serve_tp2.sh` against the remote HF repo. First select a large enough local/external model filesystem:

```bash
bash scripts/materialize_tp2.sh /large/models/DSV4.1-Flash-SAGE-EXL3-TP2
python3 scripts/check_tp2_pack.py /large/models/DSV4.1-Flash-SAGE-EXL3-TP2 --reserve-gib 32
```

If the checker reports mixed-K layers, generate/re-encode the SAGE vLLM-compatible layer-uniform pack before deployment.

After the corrected pack is mounted identically on both Sparks:

```bash
MODEL_DIR=/large/models \
MODEL=/models/DSV4.1-Flash-SAGE-EXL3-TP2-vllm \
MAX_MODEL_LEN=8192 \
MAX_NUM_SEQS=1 \
DSPARK=0 EAGER=1 TEXT_ONLY=1 \
bash scripts/serve_tp2.sh
```

Progress context only from measured headroom: **8K → 32K → 64K → 128K**. 128K is currently **unverified**.

`TP2_ALLOW_UNVALIDATED=1` is a loader-development bypass only and must not be used for deployment or benchmark claims.

## Required checkpoint metadata

A V4.1 EXL3 pack must preserve the original V4.1 architecture/configuration while declaring EXL3 tensors accurately. At minimum verify:

- `architectures` resolves to `DeepseekV41ForCausalLM`;
- `quantization_config.quant_method` is `exl3`;
- mixed-K metadata matches the physical trellis widths;
- source-format non-routed weights retain the correct V4.1 block/scale metadata;
- DSpark/source tensors are not accidentally reinterpreted as main EXL3 experts;
- Engram tables remain in their declared retained representation.

## Dry-run verification

Any expensive launch can be resolved without loading weights:

```bash
DRY_RUN=1 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash scripts/serve_tp4.sh
```

Always trust the **printed resolved command**, not the values you intended to pass.

## Conservative GB10 defaults

Defaults in `.env.example` are intentionally conservative for ordinary bring-up:

- `MAX_MODEL_LEN=65536`
- `GPU_MEMORY_UTILIZATION=0.75`
- `MAX_NUM_SEQS=4`
- `MAX_NUM_BATCHED_TOKENS=4096`
- `TEXT_ONLY=1`
- `DSPARK=0`
- `EAGER=1`

TP2 qualification should start below these defaults as described above.

## Docker 29 cross-store image-ID trap

Moby issue [#51934](https://github.com/moby/moby/issues/51934) reports `docker load` producing different image IDs across hosts/storage integrations even when the saved image layers match. In a mixed Docker 29 fleet, do not use the short `docker images` ID alone as your identity proof.

```bash
bash scripts/image_fingerprint.sh /path/to/saved-image.tar
```

Compare archive SHA256, complete RootFS layer list, storage driver, pinned source revisions, and runtime identity.

## Runtime identity and memory monitoring

```bash
bash scripts/runtime_identity.sh
bash scripts/watch_cluster_memory.sh
```

DGX Spark CPU and GPU share one coherent memory pool, so system `MemAvailable` is a primary capacity signal; CUDA free/total is shown only as a correlated view of the same physical pool.

## Engram and DGX Spark

V4.1's two Engram tables are roughly 189 GiB in the source checkpoint. Resident Engram must be treated as a measured qualification result, not an assumption. TP2's SAGE plan specifically targets **streamed/nonresident Engram**. Keep any disk/node-local Engram variant explicitly identified so EXL3 savings and I/O costs remain measurable.

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
scripts/check_tp2_pack.py           TP2 shard/K/disk fail-fast validator
scripts/materialize_tp2.sh          safe HF TP2 snapshot materializer
scripts/preflight.py                checkpoint + ABI/topology validation
scripts/preflight.sh                topology-aware preflight
scripts/serve.sh                    shared vLLM launcher
scripts/serve_tp4.sh                TP4+EP4 wrapper
scripts/serve_tp2.sh                fail-closed TP2+EP2 wrapper
scripts/tp4_min_fit.sh              fixed 8K/seq1 resident-Engram gate
scripts/watch_cluster_memory.sh     all-node unified-memory monitor
scripts/image_fingerprint.sh        Docker archive/layer/store identity receipt
scripts/smoke_test.sh               OpenAI API smoke test
scripts/runtime_identity.sh         reproducibility receipt
docs/TP4.md                         four-Spark qualification path
docs/TP2.md                         TP2 recovery + deployment contract
docs/TROUBLESHOOTING.md             failure-mode playbook
```

## Upstream projects and credit

This recipe builds on DeepSeek, vLLM, Turboderp/ExLlamaV3, and `vcruz305/vllm-exl3`. See `THIRD_PARTY_NOTICES.md` for provenance and licensing boundaries.

## License

Recipe code authored in this repository is released under **AGPL-3.0-only**. Model weights, vLLM, ExLlamaV3, CUDA components and container layers keep their own licenses.
