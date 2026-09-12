# Validation gates

The recipe is intentionally fail-closed. A successful conversion, complete Hugging Face upload, healthy Ray cluster, or fluent generation is not independently enough to call a DeepSeek-V4.1 EXL3 artifact deployable.

## Gate A: remote pack layout before download

For a public Hugging Face checkpoint, inspect the config, index and safetensors **headers only** before moving hundreds of GiB:

```bash
python3 scripts/probe_remote_pack.py --tp 4
# or
python3 scripts/probe_remote_pack.py --tp 2
```

For another repository/revision:

```bash
python3 scripts/probe_remote_pack.py \
  --tp 4 \
  --repo owner/model \
  --revision <immutable-revision>
```

The probe uses HTTP Range requests for every safetensors shard. It downloads the first eight bytes and then exactly the declared header range. If the server returns a normal `200` response instead of `206 Partial Content`, the probe aborts **before reading the shard body** so it cannot silently become a full model download.

It checks:

- locked/canonical config and index metadata;
- shard count;
- index ↔ shard-header tensor membership;
- safetensors `data_offsets` against the remote object size;
- known dtype × shape byte counts;
- physical EXL3 K inferred from trellis geometry;
- K2-K8 runtime capability;
- multiple physical K widths inside one routed transformer layer.

A compatible result prints:

```text
REMOTE_LAYOUT_COMPATIBLE=YES
```

This is a **layout** gate, not proof that all payload bytes are intact. Local materialization and validation remain required.

## Gate B: materialized checkpoint integrity

After downloading:

```bash
python3 scripts/validate_pack.py /models/checkpoint --topology tp4 --reserve-gib 32
```

or:

```bash
python3 scripts/validate_pack.py /models/checkpoint --topology tp2 --reserve-gib 32
```

Require:

```text
DEPLOYABLE_CURRENT_LOADER=YES
```

The local validator adds pointer/HTML/truncation detection, actual file-length validation, per-shard header hashes and filesystem reserve checks.

## Gate C: runtime identity

Before cluster qualification:

```bash
bash scripts/runtime_identity.sh
```

Every node must agree with `runtime.lock.json` on the plugin revision, ExLlamaV3 revision and native ABI. The model revision must also be immutable for a comparable qualification run.

## Gate D: Ray + real NCCL collective

`ray status` only proves that resources registered. It does not prove that GPU collectives work over the selected network path.

After the head and workers are up, run:

```bash
bash scripts/cluster_collective.sh 4
# or
bash scripts/cluster_collective.sh 2
```

The test pins one Ray GPU task per selected node, creates a temporary NCCL process group and performs repeated 16 MiB all-reduces. It verifies the numerical result and reports per-rank latency and payload rate.

Normal:

```bash
bash scripts/preflight.sh 4
```

or:

```bash
bash scripts/preflight.sh 2
```

now runs this collective automatically **after** static/runtime preflight. To bypass only for targeted debugging:

```bash
SKIP_NCCL_COLLECTIVE=1 bash scripts/preflight.sh 4
```

A bypassed collective is not a fully qualified distributed deployment.

Useful tuning variables:

```text
COLLECTIVE_MEGABYTES=16
COLLECTIVE_WARMUP=2
COLLECTIVE_ITERATIONS=5
COLLECTIVE_MASTER_PORT=29557
```

The payload GB/s reported by this helper is intentionally a simple tensor-bytes / wall-time rate, not an algorithm-normalized link-bandwidth claim. Use it for health/regression comparisons, not marketing bandwidth numbers.

## Gate E: deterministic model smoke

After model load:

```bash
bash scripts/smoke_test.sh
```

The script requires the served model to appear in `/v1/models` and requires the deterministic response content to equal exactly:

```text
EXL3 Spark OK
```

A request returning HTTP 200 or fluent-looking text does not count as a pass.

## Release-state vocabulary

Use these states distinctly:

```text
SAGE_ENCODED
PACK_STRUCTURALLY_VALID
VLLM_LOADER_COMPATIBLE
TP_TOPOLOGY_LOAD_PASSED
OUTPUT_PARITY_PASSED
CONTEXT_8K_PASSED
RELEASE_READY
```

Uploading model files to Hugging Face may happen before the final state, but the model card and recipe should clearly say which state has actually been demonstrated.
