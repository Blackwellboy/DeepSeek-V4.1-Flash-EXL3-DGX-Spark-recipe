# Troubleshooting

## My shell overrides were ignored

Older revisions sourced `.env` after the caller environment, so a command like:

```bash
MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash scripts/serve_tp4.sh
```

could silently fall back to values stored in `.env` such as 65K/seq4.

Current `scripts/lib.sh` uses conventional precedence:

```text
explicit shell environment > .env > built-in defaults
```

Before an expensive load, prove the resolved command with:

```bash
DRY_RUN=1 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=1 bash scripts/serve_tp4.sh
```

For the fixed resident-Engram gate, prefer:

```bash
bash scripts/tp4_min_fit.sh --check
```

Do not count a run as an 8K/seq1 test unless the printed command actually says 8192 and seq1.

## TP4 resident model drives a Spark to ~full system memory

Run the narrow min-fit gate rather than another broad tuning sweep:

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

Then stop changing context, sequence count, and memory-utilization knobs. Move to an explicitly named non-resident/disk-backed Engram implementation instead.

## Docker image IDs differ between Sparks after loading the same archive

Do not immediately assume the image payloads differ. Moby issue [#51934](https://github.com/moby/moby/issues/51934) documents `docker load` producing different image IDs across machines/storage integrations while the RootFS layer list is the same. The report involves the containerd image-store integration versus legacy storage behavior and is relevant to Docker 29-era fleets.

For cross-node identity, capture:

```bash
bash scripts/image_fingerprint.sh /path/to/saved-image.tar
```

Compare, in order:

1. saved archive SHA256;
2. complete `RootFS.Layers` list;
3. pinned `vllm-exl3` and ExLlamaV3 revisions;
4. `runtime_identity.sh` output;
5. Docker storage driver / image-store mode.

A different short `docker images` ID by itself is not sufficient evidence that two nodes loaded different runtime contents.

## `DeepseekV41ForCausalLM` is unknown / architecture import fails

You are almost certainly not running the dedicated V4.1 image. This recipe requires:

```text
vllm/vllm-openai:deepseekv41-flash-0909
```

Do not fix this by `pip install -U vllm`; that can remove the exact architecture/runtime the recipe depends on.

## `--quantization exl3` is unknown

Confirm the recipe image, plugin entry point and exact checkout:

```bash
bash scripts/runtime_identity.sh
```

The image must contain `/opt/vllm-exl3` at the pinned commit. `vllm_exl3.runtime_diagnostics()` should work after registration.

## Native ABI is 1 or 2

The container has a stale `vllm_exl3_c` extension. Rebuild `Dockerfile.spark`; do not copy new Python plugin files over an old `.so`.

Expected:

```text
P2B_MOE_ABI_VERSION >= 3
```

## Ray sees fewer GPUs/nodes than expected

On the head:

```bash
bash scripts/cluster_status.sh
```

Each DGX Spark should contribute one GPU. Verify `HEAD_IP`, `NODE_IP`, host networking and that every worker joined the same Ray port.

The recipe intentionally requires `NODE_IP` instead of guessing it because Spark hosts often have multiple interfaces.

## Worker cannot find the local model

For local checkpoints, every node must mount the same host-side `MODEL_DIR` at `/models`, and `MODEL` must be a path inside that mount, for example:

```bash
MODEL_DIR=/data/models
MODEL=/models/DeepSeek-V4.1-Flash-EXL3
```

If one worker has different checkpoint files, stop before benchmarking.

## TP4 shows a 576 intermediate width

That means you are not actually using expert parallelism. The intended topology is TP4 + EP4, where each rank owns whole 5120 x 2304 experts.

The launch scripts always include `--enable-expert-parallel`.

## TP2 falls back instead of using ExLlamaV3 fused MoE

Expected. TP2+EP2 has 192 local main experts, above the current ExLlamaV3 fused 128-expert envelope. The TP2 recipe defaults to the experimental ABI-3 native p2b candidate.

## Model loads but OOMs around Engram

V4.1 Engram is a major memory component. First record the exact failure/high-water mark. A disk/node-local Engram variant should be introduced explicitly and documented separately rather than silently mixed into the baseline.

For TP4, run the dedicated min-fit decision gate above before declaring resident Engram impossible.

## DSpark hangs or graph capture becomes unstable

Return to:

```bash
DSPARK=0 EAGER=1
```

Then add DSpark in eager mode. This recipe starts DSpark with adaptive verification disabled to avoid making variable verification shapes part of the first Spark baseline.

## Native path produces bad output

Immediately return to the control:

```bash
NATIVE_MOE=0 DSPARK=0 EAGER=1
```

Capture the exact model revision, prompt, plugin diagnostics and failing output. Do not interpret speed until numerical/output parity is established.

## NCCL / distributed communication errors

First confirm all nodes can reach each other on the selected host interface and that Ray node IPs are correct. Avoid changing NCCL tuning and EXL3/kernel settings at the same time. If an interface override is required, apply the same `NCCL_SOCKET_IFNAME`/related environment on every node and record it with the benchmark receipt.
