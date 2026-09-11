# Troubleshooting

## `DeepseekV41ForCausalLM` is unknown / architecture import fails

You are almost certainly not running the dedicated V4.1 image. This recipe requires:

```text
vllm/vllm-openai:deepseekv41-flash-0909
```

Do not fix this by `pip install -U vllm`; that can remove the exact architecture/runtime the recipe depends on.

## `--quantization exl3` is unknown

Confirm the recipe image, plugin entry point and exact checkout:

```bash
./scripts/runtime_identity.sh
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
./scripts/cluster_status.sh
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
