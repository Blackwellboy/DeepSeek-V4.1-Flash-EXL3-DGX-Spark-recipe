# Experimental sm_120: one GPU + large host RAM

This is the fastest current path to test `vcruz305/DSV4.1-Flash-EXL3-4.75bpw` on a machine such as Lna-Lab's RTX PRO 2000 Blackwell (`sm_120`) host with 1 TiB DDR5.

It is **not** the DGX Spark TP4 recipe and it is **not** CPU-MoE compute.

The execution model is:

```text
DeepSeek-V4.1 graph / CED / attention       current vLLM on the GPU
EXL3 routed-expert packed parameters        pinned host RAM via vLLM UVA
EXL3 expert math                             existing GPU kernels over UVA pointers
Engram FP8 tables                            pinned host RAM via vLLM Engram UVA
KV/cache/workspaces                          GPU as allowed by the 16 GB budget
```

The routed-expert math still runs on the GPU. The GPU reads packed expert weights across the host interconnect through Unified Virtual Addressing (UVA). This avoids waiting for a standalone ExLlamaV3 V4.1 graph port before testing the host-memory topology.

## Why this path exists now

Current vLLM has two relevant upstream capabilities:

1. `EngramConfig.cpu_offload` supports `DeepseekV41ForCausalLM` and stores its Engram shard in pinned host memory for UVA lookup.
2. the generic model-weight offloader supports selective parameter placement through `--offload-backend uva`, `--cpu-offload-gb`, and `--cpu-offload-params`.

The first public-source revision used to design this experiment is:

```text
vllm-project/vllm
988d9b6777d077f843cd2164a222ac8535d36ed2
```

The EXL3 placement guard was added to:

```text
vcruz305/vllm-exl3
1cb234b754566f1581f700a31ea5415879586a76 or newer
```

Do not replace those references with an older V4.1 container and assume the same features exist. Run the preflight against the exact environment you plan to serve from.

## Important distinction: CPU compute vs UVA

TonoKen3's existing source harness computes routed experts on CPU. This experiment does not reproduce that compute backend.

UVA instead keeps the packed parameters in pinned host memory and presents mapped accelerator views to CUDA. That is useful because `vllm-exl3` constructs its EXL3 handles and pointer tables after placement, so the qualification path can point them at the mapped host allocation.

This is still experimental. Correct placement does **not** prove:

- output parity;
- acceptable PCIe bandwidth behavior;
- CUDA graph compatibility;
- acceptable page-lock/pinned-memory limits;
- that every native/ExLlamaV3 fallback kernel is efficient on mapped host pointers.

## Expected host-memory pressure

The public 4.75-bpw repository is hundreds of GiB. The two retained Engram shards alone are roughly 190+ GiB of tensor payload, while the EXL3 body is roughly another quarter-terabyte class payload.

A one-GPU UVA test can therefore ask the host to keep roughly **400–500 GiB class pinned/mapped state** once expert payload, Engram and runtime overhead are considered. A 1 TiB host has capacity, but the CUDA driver, operating system, NUMA policy and `mlock`/pinned-memory behavior still need to accept it.

Do not treat `free -h` alone as proof that the allocation will work.

## Build/runtime target

For `sm_120` with a CUDA 12.8-class toolchain, compile CUDA extensions with the target corresponding to the installed PyTorch toolchain, normally:

```bash
export TORCH_CUDA_ARCH_LIST=12.0
```

Verify rather than assume:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name())
print("capability", torch.cuda.get_device_capability())
print("compiled archs", torch.cuda.get_arch_list())
PY
```

## Runtime preflight

From this recipe checkout, inside the exact Python environment intended to serve the model:

```bash
python scripts/preflight_sm120_uva.py | tee sm120-uva-preflight.json
```

The preflight requires:

- CUDA device capability `(12, 0)`;
- vLLM reporting UVA available;
- current `EngramConfig.cpu_offload` support;
- current selective `UVAOffloadConfig.cpu_offload_params` support;
- importable V4.1 NVIDIA model implementation;
- `vllm-exl3` with its EXL3 UVA placement guard installed.

A pass is only a **capability preflight**. It does not load the 453 GiB checkpoint.

## First serving attempt

The provided launcher is intentionally conservative:

```bash
./scripts/serve_sm120_uva.sh
```

Useful overrides:

```bash
MODEL=/models/DSV4.1-Flash-EXL3-4.75bpw \
CPU_OFFLOAD_GB=300 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.80 \
./scripts/serve_sm120_uva.sh
```

The launcher uses:

```text
--offload-backend uva
--cpu-offload-gb 300
--cpu-offload-params w13_trellis w13_suh w13_svh w2_trellis w2_suh w2_svh
--engram-config {"cpu_offload": true}
```

and sets:

```text
VLLM_EXL3_REQUIRE_UVA_EXPERTS=1
```

That guard turns these conditions into hard failures instead of silent fallbacks:

- the six large EXL3 routed-expert payloads were not offloaded;
- only some of them fit inside the requested offload budget;
- vLLM fell back to ordinary CPU tensors rather than mapped UVA accelerator views.

The tiny EXL3 codebook-marker tensors are intentionally not part of the mandatory host placement set.

## Qualification progression

Use this order. Do not enable DSpark or long context during the first boot.

1. **Capability preflight** — save the JSON receipt.
2. **Checkpoint metadata preflight** — run `scripts/check_checkpoint_compat.py` against the local checkpoint.
3. **Model load only** — text-only, eager, batch 1, 8K context.
4. **Short greedy parity** — fixed prompts against Lna-Lab's bit-exact source harness.
5. **Backend receipt** — capture `vllm_exl3.runtime_diagnostics()`, server logs and the `_exl3_uva_expert_status` placement evidence.
6. **Memory receipt** — GPU VRAM, host RSS/pinned memory, NUMA placement and storage activity.
7. **Base decode throughput** — no speculative decoding.
8. **DSpark** — only after deterministic base parity.
9. **48K prefill** — compare against the existing source receipt.
10. **128K then 512K context** — only from measured cache/headroom data.

A first failure is valuable. Send the **first** traceback plus the preflight JSON instead of patching around it invisibly.

## First metrics to compare

| Metric | Source harness | EXL3 UVA path |
|---|---:|---:|
| Greedy token agreement | reference | |
| PPL / NLL | reference | |
| Base decode tok/s | ~9 reported | |
| DSpark code tok/s | ~18.9 reported | |
| DSpark ja tok/s | ~12.4 reported | |
| 48K prefill ms/token | ~17 reported | |
| GPU VRAM | | |
| Host pinned/RSS | | |
| PCIe read bandwidth | | |
| Engram lookup placement | source mmap | pinned UVA |

Do not compare throughput until correctness and actual placement are proven.

## If UVA is too slow or cannot pin enough memory

Then the next path is the standalone ExLlamaV3 V4.1 forward port plus ExLlamaV3's existing CPU-compute MoE machinery, provided the exact checkpoint satisfies that backend's current format constraints (`mul1`, K <= 8, uniform expert-bias presence).

SAGE-EXL3 now contains a fail-closed V4.1 architecture contract and per-layer CED ownership plan to support that port. It intentionally does not claim that the forward implementation is finished.
