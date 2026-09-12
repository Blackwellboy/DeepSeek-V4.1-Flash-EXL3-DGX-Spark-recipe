# TP4 + EP4: four DGX Sparks

This is the preferred DeepSeek-V4.1-Flash EXL3 **runtime qualification topology**, but the currently published 4.75-bpw HF artifact must first pass the physical loader-contract gate.

**Published TP4 artifact:** [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw)

## Geometry

- 4 Spark nodes / 4 GB10 GPUs total
- vLLM tensor parallel size: 4
- expert parallel: enabled
- 384 main routed experts -> **96 whole experts per rank**
- expert hidden/intermediate: **5120 × 2304** on every rank
- top-k routing: 6

Both expert dimensions are compatible with the current EXL3 execution geometry. There is no 128-total-expert ceiling in ExLlamaV3's fused MoE path.

## Gate 0: physical checkpoint compatibility

The published TP4 model card describes **mixed K per tensor**. Current `vllm-exl3` routed allocation is still **one K per `RoutedExperts` transformer layer**. A complete HF upload is therefore not, by itself, proof that the pack is directly loadable by this vLLM integration.

Materialize the exact locked HF revision onto a large filesystem only when intentionally validating the source artifact:

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-TP4
```

Then require:

```bash
python3 scripts/validate_pack.py \
  /large/models/DSV4.1-Flash-EXL3-TP4 \
  --topology tp4 \
  --reserve-gib 32
```

and:

```text
DEPLOYABLE_CURRENT_LOADER=YES
```

The validator checks shard materialization, index/header agreement, safetensors offsets, dtype/shape byte counts, EXL3 K geometry, and whether each routed transformer layer has one physical K.

If routed layers contain more than one K, the short-term runtime fix is a **layer-uniform SAGE compatibility repack** that preserves mixed precision across layers while coalescing K inside each routed layer. Do not bypass the shape mismatch hard-fail for production inference.

`TP4_ALLOW_UNVALIDATED=1` exists only for loader-development experiments.

## Gate 1: exact runtime identity

Every Spark must run the same locked runtime:

```bash
bash scripts/doctor.sh 4
bash scripts/runtime_identity.sh
```

The canonical plugin, ExLlamaV3 revision, ABI and model revision come from `runtime.lock.json`.

## Gate 2: resident-Engram minimum fit

Once a compatible local TP4 pack exists, prove the model can fit at all with large-context and batch pressure removed.

First verify the resolved command without loading the model:

```bash
bash scripts/tp4_min_fit.sh --check
```

The launch must show:

```text
Max model len:          8192
Max num seqs:           1
Max batched tokens:     1024
Native V4.1 MoE:        0
DSpark:                 0
Eager:                  1
Text only:              1
```

Monitor all four nodes from another terminal:

```bash
bash scripts/watch_cluster_memory.sh
```

Then run:

```bash
bash scripts/tp4_min_fit.sh
```

### Capacity decision

If the true 8K/seq1 run loads and serves, resident TP4 is technically viable and larger contexts/batches can be qualified separately.

If the verified run still drives a Spark into the near-full unified-memory cliff during Engram load and a node becomes unresponsive, record:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
```

At that point stop tuning context/batch/memory-utilization and move to an explicitly identified nonresident/disk-backed Engram variant.

## Baseline after min-fit passes

Start with the correctness control:

```bash
NATIVE_MOE=0 DSPARK=0 EAGER=1 TEXT_ONLY=1 bash scripts/serve_tp4.sh
```

Required evidence before moving on:

- all four Ray GPU resources visible;
- 96 routed experts owned per main-stack rank;
- source-format non-routed weights delegated correctly;
- DSpark/source blocks are not mistaken for main EXL3 experts;
- deterministic smoke test passes exactly;
- per-rank unified-memory high-water mark captured;
- actual EXL3 backend dispatch recorded;
- no persistent dense reconstruction of the full routed expert bank.

## Native p2b A/B

Only after the ExLlamaV3 control is correct:

```bash
NATIVE_MOE=1 DSPARK=0 EAGER=1 TEXT_ONLY=1 bash scripts/serve_tp4.sh
```

The custom native path remains limited to qualified K2-K4 cases. Hold checkpoint revision, prompt, sampling, context and batch fixed when comparing.

## DSpark

After the non-speculative baseline is stable:

```bash
DSPARK=1 NATIVE_MOE=0 EAGER=1 bash scripts/serve_tp4.sh
```

Record proposed tokens, accepted tokens, mean accepted length and output speed. Do not infer DSpark value from decode speed alone.

## CUDA graphs

Only after eager mode works:

```bash
EAGER=0 bash scripts/serve_tp4.sh
```

Warm all required kernels before interpreting graph performance.

## Context progression

The safe first-boot profile is 8K. Increase only with measured memory headroom:

1. 8,192
2. 65,536
3. 131,072
4. 300,000
5. longer only after explicit qualification

V4.1 advertises much longer context, but practical Spark capacity is determined by the exact weights, Engram policy, graphs, workspaces, batch and KV behavior.

## Engram

Resident Engram is a qualification experiment, not an assumption. Keep any disk/node-local Engram variant separately identified so EXL3 memory savings and Engram I/O costs remain measurable.
