# SM12x sparse-MLA for DeepSeek V4.1 Flash EXL3 TP4

## Why

Image FlashInfer 0.6.18 has no SM120 DSV4 decode for `(num_heads=16, topk=1152)`.
After full EXL3 weight load, engine init fails in sparse MLA — not OOM/COW.

## Path (minimum)

1. Rebuild/retag runtime with FlashInfer **0.7.0rc1** at
   `07869c61ba581e6d6b8ad8d142f4a6c89b707cc1` (+ pinned CUTLASS/CCCL/SPDLOG).
2. Prewarm `sparse_mla_sm120` at image build (`MAX_JOBS=4`) — the image build
   must fail if prewarm fails; never defer four-worker JIT to engine startup.
3. Use the pinned SM12x V4.1 architecture patch source documented by the recipe.
4. Launch with `--block-size 128` (global token block). Physical sparse pages
   remain 64 states.

## Offline gates

```text
FLASHINFER_DSV4_16x1152=PASS
COMMON_BLOCK_SIZE_SELECTION=PASS
SPARSE_MLA_PHYSICAL_PAGE_CONTRACT=PASS
INDEXER_STATES_PER_PAGE=64
DEEPGEMM_BLOCK_KV=64
SPARSE_MLA_PREBUILT=YES
RUNTIME_JIT_REQUIRED=NO
```

## Attribution

See `THIRD_PARTY_NOTICES.md` and the pinned runtime-patch provenance referenced by
this recipe.

## Build image

```bash
docker build -f Dockerfile.fi07-sm12x -t deepseek-v41-exl3:fi07-sm12x .
```

## Launch hint

`scripts/lib.sh` / `scripts/start_cluster.sh` consume `IMAGE`, not
`VLLM_DSV41_IMAGE`.

```bash
IMAGE=deepseek-v41-exl3:fi07-sm12x \
EXTRA_VLLM_ARGS='--block-size 128' \
  bash scripts/serve_tp4.sh
```

For the proven 4x GB10 path, keep the separately qualified disk-Engram profile,
`gpu_memory_utilization=0.70`, 16 GiB abort guard, and the repaired/validated
4.75bpw model snapshot until Victor republishes a complete pack.
