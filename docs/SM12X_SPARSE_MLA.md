# SM12x sparse-MLA for DeepSeek V4.1 Flash EXL3 TP4

## Why

Image FlashInfer 0.6.18 has no SM120 DSV4 decode for `(num_heads=16, topk=1152)`.
After full EXL3 weight load, engine init fails in sparse MLA — not OOM/COW.

## Path (minimum)

1. Rebuild/retag runtime with FlashInfer **0.7.0rc1** at
   `07869c61ba581e6d6b8ad8d142f4a6c89b707cc1` (+ pinned CUTLASS/CCCL/SPDLOG).
2. Prewarm `sparse_mla_sm120` at image build (`MAX_JOBS=4`) — do not let four
   Ray workers JIT it at boot.
3. Mount `overlays/sm12x-sparse-mla` (64-state pages / indexer / SWA).
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

See `THIRD_PARTY_NOTICES.md` and `overlays/sm12x-sparse-mla/PROVENANCE.md`.

## Build image

```bash
docker build -f Dockerfile.fi07-sm12x -t deepseek-v41-exl3:fi07-sm12x .
```

## Launch hint

```bash
EXTRA_VLLM_ARGS='--block-size 128' \
VLLM_DSV41_IMAGE=deepseek-v41-exl3:fi07-sm12x \
  # mount overlays/sm12x-sparse-mla per mounts.txt, then serve_tp4
```
