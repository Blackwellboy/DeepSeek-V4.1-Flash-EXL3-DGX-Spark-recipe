# TP2 + EP2: two DGX Sparks

TP2 is the aggressive target. Treat it as a separate checkpoint, storage, loader, and memory problem rather than a smaller copy of TP4.

**Current SAGE source artifact:** [`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw)

> **Deployment status:** the currently published 3.30-bpw snapshot is **not yet the recommended direct vLLM TP2 deployment pack**. A lab campaign reported invalid/unmaterialized shard headers and tensor-level mixed K2-K8. Current `vllm-exl3` now accepts K2-K8 configuration values, but routed-MoE allocation still requires **one K per transformer layer**. The existing tensor-mixed snapshot must be validated and, where needed, repacked/re-encoded to the layer-uniform compatibility contract before deployment.

`serve_tp2.sh` therefore fails closed on a remote/unvalidated model by default.

## Geometry

- 2 Spark nodes / 2 GB10 GPUs total
- vLLM tensor parallel size: 2
- expert parallel: enabled
- 384 main routed experts -> **192 whole experts per rank**
- expert hidden/intermediate: **5120 x 2304** on every rank
- top-k routing: 6

Both dimensions are 128-aligned. **192 local experts is not an ExLlamaV3 fused-MoE expert-count limit.** Upstream ExLlamaV3 sizes the fused pointer tables from the actual number of experts and includes K1-K8 fused kernel instances. The historical `>128` fallback in `vllm-exl3` refers to the number of tokens assigned to a single expert in a batch, not the total local expert count.

## Recovery step 1: materialize the checkpoint correctly

Do **not** use a plain `git clone` as proof that Hugging Face model weights are present. Hugging Face Xet-backed Git repositories remain compatible with Git LFS pointer files, so a checkout can contain tiny pointer stubs instead of the safetensors payloads if the large-object transport was not materialized.

Use a filesystem with substantial free space:

```bash
bash scripts/materialize_tp2.sh /large/models/DSV4.1-Flash-SAGE-EXL3-TP2
```

The helper uses `hf download`, `huggingface-cli`, or `huggingface_hub.snapshot_download`, then runs the TP2 pack checker. It also relocates **HF Hub and Xet caches to the same large filesystem** by default so a nearly-full Spark root disk is not consumed by hidden cache/staging traffic.

The default pre-download free-space gate is intentionally conservative (`TP2_DOWNLOAD_MIN_FREE_GIB=500`). Override it only after measuring the exact snapshot plus Docker/build/cache staging requirements.

If a Spark root filesystem is short by ~60 GB, move the model snapshot **and Hugging Face/Xet caches** to a larger NVMe/shared mount. Freeing enough space to finish a download does **not** solve loader or unified-memory compatibility by itself.

## Recovery step 2: validate shards and K geometry

Run:

```bash
python3 scripts/check_tp2_pack.py \
  /large/models/DSV4.1-Flash-SAGE-EXL3-TP2 \
  --reserve-gib 32
```

The checker reads only safetensors headers and reports:

- indexed shard count and missing shards;
- Git LFS/Xet-compatible pointer stubs;
- HTML/XML downloads accidentally saved as shard files;
- truncated/invalid safetensors headers;
- EXL3 trellis K histogram inferred from `shape[-1] / 16`;
- transformer layers containing more than one K;
- materialized shard bytes;
- filesystem free space/reserve;
- `DEPLOYABLE_CURRENT_LOADER=YES|NO`.

If the reported “29 of 31 invalid shards” are pointer stubs, re-materialization may fix that blocker without re-quantizing. If they are genuinely truncated or invalid payloads after a proper HF materialization, the affected files need to be re-uploaded/repacked; the recipe cannot repair corrupted model bytes.

## Recovery step 3: make SAGE mixed K compatible with current vLLM

The current vLLM integration supports mixed precision **between transformer layers**, but one `RoutedExperts` layer currently allocates one trellis K width for all of its routed expert projections.

`vllm-exl3` pinned by this recipe now accepts config K2-K8:

- ExLlamaV3's normal fused EXL3 MoE path has K1-K8 kernel instances;
- `vllm-exl3`'s separate custom native p2b path remains K2-K4 only;
- tensor-level mixed K inside one routed layer remains unsupported.

The correctness-first TP2 backend is therefore **ExLlamaV3 (`NATIVE_MOE=0`)**, not the custom p2b experiment.

The fastest compatibility path is **not** global uniform K. Instead, coalesce the exact SAGE tensor-level recipe to one K per transformer layer while preserving K2-K8 variation across the 40 layers:

```bash
python tools/coalesce_v41_vllm_recipe.py \
  /path/to/exact-tp2-sage-recipe.yaml \
  --out recipes/recipe-tp2-vllm-layer-uniform.yaml \
  --min-k 2 \
  --max-k 8
```

That tool lives in `vcruz305/SAGE-EXL3`. It chooses layer K values nearest the original SAGE allocation and writes a compatibility summary. Because V4.1 routed layers have equal routed-expert tensor geometry, the aggregate average K can be preserved on roughly a 0.025-K grid.

**A metadata edit is not enough.** Any tensor whose selected K changes must be re-encoded. Use the SAGE encode bank to reuse exact `(tensor key, K)` variants and encode only missing deltas, then repack and rerun release/fidelity validation.

See `SAGE-EXL3/docs/V41_VLLM_TP2_COMPAT.md`.

## Memory contract: streamed/nonresident Engram

TP2 should not be qualified with resident Engram as if it were a smaller TP4 run. The SAGE TP2 direction is explicitly **streamed/nonresident Engram/PLE**.

The 4.75-bpw TP4 body is also not the normal TP2 pack. With about 233 GiB of non-PLE body, TP2 would place roughly **116.5 GiB body per Spark** before runtime/KV/workspaces, which is not a practical operating point in GB10's shared 128 GB physical memory pool.

The TP2 SAGE build exists to reduce the body substantially, but the actual corrected repack must be measured after encoding. Do not infer runtime fit solely from the nominal 3.30 bpw name.

CPU/UVA offload does not create a second physical memory pool on DGX Spark. Grace CPU and Blackwell GPU share the same LPDDR5x. It may alter access/placement behavior, but it is not a replacement for a genuinely smaller body plus nonresident Engram.

## Safe TP2 deployment sequence

Until a corrected layer-uniform vLLM pack is published, use the current HF repo only as a recovery/source artifact.

After a corrected snapshot exists:

1. materialize it on a sufficiently large mount on both Sparks;
2. run `check_tp2_pack.py` and require zero bad shards and zero mixed-K layers;
3. build the pinned recipe image with K2-K8-capable `vllm-exl3`;
4. start with `NATIVE_MOE=0`, text-only, eager, batch/seq 1, DSpark off;
5. prove 8K first;
6. then 32K;
7. then 64K;
8. attempt 128K **only with measured unified-memory headroom**.

128K is currently **unverified** and must not be advertised as working.

## Launch gate

With a validated local snapshot mounted at `/models/...`, the normal wrapper runs the checker inside the head container before vLLM launch:

```bash
MODEL_DIR=/large/models \
MODEL=/models/DSV4.1-Flash-SAGE-EXL3-TP2-vllm \
GPU_MEMORY_UTILIZATION=0.75 \
MAX_MODEL_LEN=8192 \
MAX_NUM_SEQS=1 \
NATIVE_MOE=0 \
DSPARK=0 \
EAGER=1 \
TEXT_ONLY=1 \
bash scripts/serve_tp2.sh
```

`TP2_ALLOW_UNVALIDATED=1` exists only for loader-development experiments. It is not a deployment recommendation and should never be used for benchmark claims.

## Qualification gates

Before calling TP2 usable, retain evidence for:

1. fully materialized and valid safetensors shards;
2. compatible layer-uniform K2-K8 routed layout;
3. checkpoint/repack provenance and exact plugin revision;
4. Ray sees both GPUs;
5. each rank owns 192 main experts;
6. streamed/nonresident Engram behavior is active and identified;
7. deterministic output parity against a known-good V4.1 baseline;
8. no OOM during load and first generation;
9. repeated decode stability;
10. per-rank unified-memory high-water mark;
11. actual EXL3 backend dispatch;
12. context progression with 128K still treated as unverified until measured;
13. DSpark only after the non-speculative baseline is stable.

## Performance reporting

Report both per-request decode speed and aggregate throughput. A two-node result is only comparable to TP4 when checkpoint identity, K schedule, Engram strategy, context, prompt, output length, DSpark policy, graph mode and sampling are all recorded.
