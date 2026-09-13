# Disk-backed Engram qualification (experimental)

Disk-backed / node-local Engram is an **explicit experimental variable**, not part of the resident TP4 baseline. Keep EXL3 memory savings and Engram I/O costs measurable and separate.

Profile: [`profiles/tp4-disk-engram.env`](../profiles/tp4-disk-engram.env)  
Launcher: [`scripts/tp4_disk_engram_min_fit.sh`](../scripts/tp4_disk_engram_min_fit.sh)  
Overlay (optional apply): [`overlays/disk-engram/`](../overlays/disk-engram/)

## Why resident Engram fails on GB10 UMA

On DGX Spark / GB10, CPU and GPU share one physical unified-memory pool (~121–128 GiB MemTotal class). Resident Engram (GPU or pinned-CPU/`cpu_offload`) still materializes the large PLE embed tables into that same pool alongside the EXL3 body.

Hardware evidence summary from corrected **8K / seq1** min-fit attempts with resident Engram:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
topology: TP4+EP4, eager, text-only, DSpark=0, native MoE=0
context/batch: max_model_len=8192, max_num_seqs=1, max_num_batched_tokens=1024
observation: MemAvailable collapses toward ~0.5 GiB; host ~121 GiB used;
             SSH / node responsiveness can wedge near the UMA cliff
```

`EngramConfig.cpu_offload=True` does **not** create physical capacity on GB10 UMA: pinned host placement still consumes the shared pool. Prefer `disk_backed` for Spark qualification after recording `CAPACITY_FAIL`.

## Disk-backed gates (unit / parity / stress)

With a runtime that implements `EngramConfig.disk_backed` (see `overlays/disk-engram/`), offline qualification gates observed:

| Gate | Result | Notes |
|---|---|---|
| Synthetic lookup parity | PASS | Disk path vs resident path on synthetic rows |
| Real-slice parity | PASS_BIT_EXACT | Real checkpoint slice, bit-exact agreement |
| Engram stress / reclaim | PASS | Repeated lookups; MemAvailable stayed ~116 GiB class with bounded staging; swap unused |

These gates prove storage/lookup correctness under disk-backed mode. They are **not** a full TP4 serve throughput claim and do not replace smoke tests after a successful load.

## Runtime requirements

1. Image-pinned DeepSeek V4.1 vLLM must expose `EngramConfig.disk_backed` (stock image may not). Apply [`overlays/disk-engram/`](../overlays/disk-engram/) to the pinned site-packages or rebuild with the same pin.
2. Export `VLLM_ENGRAM_DISK_BACKED=1` into Ray workers (container start env) and pass `--engram-config {"cpu_offload":false,"disk_backed":true}` on serve (profile sets `EXTRA_VLLM_ARGS`).
3. Backing files must be **node-local NVMe** (fail closed on unsuitable filesystems in the overlay).
4. Arm the OOM guard before load:

```bash
OOM_GUARD_HOSTS="spark-a spark-b spark-c spark-d" \
  OOM_GUARD_WARN_GIB=24 OOM_GUARD_ABORT_GIB=16 \
  bash scripts/watch_oom_guard.sh start
```

5. Then:

```bash
bash scripts/tp4_disk_engram_min_fit.sh --check
bash scripts/tp4_disk_engram_min_fit.sh
```

## Mixed-K capability (separate PR)

Published TP4 packs may use tensor-granular mixed K. Current `vllm-exl3` routed allocation remains one physical K per `RoutedExperts` transformer layer unless a mixed-K / arena loader lands upstream.

**Placeholder:** mixed-K loadability depends on a separate `vllm-exl3` PR — link TBD when published (`https://github.com/vcruz305/vllm-exl3/pull/NNN`). Do not fold arena/mixed-K plugin code into this recipe branch.

## `.env` precedence note

Shell-prefix overrides must win over recipe `.env` values. That precedence bug was independently reproduced during qualification and is **already fixed** in [`scripts/lib.sh`](../scripts/lib.sh) on main. This branch does not re-implement that fix.

## What this branch is not

- Not a silent change to TP2 or resident TP4 baselines.
- Not a vendor of `vllm-exl3` arena / mixed-K plugin code (track that as a separate plugin PR).
- Not a throughput or long-context publication by itself.
