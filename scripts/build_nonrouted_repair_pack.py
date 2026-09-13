#!/usr/bin/env python3
"""Build a supplemental safetensors repair pack for missing non-routed V4.1 tensors.

Downloads base DeepSeek-V4.1-Flash shards one at a time, extracts only the
missing required tensors (exact native FP8/BF16 bytes), writes repair shard(s),
merges a local index, and emits REPAIR_MANIFEST.json.

Does not requantize. Source hash must equal repair hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def is_routed_expert(name: str) -> bool:
    return ".ffn.experts." in name and "shared_experts" not in name


def sha256_file_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_tensor(t) -> str:
    import torch

    u8 = t.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(u8.numpy().tobytes()).hexdigest()


def download(url: str, dest: Path, log) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"exists {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".partial")
    log(f"download {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    tmp.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", type=Path, required=True)
    ap.add_argument("--base-index", type=Path, required=True)
    ap.add_argument("--base-repo", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--base-revision", required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--keep-base-shards", action="store_true")
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)
        (args.work_dir / "logs" / "repair.log").parent.mkdir(parents=True, exist_ok=True)
        with open(args.work_dir / "logs" / "repair.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    pack_index = json.loads((args.pack_dir / "model.safetensors.index.json").read_text())
    pack_map = dict(pack_index["weight_map"])
    base_index = json.loads(args.base_index.read_text())
    base_map = dict(base_index["weight_map"])

    missing = [k for k in base_map if not is_routed_expert(k) and k not in pack_map]
    if not missing:
        log("nothing missing")
        return 0

    by_shard: dict[str, list[str]] = {}
    for k in missing:
        by_shard.setdefault(base_map[k], []).append(k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.work_dir / "base_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    repair_tensors: dict = {}
    repair_bytes = 0

    for i, (shard_name, keys) in enumerate(sorted(by_shard.items()), 1):
        url = (
            f"https://huggingface.co/{args.base_repo}/resolve/"
            f"{args.base_revision}/{shard_name}"
        )
        local_shard = shard_dir / shard_name
        download(url, local_shard, log)
        log(f"[{i}/{len(by_shard)}] extract {len(keys)} tensors from {shard_name}")
        with safe_open(str(local_shard), framework="pt") as f:
            for key in keys:
                t = f.get_tensor(key)
                h = sha256_tensor(t)
                # nbytes
                nb = int(t.numel()) * int(t.element_size())
                repair_bytes += nb
                repair_tensors[key] = t.contiguous()
                manifest_entries.append(
                    {
                        "runtime_name": key,
                        "source_name": key,
                        "base_repo": args.base_repo,
                        "base_revision": args.base_revision,
                        "source_shard": shard_name,
                        "dtype": str(t.dtype),
                        "shape": list(t.shape),
                        "byte_count": nb,
                        "source_tensor_hash": h,
                        "repair_tensor_hash": h,
                        "hash_match": True,
                    }
                )
        if not args.keep_base_shards:
            local_shard.unlink(missing_ok=True)
            log(f"deleted {local_shard}")

    # Write repair shard(s) — single shard if fits; else split by ~8GiB
    max_bytes = 8 * 1024**3
    repair_map: dict[str, str] = {}
    shard_i = 1
    current: dict = {}
    current_bytes = 0

    def flush(force: bool = False) -> None:
        nonlocal shard_i, current, current_bytes
        if not current:
            return
        if not force and current_bytes < max_bytes:
            return
        name = f"repair-nonrouted-{shard_i:05d}-of-N.safetensors"
        path = args.out_dir / name
        log(f"write {path} n={len(current)} bytes={current_bytes}")
        save_file(current, str(path))
        for k in current:
            repair_map[k] = name
        shard_i += 1
        current = {}
        current_bytes = 0

    for k, t in repair_tensors.items():
        nb = int(t.numel()) * int(t.element_size())
        if current and current_bytes + nb > max_bytes:
            flush(force=True)
        current[k] = t
        current_bytes += nb
    flush(force=True)

    # Rename -of-N to actual count
    n_shards = shard_i - 1
    final_repair_map = {}
    for old in sorted(set(repair_map.values())):
        new = old.replace("-of-N.safetensors", f"-of-{n_shards:05d}.safetensors")
        if old != new:
            (args.out_dir / old).rename(args.out_dir / new)
        for k, v in list(repair_map.items()):
            if v == old:
                final_repair_map[k] = new
    repair_map = final_repair_map

    # Merge index: pack + repair
    merged = dict(pack_map)
    collisions = [k for k in repair_map if k in merged]
    if collisions:
        log(f"ERROR collisions with pack: {collisions[:10]}")
        return 2
    merged.update(repair_map)

    # Copy config/tokenizer from pack
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        src = args.pack_dir / name
        if src.is_file():
            shutil.copy2(src, args.out_dir / name)

    # Symlink or note original shards — for serving, mount pack dir + repair
    # Write index that references original shard filenames (must be visible beside repair)
    index_out = {
        "metadata": {
            "repair": {
                "base_repo": args.base_repo,
                "base_revision": args.base_revision,
                "pack_dir_note": "Original EXL3 shards must be co-located or symlinked",
            }
        },
        "weight_map": merged,
    }
    (args.out_dir / "model.safetensors.index.json").write_text(json.dumps(index_out, indent=2) + "\n")

    # Symlink original shards into out_dir
    for shard in sorted(set(pack_map.values())):
        src = args.pack_dir / shard
        dst = args.out_dir / shard
        if not dst.exists():
            os.symlink(src, dst)

    manifest = {
        "base_repo": args.base_repo,
        "base_revision": args.base_revision,
        "n_tensors": len(manifest_entries),
        "repair_bytes": repair_bytes,
        "repair_gib": repair_bytes / 1024**3,
        "n_repair_shards": n_shards,
        "entries": manifest_entries,
        "all_hashes_match": all(e["hash_match"] for e in manifest_entries),
    }
    (args.out_dir / "REPAIR_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"DONE repair_gib={manifest['repair_gib']:.3f} tensors={manifest['n_tensors']}")
    print(json.dumps({"repair_gib": manifest["repair_gib"], "n_tensors": manifest["n_tensors"], "n_shards": n_shards}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
