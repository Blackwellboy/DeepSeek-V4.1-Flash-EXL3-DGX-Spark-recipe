#!/usr/bin/env python3
"""Build supplemental safetensors repair pack for missing non-routed V4.1 tensors.

Downloads base shards one at a time (size-verified), extracts only missing
required tensors (exact native bytes), appends to repair shard(s) incrementally,
merges a local index, emits REPAIR_MANIFEST.json. Resumable via manifest.
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


def sha256_tensor(t) -> str:
    import torch

    u8 = t.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(u8.numpy().tobytes()).hexdigest()


def download(url: str, dest: Path, log) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log(f"exists {dest} ({dest.stat().st_size} bytes)")
        return
    log(f"download {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-repair/1.0"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        expected = r.headers.get("Content-Length")
        expected_n = int(expected) if expected else None
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    got = tmp.stat().st_size
    if expected_n is not None and got != expected_n:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete download {dest.name}: got {got} expected {expected_n}")
    if got < 1_000_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download too small: {got}")
    tmp.rename(dest)
    log(f"downloaded_ok {dest.name} bytes={got}")


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

    # Normalize filesystem roots once so merged-pack symlinks never depend on
    # the caller's current working directory.
    args.pack_dir = args.pack_dir.resolve()
    args.base_index = args.base_index.resolve()
    args.work_dir = args.work_dir.resolve()
    args.out_dir = args.out_dir.resolve()

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_path = args.work_dir / "logs" / "repair.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    pack_index = json.loads((args.pack_dir / "model.safetensors.index.json").read_text())
    pack_map = dict(pack_index["weight_map"])
    base_map = dict(json.loads(args.base_index.read_text())["weight_map"])

    missing = [k for k in base_map if not is_routed_expert(k) and k not in pack_map]
    by_shard: dict[str, list[str]] = {}
    for k in missing:
        by_shard.setdefault(base_map[k], []).append(k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.work_dir / "base_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "REPAIR_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        done = {e["runtime_name"] for e in manifest.get("entries", [])}
        log(f"resume: {len(done)} tensors already in manifest")
    else:
        manifest = {
            "base_repo": args.base_repo,
            "base_revision": args.base_revision,
            "entries": [],
            "repair_bytes": 0,
        }
        done = set()

    repair_map_path = args.work_dir / "repair_weight_map.json"
    if repair_map_path.is_file():
        repair_map = json.loads(repair_map_path.read_text())
    else:
        repair_map = {}

    batch: dict = {}
    batch_bytes = 0
    max_batch = 4 * 1024**3
    shard_i = 1 + len({v for v in repair_map.values()})

    def flush_batch(force: bool = False) -> None:
        nonlocal batch, batch_bytes, shard_i
        if not batch:
            return
        if not force and batch_bytes < max_batch:
            return
        name = f"repair-nonrouted-{shard_i:05d}.safetensors"
        path = args.out_dir / name
        log(f"write {path} n={len(batch)} bytes={batch_bytes}")
        if path.exists():
            existing = {}
            with safe_open(str(path), framework="pt") as f:
                for k in f.keys():
                    existing[k] = f.get_tensor(k)
            existing.update(batch)
            save_file(existing, str(path))
        else:
            save_file(batch, str(path))
        for k in batch:
            repair_map[k] = name
        repair_map_path.write_text(json.dumps(repair_map, indent=2))
        shard_i += 1
        batch = {}
        batch_bytes = 0

    for i, (shard_name, keys) in enumerate(sorted(by_shard.items()), 1):
        todo = [k for k in keys if k not in done]
        if not todo:
            log(f"[{i}/{len(by_shard)}] skip {shard_name} (all done)")
            continue
        url = (
            f"https://huggingface.co/{args.base_repo}/resolve/"
            f"{args.base_revision}/{shard_name}"
        )
        local_shard = shard_dir / shard_name
        partial = local_shard.with_suffix(local_shard.suffix + ".partial")
        partial.unlink(missing_ok=True)
        if local_shard.exists():
            try:
                req = urllib.request.Request(
                    url, method="HEAD", headers={"User-Agent": "vllm-repair/1.0"}
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    exp = int(r.headers.get("Content-Length") or 0)
                if exp and local_shard.stat().st_size != exp:
                    log(
                        f"size mismatch {local_shard.name}: "
                        f"{local_shard.stat().st_size} != {exp}; redownload"
                    )
                    local_shard.unlink()
            except Exception as e:
                log(f"HEAD warn {e}")
        download(url, local_shard, log)
        log(f"[{i}/{len(by_shard)}] extract {len(todo)} tensors from {shard_name}")
        with safe_open(str(local_shard), framework="pt") as f:
            for key in todo:
                t = f.get_tensor(key).contiguous()
                h = sha256_tensor(t)
                nb = int(t.numel()) * int(t.element_size())
                batch[key] = t
                batch_bytes += nb
                manifest["entries"].append(
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
                manifest["repair_bytes"] = int(manifest.get("repair_bytes", 0)) + nb
                done.add(key)
                if batch_bytes >= max_batch:
                    flush_batch(force=True)
                    manifest_path.write_text(json.dumps(manifest, indent=2))
        flush_batch(force=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if not args.keep_base_shards:
            local_shard.unlink(missing_ok=True)
            log(f"deleted {local_shard}")

    flush_batch(force=True)

    merged = dict(pack_map)
    collisions = [k for k in repair_map if k in merged]
    if collisions:
        log(f"ERROR collisions: {collisions[:5]}")
        return 2
    merged.update(repair_map)
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
    (args.out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "repair": {
                        "base_repo": args.base_repo,
                        "base_revision": args.base_revision,
                    }
                },
                "weight_map": merged,
            },
            indent=2,
        )
        + "\n"
    )
    for shard in sorted(set(pack_map.values())):
        src = (args.pack_dir / shard).resolve()
        dst = args.out_dir / shard
        if not dst.exists() and src.exists():
            os.symlink(str(src), str(dst))

    manifest["n_tensors"] = len(manifest["entries"])
    manifest["repair_gib"] = manifest["repair_bytes"] / 1024**3
    manifest["all_hashes_match"] = all(e["hash_match"] for e in manifest["entries"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"DONE repair_gib={manifest['repair_gib']:.3f} tensors={manifest['n_tensors']}")
    print(
        json.dumps(
            {
                "repair_gib": manifest["repair_gib"],
                "n_tensors": manifest["n_tensors"],
                "all_hashes_match": manifest["all_hashes_match"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
