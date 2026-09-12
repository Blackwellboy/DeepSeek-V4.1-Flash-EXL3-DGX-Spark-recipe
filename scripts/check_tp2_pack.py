#!/usr/bin/env python3
"""Fail-fast TP2 checkpoint validator.

Checks a *materialized local snapshot* before Docker/vLLM startup:
- index-referenced shards exist;
- shard files are real safetensors payloads, not Git LFS/Xet pointer stubs/HTML;
- safetensors headers are structurally valid without loading tensor payloads;
- EXL3 trellis K is inferred from shape[-1] / 16;
- reports K distribution and per-layer mixed-K geometry;
- reports local filesystem free-space headroom.

This does not modify weights. A pointer/truncated shard must be re-materialized with
Hugging Face tooling; a tensor-mixed layer requires a compatible loader or repack.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
MAX_HEADER_BYTES = 256 * 1024 * 1024


def gib(n: int) -> float:
    return n / (1024**3)


def read_header(path: Path) -> tuple[str, dict[str, Any] | None, str | None]:
    size = path.stat().st_size
    with path.open("rb") as fh:
        first = fh.read(256)
        if first.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return "pointer", None, "Git LFS pointer stub; large object was not materialized"
        if first.startswith((b"<html", b"<!DOCTYPE", b"<?xml")):
            return "html", None, "HTML/XML response saved instead of a safetensors shard"
        if size < 16:
            return "truncated", None, f"file is only {size} bytes"
        fh.seek(0)
        raw = fh.read(8)
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 1 or header_len > MAX_HEADER_BYTES:
            return "invalid", None, f"implausible safetensors header length {header_len}"
        if 8 + header_len > size:
            return "truncated", None, f"header needs {8 + header_len} bytes but file is {size} bytes"
        header_raw = fh.read(header_len)
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "invalid", None, f"header JSON decode failed: {exc}"
    if not isinstance(header, dict):
        return "invalid", None, "safetensors header is not a JSON object"
    return "ok", header, None


def trellis_k(meta: Any) -> int | None:
    if not isinstance(meta, dict):
        return None
    shape = meta.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    try:
        words = int(shape[-1])
    except (TypeError, ValueError):
        return None
    if words <= 0 or words % 16:
        return None
    return words // 16


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("--reserve-gib", type=float, default=32.0,
                   help="minimum free space to retain after the materialized snapshot")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    root = args.model_dir.expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    result: dict[str, Any] = {
        "model_dir": str(root),
        "index": str(index_path),
        "reserve_gib": args.reserve_gib,
        "errors": [],
        "warnings": [],
    }
    if not root.is_dir():
        result["errors"].append("model directory does not exist")
    if not index_path.is_file():
        result["errors"].append("model.safetensors.index.json is missing")

    index: dict[str, Any] = {}
    if not result["errors"]:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result["errors"].append(f"index JSON is invalid: {exc}")

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        result["errors"].append("index weight_map is missing/empty")
        weight_map = {}

    shards = sorted({str(v) for v in weight_map.values()})
    result["index_tensor_count"] = len(weight_map)
    result["index_shard_count"] = len(shards)

    statuses: Counter[str] = Counter()
    bad_shards: list[dict[str, str]] = []
    k_hist: Counter[int] = Counter()
    layer_ks: dict[int, set[int]] = defaultdict(set)
    total_bytes = 0

    for shard_name in shards:
        path = root / shard_name
        if not path.is_file():
            statuses["missing"] += 1
            bad_shards.append({"shard": shard_name, "status": "missing", "reason": "not found"})
            continue
        total_bytes += path.stat().st_size
        status, header, reason = read_header(path)
        statuses[status] += 1
        if status != "ok" or header is None:
            bad_shards.append({"shard": shard_name, "status": status, "reason": reason or "unknown"})
            continue
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not (name.endswith("trellis") or name.endswith("_trellis")):
                continue
            k = trellis_k(meta)
            if k is None:
                continue
            k_hist[k] += 1
            match = LAYER_RE.search(name)
            if match:
                layer_ks[int(match.group(1))].add(k)

    mixed_layers = {str(layer): sorted(ks) for layer, ks in sorted(layer_ks.items()) if len(ks) > 1}
    result["shard_status"] = dict(sorted(statuses.items()))
    result["bad_shards"] = bad_shards
    result["materialized_shard_bytes"] = total_bytes
    result["materialized_shard_gib"] = gib(total_bytes)
    result["trellis_k_histogram"] = {str(k): n for k, n in sorted(k_hist.items())}
    result["mixed_k_layers"] = mixed_layers

    usage = shutil.disk_usage(root if root.exists() else Path.cwd())
    result["filesystem_free_gib"] = gib(usage.free)
    result["filesystem_total_gib"] = gib(usage.total)
    result["disk_reserve_ok"] = gib(usage.free) >= args.reserve_gib

    if bad_shards:
        result["errors"].append(
            f"{len(bad_shards)} of {len(shards)} indexed shards are not valid materialized safetensors"
        )
    unsupported = sorted(k for k in k_hist if k < 2 or k > 8)
    if unsupported:
        result["errors"].append(f"trellis K outside EXL3 K2-K8: {unsupported}")
    if any(k in k_hist for k in (7, 8)):
        result["warnings"].append(
            "pack contains K7/K8; the currently pinned vllm-exl3 config rejects K7/K8"
        )
    if mixed_layers:
        result["warnings"].append(
            "pack contains multiple K widths inside the same transformer layer; current vllm-exl3 routed-MoE allocation is layer-uniform K"
        )
    if not result["disk_reserve_ok"]:
        result["errors"].append(
            f"filesystem free space {result['filesystem_free_gib']:.1f} GiB is below reserve {args.reserve_gib:.1f} GiB"
        )

    result["deployable_with_current_pinned_loader"] = not result["errors"] and not mixed_layers and not any(
        k in k_hist for k in (7, 8)
    )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"TP2 pack: {root}")
        print(f"Indexed shards: {len(shards)}  tensors: {len(weight_map)}")
        print(f"Shard status: {dict(statuses)}")
        print(f"Trellis K histogram: {dict(sorted(k_hist.items()))}")
        print(f"Mixed-K layers: {len(mixed_layers)}")
        print(f"Materialized shard bytes: {gib(total_bytes):.2f} GiB")
        print(f"Filesystem free: {gib(usage.free):.2f} GiB")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
        for error in result["errors"]:
            print(f"ERROR: {error}")
        if bad_shards:
            print("Bad shards:")
            for item in bad_shards:
                print(f"  {item['shard']}: {item['status']} - {item['reason']}")
        print("DEPLOYABLE_CURRENT_LOADER=" + ("YES" if result["deployable_with_current_pinned_loader"] else "NO"))

    return 0 if result["deployable_with_current_pinned_loader"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
