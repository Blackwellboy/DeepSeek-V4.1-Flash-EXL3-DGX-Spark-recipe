#!/usr/bin/env python3
"""Backward-compatible TP2 wrapper around the generic physical pack validator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_pack import validate_pack


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--reserve-gib", type=float, default=32.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = validate_pack(args.model_dir, "tp2", args.reserve_gib)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"TP2 pack: {result['model_dir']}")
        print(f"Shard status: {result.get('shard_status', {})}")
        print(f"Trellis K histogram: {result.get('trellis_k_histogram', {})}")
        print(f"Mixed-K layers: {len(result.get('mixed_k_layers', {}))}")
        print(f"Materialized shard bytes: {result.get('materialized_shard_gib', 0):.2f} GiB")
        print(f"Filesystem free: {result.get('filesystem_free_gib', 0):.2f} GiB")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
        for error in result["errors"]:
            print(f"ERROR: {error}")
        if result.get("bad_shards"):
            print("Bad shards:")
            for item in result["bad_shards"]:
                print(f"  {item['shard']}: {item['status']} - {item['reason']}")
        print(
            "DEPLOYABLE_CURRENT_LOADER="
            + ("YES" if result["deployable_with_current_pinned_loader"] else "NO")
        )
    return 0 if result["deployable_with_current_pinned_loader"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
