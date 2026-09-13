#!/usr/bin/env python3
"""TP2 validator with fail-closed immutable-snapshot metadata attestation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attested_pack import validate_pack_with_locked_metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--reserve-gib", type=float, default=32.0)
    parser.add_argument(
        "--strict-locked-snapshot",
        action="store_true",
        help=(
            "require exact locked snapshot identity; required before a runtime-only "
            "metadata attestation may satisfy missing legacy config declarations"
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--print-hf-overrides",
        action="store_true",
        help="print only the attested runtime --hf-overrides JSON; fail if pack validation fails",
    )
    args = parser.parse_args()

    result = validate_pack_with_locked_metadata(
        args.model_dir,
        "tp2",
        args.reserve_gib,
        strict_locked_snapshot=args.strict_locked_snapshot,
    )
    ok = bool(result["deployable_with_current_pinned_loader"])

    if args.print_hf_overrides:
        if not ok:
            for error in result["errors"]:
                print(f"ERROR: {error}", file=__import__("sys").stderr)
            for mismatch in result.get("metadata_attestation_mismatches", []):
                print(f"ERROR: metadata attestation: {mismatch}", file=__import__("sys").stderr)
            return 2
        print(json.dumps(result.get("runtime_hf_overrides", {}), separators=(",", ":"), sort_keys=True))
        return 0

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"TP2 pack: {result['model_dir']}")
        print(f"Shard status: {result.get('shard_status', {})}")
        print(f"Trellis K histogram: {result.get('trellis_k_histogram', {})}")
        print(f"Mixed-K layers: {len(result.get('mixed_k_layers', {}))}")
        print(f"Materialized shard bytes: {result.get('materialized_shard_gib', 0):.2f} GiB")
        print(f"Filesystem free: {result.get('filesystem_free_gib', 0):.2f} GiB")
        print(f"Runtime metadata source: {result.get('metadata_contract_source', 'unknown')}")
        print(f"Metadata attestation match: {result.get('metadata_attestation_match', False)}")
        for mismatch in result.get("metadata_attestation_mismatches", []):
            print(f"ATTESTATION_MISMATCH: {mismatch}")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
        for error in result["errors"]:
            print(f"ERROR: {error}")
        if result.get("bad_shards"):
            print("Bad shards:")
            for item in result["bad_shards"]:
                print(f"  {item['shard']}: {item['status']} - {item['reason']}")
        print("DEPLOYABLE_CURRENT_LOADER=" + ("YES" if ok else "NO"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
