#!/usr/bin/env python3
"""Verify the TP2 runtime-metadata attestation against the locked HF snapshot.

Only config/index JSON and safetensors header byte ranges are fetched. Tensor
payloads are never downloaded. This independently binds the checked-in local
header receipt to the immutable Hugging Face revision before hardware loading.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any

from attested_pack import load_runtime_metadata_attestation
from probe_remote_pack import (
    MAX_REMOTE_HEADER_BYTES,
    fetch_range,
    fetch_small,
    hf_resolve_url,
)
from runtime_lock import load_lock


def probe(token: str | None = None) -> dict[str, Any]:
    lock = load_lock()
    contract = lock["models"]["tp2"]
    attestation = load_runtime_metadata_attestation("tp2")
    result: dict[str, Any] = {
        "schema": "dsv41-exl3-remote-metadata-attestation.v1",
        "repo_id": contract["repo_id"],
        "revision": contract["revision"],
        "errors": [],
        "shard_header_sha256": {},
    }
    errors: list[str] = result["errors"]
    if attestation is None:
        errors.append("runtime lock has no TP2 metadata attestation")
        result["remote_metadata_attestation_match"] = False
        return result

    repo = str(contract["repo_id"])
    revision = str(contract["revision"])
    config_raw = fetch_small(hf_resolve_url(repo, revision, "config.json"), token)
    index_raw = fetch_small(
        hf_resolve_url(repo, revision, "model.safetensors.index.json"), token
    )
    result["config_sha256"] = hashlib.sha256(config_raw).hexdigest()
    try:
        index = json.loads(index_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"remote index JSON invalid: {exc}")
        result["remote_metadata_attestation_match"] = False
        return result
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        errors.append("remote index weight_map is missing/empty")
        result["remote_metadata_attestation_match"] = False
        return result

    shards = sorted({str(v) for v in weight_map.values()})
    result["index_tensor_count"] = len(weight_map)
    result["index_shard_count"] = len(shards)
    shard_sizes: dict[str, int] = {}
    header_hashes: dict[str, str] = {}
    for shard in shards:
        url = hf_resolve_url(repo, revision, shard)
        prefix, total = fetch_range(url, token, 0, 7)
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 1 or header_len > MAX_REMOTE_HEADER_BYTES:
            errors.append(f"{shard}: implausible safetensors header length {header_len}")
            continue
        header_raw, total_again = fetch_range(url, token, 8, 8 + header_len - 1)
        if total_again != total:
            errors.append(f"{shard}: size changed during range probe {total}->{total_again}")
            continue
        try:
            parsed = json.loads(header_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{shard}: invalid safetensors header JSON: {exc}")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"{shard}: safetensors header is not a JSON object")
            continue
        shard_sizes[shard] = total
        header_hashes[shard] = hashlib.sha256(header_raw).hexdigest()

    result["shard_header_sha256"] = dict(sorted(header_hashes.items()))
    result["remote_total_shard_bytes"] = sum(shard_sizes.values())

    expected = {
        "config_sha256": attestation.get("config_sha256"),
        "index_tensor_count": attestation.get("index_tensor_count"),
        "index_shard_count": int(contract.get("expected_shards", 0)),
        "shard_header_sha256": attestation.get("shard_header_sha256"),
    }
    observed = {
        "config_sha256": result["config_sha256"],
        "index_tensor_count": result["index_tensor_count"],
        "index_shard_count": result["index_shard_count"],
        "shard_header_sha256": result["shard_header_sha256"],
    }
    for key in expected:
        if observed[key] != expected[key]:
            errors.append(f"{key} does not match checked-in TP2 attestation")

    # Remote total size includes the 8-byte prefix + header bytes in addition to
    # tensor payload. The local validator's materialized_shard_bytes is full file
    # size too, so this is directly comparable.
    if result["remote_total_shard_bytes"] != attestation.get("materialized_shard_bytes"):
        errors.append("remote_total_shard_bytes does not match checked-in TP2 attestation")

    result["remote_metadata_attestation_match"] = not errors
    return result


def main() -> int:
    token = os.environ.get("HF_TOKEN") or None
    try:
        result = probe(token)
    except Exception as exc:
        result = {
            "schema": "dsv41-exl3-remote-metadata-attestation.v1",
            "errors": [f"remote attestation probe failed: {type(exc).__name__}: {exc}"],
            "remote_metadata_attestation_match": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "REMOTE_METADATA_ATTESTATION="
        + ("PASS" if result.get("remote_metadata_attestation_match") else "FAIL")
    )
    return 0 if result.get("remote_metadata_attestation_match") else 2


if __name__ == "__main__":
    raise SystemExit(main())
