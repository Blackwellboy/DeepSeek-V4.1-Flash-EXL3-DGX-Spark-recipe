#!/usr/bin/env python3
"""Runtime-metadata attestation for immutable mixed-format EXL3 snapshots.

The physical pack validator intentionally requires mixed-format delegation metadata
in ``config.json``. Some already-published canonical snapshots predate that
contract. Rather than mutate those model files or globally relax validation, this
module permits a runtime-only metadata override only when the *entire locked
snapshot identity* matches a checked-in attestation.

A changed config, shard header, tensor count, shard byte count, K histogram, or
codebook-marker count immediately invalidates the attestation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from runtime_lock import load_lock
from validate_pack import validate_pack

RECIPE_ROOT = Path(__file__).resolve().parents[1]

# These are the only base-validator errors an immutable metadata attestation may
# satisfy. Structural/layout/capacity errors are never removed.
ATTESTABLE_METADATA_ERRORS = {
    "non_routed_quantization is missing",
    "non_routed_quantization.quant_method must be deepseek_v4_fp8",
    "non_routed_quantization.weight_block_size must be [32, 32]",
    "mtp_experts must be 'source'",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime_metadata_attestation(topology: str) -> dict[str, Any] | None:
    lock = load_lock()
    model_contract = lock["models"][topology]
    rel = model_contract.get("runtime_metadata_attestation")
    if not rel:
        return None
    path = (RECIPE_ROOT / str(rel)).resolve()
    try:
        path.relative_to(RECIPE_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"metadata attestation escapes recipe root: {path}") from exc
    if not path.is_file():
        raise RuntimeError(f"metadata attestation is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"metadata attestation must be a JSON object: {path}")
    return data


def attestation_mismatches(
    attestation: dict[str, Any],
    result: dict[str, Any],
    config_sha256: str,
) -> list[str]:
    """Return every immutable snapshot identity mismatch."""
    mismatches: list[str] = []

    def check(name: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            mismatches.append(f"{name}: observed={observed!r} expected={expected!r}")

    check("config_sha256", config_sha256, attestation.get("config_sha256"))
    check(
        "index_tensor_count",
        result.get("index_tensor_count"),
        attestation.get("index_tensor_count"),
    )
    check(
        "materialized_shard_bytes",
        result.get("materialized_shard_bytes"),
        attestation.get("materialized_shard_bytes"),
    )
    check(
        "trellis_k_histogram",
        result.get("trellis_k_histogram"),
        attestation.get("trellis_k_histogram"),
    )
    check(
        "codebook_marker_counts",
        result.get("codebook_marker_counts"),
        attestation.get("codebook_marker_counts"),
    )
    check(
        "shard_header_sha256",
        result.get("shard_header_sha256"),
        attestation.get("shard_header_sha256"),
    )
    return mismatches


def validate_pack_with_locked_metadata(
    model_dir: Path,
    topology: str,
    reserve_gib: float,
    *,
    strict_locked_snapshot: bool = False,
) -> dict[str, Any]:
    """Run physical validation, then satisfy metadata only by exact attestation.

    Attestation is intentionally allowed only with ``strict_locked_snapshot``.
    Arbitrary local repacks must carry their own complete runtime metadata.
    """
    result = validate_pack(
        model_dir,
        topology,
        reserve_gib,
        strict_locked_snapshot=strict_locked_snapshot,
    )
    root = Path(model_dir).expanduser().resolve()
    config_path = root / "config.json"
    config_sha = _sha256_file(config_path) if config_path.is_file() else ""
    result["config_sha256"] = config_sha
    result["metadata_contract_source"] = "checkpoint"
    result["metadata_attestation_match"] = False
    result["metadata_attestation_mismatches"] = []
    result["runtime_hf_overrides"] = {}

    metadata_errors = [
        error for error in result.get("errors", []) if error in ATTESTABLE_METADATA_ERRORS
    ]
    result["checkpoint_metadata_errors"] = list(metadata_errors)
    if not metadata_errors:
        result["deployable_with_current_pinned_loader"] = not result.get("errors")
        return result

    attestation = load_runtime_metadata_attestation(topology)
    if not strict_locked_snapshot or attestation is None:
        if attestation is not None and not strict_locked_snapshot:
            result["metadata_attestation_mismatches"] = [
                "locked metadata attestation requires --strict-locked-snapshot"
            ]
        result["deployable_with_current_pinned_loader"] = False
        return result

    lock = load_lock()
    model_contract = lock["models"][topology]
    identity_mismatches: list[str] = []
    if attestation.get("model_repo") != model_contract.get("repo_id"):
        identity_mismatches.append("attestation model_repo does not match runtime lock")
    if attestation.get("revision") != model_contract.get("revision"):
        identity_mismatches.append("attestation revision does not match runtime lock")
    identity_mismatches.extend(attestation_mismatches(attestation, result, config_sha))
    result["metadata_attestation_mismatches"] = identity_mismatches

    if identity_mismatches:
        result["deployable_with_current_pinned_loader"] = False
        return result

    overrides = attestation.get("runtime_hf_overrides")
    if not isinstance(overrides, dict) or not isinstance(
        overrides.get("quantization_config"), dict
    ):
        result["metadata_attestation_mismatches"] = [
            "attestation runtime_hf_overrides.quantization_config is missing/invalid"
        ]
        result["deployable_with_current_pinned_loader"] = False
        return result

    # Remove only the four metadata errors listed above. Any structural, K,
    # safetensors, disk-reserve, or other error remains a hard failure.
    result["errors"] = [
        error for error in result.get("errors", []) if error not in ATTESTABLE_METADATA_ERRORS
    ]
    result["metadata_contract_source"] = "locked_snapshot_attestation"
    result["metadata_attestation_match"] = True
    result["runtime_hf_overrides"] = overrides
    result["deployable_with_current_pinned_loader"] = not result["errors"]
    return result
