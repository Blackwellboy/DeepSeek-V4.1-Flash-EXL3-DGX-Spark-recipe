#!/usr/bin/env python3
"""Fail-closed TP4 disk-Engram readiness probe across Ray nodes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@ray.remote(num_cpus=0)
def _probe(model_dir: str) -> dict[str, object]:
    result: dict[str, object] = {
        "host": socket.gethostname(),
        "model_dir": model_dir,
        "errors": [],
    }
    errors: list[str] = result["errors"]  # type: ignore[assignment]

    if not _truthy(os.environ.get("VLLM_ENGRAM_DISK_BACKED")):
        errors.append("VLLM_ENGRAM_DISK_BACKED is not enabled in this Ray worker")
    if os.environ.get("VLLM_ENGRAM_MODEL_DIR") != model_dir:
        errors.append(
            "VLLM_ENGRAM_MODEL_DIR mismatch: "
            f"{os.environ.get('VLLM_ENGRAM_MODEL_DIR')!r} != {model_dir!r}"
        )
    if os.environ.get("VLLM_EXL3_MODEL_DIR") != model_dir:
        errors.append(
            "VLLM_EXL3_MODEL_DIR mismatch: "
            f"{os.environ.get('VLLM_EXL3_MODEL_DIR')!r} != {model_dir!r}"
        )

    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    config = root / "config.json"
    if not root.is_dir():
        errors.append(f"model directory missing: {root}")
    if not index.is_file():
        errors.append(f"safetensors index missing: {index}")
    if not config.is_file():
        errors.append(f"config missing: {config}")

    try:
        mount = subprocess.check_output(
            ["findmnt", "-no", "FSTYPE,SOURCE,TARGET", "-T", str(root)],
            text=True,
        ).strip()
        result["mount"] = mount
        fstype = mount.split()[0].lower() if mount else ""
        if fstype in {"nfs", "nfs4", "cifs", "smb", "fuse.sshfs", "fuse.rclone"}:
            errors.append(f"Engram backing is not node-local storage: {mount}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"findmnt failed for Engram backing: {exc}")

    try:
        from vllm.config.engram import EngramConfig
        from vllm.models.deepseek_v4_1.common.engram_disk import (
            should_skip_engram_embed_tensor,
        )

        if not hasattr(EngramConfig, "disk_backed"):
            errors.append("EngramConfig.disk_backed is missing from runtime overlay")
        if not should_skip_engram_embed_tensor("layers.1.engram.embed.weight"):
            errors.append("weight-loader Engram skip is not active")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"disk-Engram overlay import failed: {exc!r}")

    try:
        import vllm_exl3

        vllm_exl3.register()
        mixed = vllm_exl3.runtime_diagnostics()["mixed_k"]
        result["mixed_k"] = mixed
        if mixed.get("tensor_level_mixed_k_within_layer") is not True:
            errors.append("vllm-exl3 does not advertise tensor-level mixed-K support")
        if mixed.get("heterogeneous_dispatch") != "python_loop":
            errors.append("unexpected heterogeneous mixed-K dispatch contract")
        if mixed.get("cudagraph_qualified") is not False:
            errors.append("mixed-K qualification contract must remain eager-first")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"vllm-exl3 mixed-K runtime probe failed: {exc!r}")

    result["ok"] = not errors
    return result


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_disk_engram_cluster.py TP MODEL_DIR", file=sys.stderr)
        return 2
    expected = int(sys.argv[1])
    model_dir = sys.argv[2]
    if expected < 1:
        raise SystemExit("TP must be >= 1")

    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive") and float(node.get("Resources", {}).get("GPU", 0)) >= 1
    ]
    if len(nodes) < expected:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"Ray exposes {len(nodes)} live GPU nodes; expected {expected}",
                },
                indent=2,
            )
        )
        return 2

    refs = []
    for node in nodes[:expected]:
        node_id = str(node["NodeID"])
        refs.append(
            _probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote(model_dir)
        )
    reports = ray.get(refs)
    ok = len(reports) == expected and all(bool(item.get("ok")) for item in reports)
    receipt = {
        "schema": "dsv41-disk-engram-cluster-preflight.v1",
        "expected_nodes": expected,
        "checked_nodes": len(reports),
        "model_dir": model_dir,
        "ok": ok,
        "nodes": reports,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
