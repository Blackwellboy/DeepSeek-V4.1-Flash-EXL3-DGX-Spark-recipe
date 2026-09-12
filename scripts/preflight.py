#!/usr/bin/env python3
"""Preflight a DeepSeek-V4.1 EXL3 checkpoint and every active Spark runtime node."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

from runtime_lock import load_lock
from validate_pack import validate_pack


def git_head(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def load_model_config(model: str, revision: str | None) -> tuple[dict[str, Any], str]:
    path = Path(model).expanduser()
    if path.exists():
        config_path = path / "config.json" if path.is_dir() else path
        if not config_path.is_file():
            raise FileNotFoundError(f"config.json not found at {config_path}")
        return json.loads(config_path.read_text(encoding="utf-8")), str(config_path)

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        repo_id=model,
        filename="config.json",
        revision=revision or None,
        token=os.environ.get("HF_TOKEN") or None,
    )
    return json.loads(Path(config_path).read_text(encoding="utf-8")), config_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--pack-reserve-gib", type=float, default=32.0)
    args = parser.parse_args()

    lock = load_lock()
    expected_plugin = str(lock["vllm_exl3"]["commit"])
    expected_exllama = str(lock["exllamav3"]["commit"])
    expected_abi = int(lock["native_abi_min"])
    topology = f"tp{args.tp}"
    model_lock = lock["models"][topology]

    issues: list[str] = []
    warnings: list[str] = []

    import torch
    import vllm
    import vllm_exl3
    import vllm_exl3_c
    from vllm_exl3 import plan_deepseek_v41

    vllm_exl3.register()
    diagnostics = vllm_exl3.runtime_diagnostics()
    plan = plan_deepseek_v41(tensor_parallel_size=args.tp, expert_parallel=True).to_dict()

    locked_repo = str(model_lock.get("repo_id") or "")
    locked_revision = str(model_lock.get("revision") or "")
    if args.model == locked_repo and locked_revision and args.revision != locked_revision:
        issues.append(
            f"canonical {topology} model must use locked revision {locked_revision}; got {args.revision or '<main>'}"
        )

    config, config_source = load_model_config(args.model, args.revision or None)
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        quant = {}
        issues.append("config.json has no quantization_config object")
    if str(quant.get("quant_method", "")).lower() != "exl3":
        issues.append("quantization_config.quant_method must be 'exl3'")

    source = quant.get("non_routed_quantization")
    if not isinstance(source, dict):
        source = {}
        issues.append("non_routed_quantization is missing")
    if str(source.get("quant_method", "")).lower() != "deepseek_v4_fp8":
        issues.append("non_routed_quantization.quant_method must be deepseek_v4_fp8")
    if source.get("weight_block_size") != [32, 32]:
        issues.append("non_routed_quantization.weight_block_size must be [32, 32]")
    if quant.get("mtp_experts") != "source":
        issues.append("mtp_experts must be 'source' for the baseline DSpark policy")

    codebook = str(quant.get("codebook", "mcg")).lower()
    if codebook != "mcg":
        warnings.append(f"codebook={codebook!r}; custom native p2b remains MCG-only")

    architectures = config.get("architectures") or []
    model_type = str(config.get("model_type", ""))
    if not (
        model_type == "deepseek_v41"
        or any("DeepseekV41" in str(item) for item in architectures)
    ):
        warnings.append(
            f"model identity does not look like DeepSeek V4.1: model_type={model_type!r}, architectures={architectures!r}"
        )

    model_path = Path(args.model).expanduser()
    physical_pack: dict[str, Any] | None = None
    if model_path.is_dir():
        physical_pack = validate_pack(model_path, topology, args.pack_reserve_gib)
        if not physical_pack["deployable_with_current_pinned_loader"]:
            issues.extend(f"pack: {item}" for item in physical_pack["errors"])
        warnings.extend(f"pack: {item}" for item in physical_pack["warnings"])
    else:
        warnings.append(
            "remote model id: physical shard/K/index validation was not run; materialize locally before claiming deployment readiness"
        )

    abi = int(getattr(vllm_exl3_c, "P2B_MOE_ABI_VERSION", 0))
    local_plugin = git_head("/opt/vllm-exl3")
    local_exllama = git_head("/opt/exllamav3")
    if abi < expected_abi:
        issues.append(f"vllm_exl3_c ABI {abi} is stale; runtime lock requires ABI >= {expected_abi}")
    if local_plugin != expected_plugin:
        issues.append(f"head vllm-exl3 revision is {local_plugin}, expected {expected_plugin}")
    if local_exllama != expected_exllama:
        issues.append(f"head ExLlamaV3 revision is {local_exllama}, expected {expected_exllama}")

    mixed_diag = diagnostics.get("mixed_k", {}) if isinstance(diagnostics, dict) else {}
    locked_caps = lock["capabilities"]
    if mixed_diag.get("config_bits") != locked_caps["accepted_exl3_config_k"]:
        issues.append(
            f"runtime accepted EXL3 K {mixed_diag.get('config_bits')} != lock {locked_caps['accepted_exl3_config_k']}"
        )
    if bool(mixed_diag.get("tensor_level_mixed_k_within_layer")) != bool(
        locked_caps["tensor_level_mixed_k_within_layer"]
    ):
        issues.append("runtime tensor-level mixed-K capability differs from runtime lock")

    capability = None
    gpu_name = None
    if torch.cuda.is_available():
        capability = list(torch.cuda.get_device_capability(0))
        gpu_name = torch.cuda.get_device_name(0)
        if tuple(capability) != (12, 1):
            warnings.append(f"local CUDA capability is {tuple(capability)}, expected DGX Spark sm_121")
    else:
        issues.append("CUDA is not available inside the head runtime container")

    ray_resources: dict[str, Any] = {}
    node_identities: list[dict[str, Any]] = []
    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
        ray_resources = dict(ray.cluster_resources())
        ray_gpus = int(ray_resources.get("GPU", 0))
        if ray_gpus < args.tp:
            issues.append(f"Ray exposes {ray_gpus} GPUs but TP{args.tp} requires {args.tp}")

        @ray.remote(num_cpus=0.01)
        def node_identity() -> dict[str, Any]:
            import subprocess as _subprocess
            import socket as _socket
            import torch as _torch
            import vllm as _vllm
            import vllm_exl3_c as _native

            def _head(path: str) -> str | None:
                try:
                    return _subprocess.check_output(
                        ["git", "-C", path, "rev-parse", "HEAD"],
                        text=True,
                        stderr=_subprocess.DEVNULL,
                    ).strip()
                except Exception:
                    return None

            return {
                "hostname": _socket.gethostname(),
                "vllm": getattr(_vllm, "__version__", "unknown"),
                "vllm_exl3_git": _head("/opt/vllm-exl3"),
                "exllamav3_git": _head("/opt/exllamav3"),
                "native_abi": int(getattr(_native, "P2B_MOE_ABI_VERSION", 0)),
                "gpu_name": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
                "cuda_capability": list(_torch.cuda.get_device_capability(0)) if _torch.cuda.is_available() else None,
            }

        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        for node in alive_nodes:
            strategy = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ident = ray.get(node_identity.options(scheduling_strategy=strategy).remote())
            ident["node_id"] = node["NodeID"]
            node_identities.append(ident)

        if len(node_identities) < args.tp:
            issues.append(f"only {len(node_identities)} live Ray nodes found for TP{args.tp}")

        for ident in node_identities:
            label = ident.get("hostname") or ident.get("node_id")
            if ident.get("vllm_exl3_git") != expected_plugin:
                issues.append(
                    f"node {label} has vllm-exl3 {ident.get('vllm_exl3_git')}, expected {expected_plugin}"
                )
            if ident.get("exllamav3_git") != expected_exllama:
                issues.append(
                    f"node {label} has ExLlamaV3 {ident.get('exllamav3_git')}, expected {expected_exllama}"
                )
            if int(ident.get("native_abi") or 0) < expected_abi:
                issues.append(f"node {label} has native ABI {ident.get('native_abi')}, expected >= {expected_abi}")
    except Exception as exc:
        issues.append(f"could not inspect Ray cluster: {exc}")

    if plan.get("preferred_first_boot_backend") != "exllamav3":
        issues.append(
            f"planner reports first-boot backend {plan.get('preferred_first_boot_backend')!r}; locked recipe expects exllamav3"
        )

    report = {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "runtime_lock": lock,
        "model": args.model,
        "revision": args.revision or None,
        "config_source": config_source,
        "model_type": model_type,
        "architectures": architectures,
        "physical_pack": physical_pack,
        "quantization": {
            "quant_method": quant.get("quant_method"),
            "bits": quant.get("bits"),
            "codebook": quant.get("codebook"),
            "scope": quant.get("scope"),
            "layer_bits_present": isinstance(quant.get("layer_bits"), dict),
            "mtp_experts": quant.get("mtp_experts"),
            "mtp_experts_start_layer": quant.get("mtp_experts_start_layer"),
            "source_quant_method": source.get("quant_method"),
            "source_weight_block_size": source.get("weight_block_size"),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "vllm": getattr(vllm, "__version__", "unknown"),
            "native_abi": abi,
            "vllm_exl3_git": local_plugin,
            "exllamav3_git": local_exllama,
            "gpu_name": gpu_name,
            "cuda_capability": capability,
        },
        "ray_resources": ray_resources,
        "ray_nodes": node_identities,
        "layout": plan,
        "vllm_exl3_diagnostics": diagnostics,
    }

    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if issues:
        print("\nPRECHECK FAILED", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 2

    print("\nPRECHECK PASSED")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
