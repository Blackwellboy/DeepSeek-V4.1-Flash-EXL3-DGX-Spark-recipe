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

EXPECTED_VLLM_EXL3 = "8f4517e80416466fa4a3ad2eb28685021d39e95f"
EXPECTED_EXLLAMAV3 = "be57335b087e4f001c5caae061544df3c06ba01e"
EXPECTED_ABI = 3


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
    args = parser.parse_args()

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
        warnings.append(
            f"codebook={codebook!r}; the ABI-3 native p2b path currently expects MCG"
        )

    architectures = config.get("architectures") or []
    model_type = str(config.get("model_type", ""))
    if not (
        model_type == "deepseek_v41"
        or any("DeepseekV41" in str(item) for item in architectures)
    ):
        warnings.append(
            f"model identity does not look like DeepSeek V4.1: model_type={model_type!r}, architectures={architectures!r}"
        )

    abi = int(getattr(vllm_exl3_c, "P2B_MOE_ABI_VERSION", 0))
    local_plugin = git_head("/opt/vllm-exl3")
    local_exllama = git_head("/opt/exllamav3")
    if abi < EXPECTED_ABI:
        issues.append(f"vllm_exl3_c ABI {abi} is stale; recipe requires ABI >= {EXPECTED_ABI}")
    if local_plugin != EXPECTED_VLLM_EXL3:
        issues.append(f"head vllm-exl3 revision is {local_plugin}, expected {EXPECTED_VLLM_EXL3}")
    if local_exllama != EXPECTED_EXLLAMAV3:
        issues.append(f"head ExLlamaV3 revision is {local_exllama}, expected {EXPECTED_EXLLAMAV3}")

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
            if ident.get("vllm_exl3_git") != EXPECTED_VLLM_EXL3:
                issues.append(
                    f"node {label} has vllm-exl3 {ident.get('vllm_exl3_git')}, expected {EXPECTED_VLLM_EXL3}"
                )
            if ident.get("exllamav3_git") != EXPECTED_EXLLAMAV3:
                issues.append(
                    f"node {label} has ExLlamaV3 {ident.get('exllamav3_git')}, expected {EXPECTED_EXLLAMAV3}"
                )
            if int(ident.get("native_abi") or 0) < EXPECTED_ABI:
                issues.append(f"node {label} has native ABI {ident.get('native_abi')}, expected >= {EXPECTED_ABI}")
    except Exception as exc:
        issues.append(f"could not inspect Ray cluster: {exc}")

    if args.tp == 2:
        warnings.append(
            "TP2 owns 192 full experts per rank; this exceeds the current ExLlamaV3 fused 128-expert envelope. "
            "Treat ABI-3 native p2b as experimental and verify pack memory separately."
        )

    report = {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "model": args.model,
        "revision": args.revision or None,
        "config_source": config_source,
        "model_type": model_type,
        "architectures": architectures,
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
