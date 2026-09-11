#!/usr/bin/env python3
"""Fail-closed compatibility inspection for DeepSeek-V4.1 EXL3 checkpoints.

This script does not load model weights. It inspects config/index metadata and the
installed ExLlamaV3 architecture registry, then reports whether the checkpoint
matches the known prerequisites for upstream ExLlamaV3's experimental CPU-MoE
offload path.

A PASS here is only a metadata preflight. It is never an end-to-end qualification.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

V41_ARCH = "DeepseekV41ForCausalLM"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def first_architecture(config: dict[str, Any]) -> str | None:
    archs = config.get("architectures")
    if isinstance(archs, list) and archs and isinstance(archs[0], str):
        return archs[0]
    text = config.get("text_config")
    if isinstance(text, dict):
        archs = text.get("architectures")
        if isinstance(archs, list) and archs and isinstance(archs[0], str):
            return archs[0]
    return None


def weight_keys(model_dir: Path) -> list[str]:
    index_candidates = [
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    ]
    for path in index_candidates:
        if not path.is_file():
            continue
        data = load_json(path)
        weight_map = data.get("weight_map")
        if isinstance(weight_map, dict):
            return [str(key) for key in weight_map]
    return []


def inferred_codebooks(qcfg: dict[str, Any], keys: Iterable[str]) -> set[str]:
    result: set[str] = set()
    declared = qcfg.get("codebook")
    if isinstance(declared, str) and declared.strip():
        result.add(declared.strip().lower())
    for key in keys:
        if key.endswith(".mul1"):
            result.add("mul1")
        elif key.endswith(".mcg"):
            result.add("mcg")
    return result


def _collect_int_k(value: Any, out: set[int]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int) and 1 <= value <= 16:
        out.add(value)
        return
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        parsed = int(value)
        if 1 <= parsed <= 16:
            out.add(parsed)
        return
    if isinstance(value, dict):
        for child in value.values():
            _collect_int_k(child, out)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_int_k(child, out)


def quantized_k_values(qcfg: dict[str, Any]) -> set[int]:
    values: set[int] = set()
    # Only inspect fields that represent actual discrete K selections. Average
    # bpw values such as 4.75 are intentionally ignored.
    bits = qcfg.get("bits")
    if isinstance(bits, int) and not isinstance(bits, bool):
        _collect_int_k(bits, values)
    for field in ("layer_bits", "tensor_bits", "non_routed_exl3"):
        if field in qcfg:
            _collect_int_k(qcfg[field], values)
    return values


def installed_exllamav3_architectures() -> tuple[set[str] | None, str | None]:
    try:
        from exllamav3.architecture.architectures import ARCHITECTURES
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if isinstance(ARCHITECTURES, dict):
        return {str(name) for name in ARCHITECTURES}, None
    return None, "ExLlamaV3 ARCHITECTURES registry was not a dict"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="local EXL3 checkpoint directory")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of the text report",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"missing {config_path}")

    config = load_json(config_path)
    qcfg = config.get("quantization_config")
    if not isinstance(qcfg, dict):
        qcfg = {}

    keys = weight_keys(model_dir)
    architecture = first_architecture(config)
    quant_method = str(qcfg.get("quant_method", "")).lower() or None
    codebooks = inferred_codebooks(qcfg, keys)
    k_values = quantized_k_values(qcfg)
    max_k = max(k_values) if k_values else None

    registry, registry_error = installed_exllamav3_architectures()
    arch_available = registry is not None and V41_ARCH in registry

    blockers: list[str] = []
    warnings: list[str] = []

    if architecture != V41_ARCH:
        blockers.append(
            f"checkpoint architecture is {architecture!r}, expected {V41_ARCH!r}"
        )
    if quant_method != "exl3":
        blockers.append(f"quant_method is {quant_method!r}, expected 'exl3'")

    if registry is None:
        blockers.append(
            "installed ExLlamaV3 architecture registry could not be inspected"
        )
    elif not arch_available:
        blockers.append(
            "installed ExLlamaV3 has no DeepseekV41ForCausalLM loader"
        )

    # Upstream ExLlamaV3 CPU MoE currently accepts mul1 experts, K <= 8, with
    # uniform expert-bias presence. We can prove the first two from metadata,
    # but bias uniformity is a runtime/layer-structure property.
    if not codebooks:
        blockers.append(
            "EXL3 codebook could not be proven from config/index metadata"
        )
    elif codebooks != {"mul1"}:
        blockers.append(
            "upstream ExLlamaV3 CPU-MoE requires mul1-only experts; "
            f"observed codebooks={sorted(codebooks)}"
        )

    if max_k is None:
        warnings.append(
            "maximum discrete K could not be proven; CPU-MoE requires K <= 8"
        )
    elif max_k > 8:
        blockers.append(
            f"maximum K={max_k}; upstream ExLlamaV3 CPU-MoE requires K <= 8"
        )

    warnings.append(
        "uniform per-expert bias presence is not proven by this metadata-only check"
    )
    warnings.append(
        "vllm-exl3 does not currently provide a host-resident CPU-MoE executor; "
        "its DeepSeek-V4.1 path relies on vLLM for the model architecture"
    )

    result = {
        "model_dir": str(model_dir),
        "architecture": architecture,
        "quant_method": quant_method,
        "codebooks": sorted(codebooks),
        "k_values": sorted(k_values),
        "max_k": max_k,
        "index_keys_seen": len(keys),
        "installed_exllamav3_v41_architecture": arch_available,
        "exllamav3_registry_error": registry_error,
        "cpu_moe_metadata_preflight": "blocked" if blockers else "conditional",
        "blockers": blockers,
        "warnings": warnings,
        "qualification_note": (
            "conditional means metadata prerequisites are not known to fail; "
            "it is not end-to-end qualification"
        ),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("DeepSeek-V4.1 EXL3 compatibility preflight")
        print(f"  checkpoint:   {model_dir}")
        print(f"  architecture: {architecture}")
        print(f"  quantization: {quant_method}")
        print(f"  codebooks:    {', '.join(sorted(codebooks)) or '<unknown>'}")
        print(f"  K values:     {sorted(k_values) if k_values else '<unknown>'}")
        print(f"  ExLlamaV3 V4.1 loader: {'yes' if arch_available else 'no'}")
        print(
            "  CPU-MoE preflight: "
            + ("BLOCKED" if blockers else "CONDITIONAL (not qualified)")
        )
        if blockers:
            print("\nBlockers:")
            for item in blockers:
                print(f"  - {item}")
        if warnings:
            print("\nWarnings:")
            for item in warnings:
                print(f"  - {item}")

    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
