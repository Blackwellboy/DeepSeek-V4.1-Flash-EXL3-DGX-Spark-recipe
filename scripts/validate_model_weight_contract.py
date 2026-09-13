#!/usr/bin/env python3
"""Fail-closed weight-contract validator for DeepSeek V4.1 EXL3 packs.

When quantization_config.scope == deepseek_v41_routed_experts, the pack must
still contain all required *non-routed* native tensors (attn, shared experts,
norms, DSpark/hc_*, gates, embed/head, Engram, vision/MTP as applicable).

Usage:
  python3 scripts/validate_model_weight_contract.py /path/to/model_dir \
      [--base-index path/to/base_model.safetensors.index.json]

Exit 0 on PASS, 2 on FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def is_routed_expert(name: str) -> bool:
    return ".ffn.experts." in name and "shared_experts" not in name


def family(name: str) -> str:
    if name.startswith("vision."):
        return "VISION"
    if name.startswith("mtp."):
        return "MTP"
    if "engram" in name:
        return "ENGRAM"
    if name in ("embed.weight", "head.weight", "norm.weight") or name.startswith(
        ("aligner.", "image_")
    ):
        return "TOP"
    if ".attn." in name or "attn_norm" in name:
        return "ATTN"
    if "shared_experts" in name:
        return "SHARED"
    if "ffn_norm" in name:
        return "NORMS_FFN"
    if "hc_" in name:
        return "DSPARK"
    if ".ffn.gate." in name:
        return "GATE"
    return "OTHER"


def load_index(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    return dict(data["weight_map"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument(
        "--base-index",
        type=Path,
        default=None,
        help="Base DeepSeek-V4.1-Flash model.safetensors.index.json for required non-routed set",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    model_dir: Path = args.model_dir
    pack_index = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    if not pack_index.is_file() or not config_path.is_file():
        print("FAIL: missing model.safetensors.index.json or config.json", file=sys.stderr)
        return 2

    cfg = json.loads(config_path.read_text())
    qcfg = cfg.get("quantization_config") or {}
    pack = load_index(pack_index)

    report: dict = {
        "model_dir": str(model_dir),
        "quant_method": qcfg.get("quant_method"),
        "scope": qcfg.get("scope"),
        "non_routed_quantization": qcfg.get("non_routed_quantization"),
        "pack_n_tensors": len(pack),
    }

    # Always require embed/head if claimed by architecture
    hard_required = []
    for name in ("embed.weight", "head.weight", "norm.weight"):
        hard_required.append(name)

    missing_hard = [n for n in hard_required if n not in pack]
    report["missing_top"] = missing_hard

    base_index_path = args.base_index
    missing_by_family: dict[str, list[str]] = {}
    if base_index_path and base_index_path.is_file():
        base = load_index(base_index_path)
        required = [k for k in base if not is_routed_expert(k)]
        missing = [k for k in required if k not in pack]
        present = [k for k in required if k in pack]
        for k in missing:
            missing_by_family.setdefault(family(k), []).append(k)
        report.update(
            {
                "EXPECTED_REQUIRED_TENSORS": len(required),
                "PACK_PRESENT_REQUIRED_TENSORS": len(present),
                "PACK_MISSING_REQUIRED_TENSORS": len(missing),
                "missing_by_family_counts": {f: len(v) for f, v in sorted(missing_by_family.items())},
                "MISSING_ATTN": len(missing_by_family.get("ATTN", [])),
                "MISSING_SHARED": len(missing_by_family.get("SHARED", [])),
                "MISSING_NORMS": len(missing_by_family.get("NORMS_FFN", [])),
                "MISSING_DSPARK": len(missing_by_family.get("DSPARK", [])),
                "MISSING_GATE": len(missing_by_family.get("GATE", [])),
                "MISSING_OTHER": len(missing_by_family.get("OTHER", [])),
            }
        )
        # Heuristic without base: backbone attn keys
        backbone_attn = [k for k in pack if k.startswith("layers.") and ".attn." in k]
        report["pack_backbone_attn_keys"] = len(backbone_attn)
    else:
        # Fail closed if scope claims routed-only EXL3 but no backbone attn present
        backbone_attn = [k for k in pack if k.startswith("layers.") and (".attn." in k or "attn_norm" in k)]
        report["pack_backbone_attn_keys"] = len(backbone_attn)
        report["EXPECTED_REQUIRED_TENSORS"] = None
        report["PACK_MISSING_REQUIRED_TENSORS"] = None
        if qcfg.get("scope") == "deepseek_v41_routed_experts" and len(backbone_attn) == 0:
            missing_by_family["ATTN"] = ["<all backbone layers.*.attn.* — none in pack>"]
            report["MISSING_ATTN"] = "ALL"
            report["PACK_MISSING_REQUIRED_TENSORS"] = "UNKNOWN_WITHOUT_BASE_INDEX"

    # Routed EXL3 presence (trellis)
    trellis = [k for k in pack if k.endswith(".trellis") and ".ffn.experts." in k]
    report["pack_routed_trellis"] = len(trellis)

    failures: list[str] = []
    if missing_hard:
        failures.append("MISSING_TOP_LEVEL:" + ",".join(missing_hard))
    if report.get("MISSING_ATTN") in ("ALL",) or (
        isinstance(report.get("MISSING_ATTN"), int) and report["MISSING_ATTN"] > 0
    ):
        failures.append("MISSING_BACKBONE_ATTENTION")
    if isinstance(report.get("MISSING_SHARED"), int) and report["MISSING_SHARED"] > 0:
        failures.append("MISSING_SHARED_EXPERTS")
    if isinstance(report.get("MISSING_DSPARK"), int) and report["MISSING_DSPARK"] > 0:
        failures.append("MISSING_DSPARK")
    if isinstance(report.get("MISSING_NORMS"), int) and report["MISSING_NORMS"] > 0:
        failures.append("MISSING_FFN_NORMS")
    if isinstance(report.get("MISSING_GATE"), int) and report["MISSING_GATE"] > 0:
        failures.append("MISSING_FFN_GATES")
    if qcfg.get("scope") == "deepseek_v41_routed_experts" and report["pack_backbone_attn_keys"] == 0:
        if "MISSING_BACKBONE_ATTENTION" not in failures:
            failures.append("MISSING_BACKBONE_ATTENTION")

    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"

    text = json.dumps(report, indent=2)
    if args.json_out:
        args.json_out.write_text(text)
    print(text)
    if failures:
        print("FAIL", *failures)
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
