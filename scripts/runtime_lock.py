#!/usr/bin/env python3
"""Read and validate the recipe's single runtime contract lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "runtime.lock.json"


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime lock must be a JSON object")
    required = {
        "schema",
        "base_image",
        "vllm_exl3",
        "exllamav3",
        "native_abi_min",
        "torch_cuda_arch_list",
        "capabilities",
        "models",
        "first_boot",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"runtime lock missing keys: {missing}")
    if data["schema"] != "dsv41-exl3-runtime-lock.v1":
        raise ValueError(f"unsupported runtime lock schema: {data['schema']!r}")
    for key in ("vllm_exl3", "exllamav3"):
        commit = str(data[key].get("commit", ""))
        if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit.lower()):
            raise ValueError(f"{key}.commit must be a full 40-character git SHA")
    return data


def get_path(data: Any, dotted: str) -> Any:
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(dotted)
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        else:
            raise KeyError(dotted)
    return cur


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    get = sub.add_parser("get")
    get.add_argument("path")
    dump = sub.add_parser("dump")
    dump.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    data = load_lock()
    if args.cmd == "validate":
        print(f"LOCK_OK={LOCK_PATH}")
        return 0
    if args.cmd == "get":
        value = get_path(data, args.path)
        if value is None:
            return 3
        if isinstance(value, bool):
            print("1" if value else "0")
        elif isinstance(value, (dict, list)):
            print(json.dumps(value, separators=(",", ":")))
        else:
            print(value)
        return 0
    print(json.dumps(data, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
