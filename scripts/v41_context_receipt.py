#!/usr/bin/env python3
"""Print a DeepSeek V4.1 cache/capacity receipt from the pinned vllm-exl3 runtime."""
from __future__ import annotations

import argparse
import json

from vllm_exl3.v41_context import validate_v41_context_scaling


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--model-resident-gib", type=float, required=True)
    p.add_argument("--other-runtime-gib", type=float, default=0.0)
    p.add_argument("--total-mem-gib", type=float, default=128.0)
    p.add_argument("--mem-util", type=float, default=0.75)
    p.add_argument(
        "--backend-bytes-per-token",
        type=int,
        default=None,
        help="Measured backend cache allocation per token. Omit to emit a logical-lower-bound receipt only.",
    )
    args = p.parse_args()
    receipt = validate_v41_context_scaling(
        args.context,
        model_resident_gib=args.model_resident_gib,
        other_runtime_gib=args.other_runtime_gib,
        total_mem_gib=args.total_mem_gib,
        mem_util=args.mem_util,
        backend_bytes_per_token=args.backend_bytes_per_token,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["capacity_claim_allowed"]:
        print(
            "WARNING: logical 890 B/token is not a backend allocation measurement; "
            "do not use this receipt alone to claim a context size fits.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
