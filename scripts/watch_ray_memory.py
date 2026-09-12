#!/usr/bin/env python3
"""Continuously sample system + CUDA memory on every Ray node.

DGX Spark has one coherent 128 GB physical memory pool. System MemAvailable is
therefore the primary capacity signal; CUDA free/total is printed alongside it
for correlation, not treated as an independent pool.
"""

from __future__ import annotations

import argparse
import os
import socket
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as fh:
        for line in fh:
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if fields:
                out[key] = int(fields[0]) * 1024
    return out


@ray.remote(num_cpus=0.01)
def sample_node() -> dict[str, object]:
    mem = _meminfo()
    record: dict[str, object] = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "mem_total": mem.get("MemTotal", 0),
        "mem_available": mem.get("MemAvailable", 0),
        "swap_free": mem.get("SwapFree", 0),
    }
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            record["cuda_free"] = int(free)
            record["cuda_total"] = int(total)
    except Exception as exc:  # diagnostics must survive torch/driver issues
        record["cuda_error"] = repr(exc)
    return record


def gib(value: int | float | None) -> float:
    return float(value or 0) / (1024**3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")

    while True:
        nodes = [node for node in ray.nodes() if node.get("Alive")]
        rows = []
        for node in nodes:
            node_id = node["NodeID"]
            task = sample_node.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote()
            rows.append(ray.get(task))

        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{stamp}] alive_nodes={len(rows)}", flush=True)
        for row in sorted(rows, key=lambda item: str(item.get("host"))):
            line = (
                f"{row.get('host','?'):>20}  "
                f"MemAvailable={gib(row.get('mem_available')):6.2f} GiB / "
                f"{gib(row.get('mem_total')):6.2f} GiB  "
                f"SwapFree={gib(row.get('swap_free')):6.2f} GiB"
            )
            if "cuda_total" in row:
                line += (
                    f"  CUDA_free={gib(row.get('cuda_free')):6.2f} GiB / "
                    f"{gib(row.get('cuda_total')):6.2f} GiB"
                )
            elif "cuda_error" in row:
                line += f"  CUDA_error={row['cuda_error']}"
            print(line, flush=True)

        if args.once:
            return 0
        time.sleep(max(0.25, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
