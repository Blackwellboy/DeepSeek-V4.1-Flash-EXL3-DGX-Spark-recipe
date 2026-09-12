#!/usr/bin/env python3
"""Prove a Ray cluster can execute a real cross-node NCCL all-reduce.

Run from the head runtime container before loading model weights. One GPU task is
pinned to each selected Ray node, then all ranks initialize a temporary NCCL process
group and verify/timestamp repeated all-reduces.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
import socket
import time
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_gpus=1, num_cpus=0.05)
def _rank_collective(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    elements: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in Ray GPU task")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    init_method = f"tcp://{master_addr}:{master_port}"
    started = False
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=120),
        )
        started = True
        tensor = torch.full((elements,), float(rank + 1), dtype=torch.float32, device=device)
        expected = world_size * (world_size + 1) / 2.0

        for _ in range(warmup):
            tensor.fill_(float(rank + 1))
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        elapsed: list[float] = []
        for _ in range(iterations):
            tensor.fill_(float(rank + 1))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - t0)

        actual = float(tensor[0].item())
        if abs(actual - expected) > 1e-4:
            raise RuntimeError(f"all-reduce parity failed: expected {expected}, got {actual}")

        nbytes = tensor.numel() * tensor.element_size()
        avg = sum(elapsed) / len(elapsed)
        # Algorithm-independent payload rate: bytes of tensor reduced per wall second.
        payload_gbps = (nbytes / avg) / 1e9
        return {
            "rank": rank,
            "hostname": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "bytes": nbytes,
            "iterations": iterations,
            "latency_ms": [value * 1000.0 for value in elapsed],
            "avg_ms": avg * 1000.0,
            "payload_gbps": payload_gbps,
            "parity_value": actual,
        }
    finally:
        if started:
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--megabytes", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--master-port", type=int, default=29557)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.megabytes < 1 or args.iterations < 1 or args.warmup < 0:
        raise SystemExit("invalid collective size/iteration settings")

    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    alive = [node for node in ray.nodes() if node.get("Alive") and node.get("Resources", {}).get("GPU", 0) >= 1]
    driver_node_id = str(ray.get_runtime_context().get_node_id())
    head = next((node for node in alive if str(node.get("NodeID")) == driver_node_id), None)
    if head is None:
        raise SystemExit("could not identify the head/driver Ray node")
    others = [node for node in alive if node is not head]
    selected = [head] + others
    if len(selected) < args.tp:
        raise SystemExit(f"only {len(selected)} live GPU Ray nodes; TP{args.tp} needs {args.tp}")
    selected = selected[: args.tp]

    master_addr = os.environ.get("VLLM_HOST_IP") or str(head.get("NodeManagerAddress") or "")
    if not master_addr:
        raise SystemExit("could not determine rank-0/master address; set VLLM_HOST_IP")

    elements = args.megabytes * 1024 * 1024 // 4
    refs = []
    nodes: list[dict[str, Any]] = []
    for rank, node in enumerate(selected):
        node_id = str(node["NodeID"])
        nodes.append({
            "rank": rank,
            "node_id": node_id,
            "node_manager_address": node.get("NodeManagerAddress"),
        })
        refs.append(
            _rank_collective.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote(
                rank,
                args.tp,
                master_addr,
                args.master_port,
                elements,
                args.warmup,
                args.iterations,
            )
        )

    try:
        ranks = ray.get(refs, timeout=180)
    except Exception as exc:
        report = {
            "ok": False,
            "topology": f"tp{args.tp}",
            "master_addr": master_addr,
            "master_port": args.master_port,
            "nodes": nodes,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    report = {
        "ok": True,
        "topology": f"tp{args.tp}",
        "master_addr": master_addr,
        "master_port": args.master_port,
        "message_megabytes": args.megabytes,
        "nodes": nodes,
        "ranks": sorted(ranks, key=lambda item: item["rank"]),
        "slowest_avg_ms": max(float(item["avg_ms"]) for item in ranks),
        "minimum_payload_gbps": min(float(item["payload_gbps"]) for item in ranks),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"NCCL collective PASS: TP{args.tp}, {args.megabytes} MiB all-reduce")
        for item in report["ranks"]:
            print(
                f"  rank {item['rank']} {item['hostname']} {item['gpu']}: "
                f"avg={item['avg_ms']:.3f} ms payload={item['payload_gbps']:.3f} GB/s"
            )
        print(f"  slowest avg: {report['slowest_avg_ms']:.3f} ms")
        print(f"  minimum payload rate: {report['minimum_payload_gbps']:.3f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
