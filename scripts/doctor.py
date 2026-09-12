#!/usr/bin/env python3
"""Host/runtime readiness doctor for the DGX Spark recipe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

from runtime_lock import load_lock
from validate_pack import validate_pack


def run(*cmd: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def free_gib(path: Path) -> float | None:
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", ""))
    parser.add_argument("--model", default=os.environ.get("MODEL", ""))
    parser.add_argument("--reserve-gib", type=float, default=32.0)
    args = parser.parse_args()

    lock = load_lock()
    topology = f"tp{args.tp}"
    report: dict[str, Any] = {
        "schema": "dsv41-exl3-doctor.v1",
        "topology": topology,
        "runtime_lock": lock,
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]

    report["host"] = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if platform.machine() not in {"aarch64", "arm64"}:
        warnings.append(f"host architecture is {platform.machine()}; DGX Spark is aarch64")

    docker = run("docker", "version", "--format", "{{json .Server}}")
    report["docker_version"] = docker
    if not docker.get("ok"):
        errors.append("Docker server is not available")

    docker_info = run(
        "docker",
        "info",
        "--format",
        "{{json .DriverStatus}}|{{.Driver}}|{{json .Plugins.Network}}",
    )
    report["docker_info"] = docker_info

    gpu = run(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    )
    report["gpu"] = gpu
    if not gpu.get("ok"):
        errors.append("nvidia-smi is not available on the host")

    report["rdma"] = {
        "dev_infiniband_exists": Path("/dev/infiniband").exists(),
        "sys_class_infiniband_exists": Path("/sys/class/infiniband").exists(),
    }
    report["network"] = run("ip", "-br", "link")

    image = os.environ.get("IMAGE", "deepseek-v41-exl3:spark")
    report["image"] = run(
        "docker", "image", "inspect", image, "--format", "{{.Id}} {{json .RepoDigests}}"
    )
    if not report["image"].get("ok"):
        warnings.append(f"runtime image {image!r} is not built on this host yet")

    for label, path_text in {
        "cwd": os.getcwd(),
        "hf_home": os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")),
        "model_dir": args.model_dir,
    }.items():
        if not path_text:
            continue
        path = Path(path_text).expanduser()
        existing = path if path.exists() else path.parent
        report.setdefault("filesystems", {})[label] = {
            "path": str(path),
            "probe_path": str(existing),
            "free_gib": free_gib(existing),
        }

    local_model: Path | None = None
    if args.model and args.model.startswith("/models/") and args.model_dir:
        local_model = Path(args.model_dir).expanduser() / Path(args.model).name
    elif args.model and Path(args.model).expanduser().is_dir():
        local_model = Path(args.model).expanduser()

    if local_model is not None and local_model.is_dir():
        pack = validate_pack(local_model, topology, args.reserve_gib)
        report["physical_pack"] = pack
        if not pack["deployable_with_current_pinned_loader"]:
            errors.extend(f"pack: {item}" for item in pack["errors"])
    else:
        report["physical_pack"] = None
        warnings.append(
            "no materialized local model was provided; full shard/index/K compatibility cannot be proven"
        )

    container = os.environ.get("CONTAINER_NAME", "dsv41-exl3")
    running = run("docker", "ps", "--filter", f"name=^{container}$", "--format", "{{.Names}}")
    container_running = running.get("ok") and running.get("stdout") == container
    report["container_running"] = bool(container_running)
    if container_running:
        runtime = run(
            "docker",
            "exec",
            container,
            "python",
            "-c",
            (
                "import json,subprocess,vllm_exl3_c;"
                "h=lambda p:subprocess.check_output(['git','-C',p,'rev-parse','HEAD'],text=True).strip();"
                "print(json.dumps({'vllm_exl3':h('/opt/vllm-exl3'),'exllamav3':h('/opt/exllamav3'),"
                "'abi':int(getattr(vllm_exl3_c,'P2B_MOE_ABI_VERSION',0))}))"
            ),
        )
        report["container_runtime"] = runtime
        if runtime.get("ok"):
            try:
                ident = json.loads(str(runtime.get("stdout") or "{}"))
                if ident.get("vllm_exl3") != lock["vllm_exl3"]["commit"]:
                    errors.append("running container vllm-exl3 revision does not match runtime lock")
                if ident.get("exllamav3") != lock["exllamav3"]["commit"]:
                    errors.append("running container ExLlamaV3 revision does not match runtime lock")
                if int(ident.get("abi") or 0) < int(lock["native_abi_min"]):
                    errors.append("running container native ABI is below runtime lock minimum")
            except Exception as exc:
                errors.append(f"could not parse container runtime identity: {exc}")

    report["ok"] = not errors
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(f"DeepSeek V4.1 EXL3 doctor: {topology}")
        print(f"Host: {report['host']['hostname']} {report['host']['machine']}")
        print(f"Docker: {'OK' if docker.get('ok') else 'FAIL'}")
        print(f"GPU: {'OK' if gpu.get('ok') else 'FAIL'}")
        print(f"RDMA device: {'present' if report['rdma']['dev_infiniband_exists'] else 'not detected'}")
        if local_model is not None:
            print(f"Local model: {local_model}")
        for warning in warnings:
            print(f"WARNING: {warning}")
        for error in errors:
            print(f"ERROR: {error}")
        print("DOCTOR=" + ("PASS" if report["ok"] else "FAIL"))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
