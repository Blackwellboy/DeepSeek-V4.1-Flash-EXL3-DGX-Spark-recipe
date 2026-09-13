# SPDX-License-Identifier: Apache-2.0
# Experimental disk-backed Engram storage for DGX Spark / GB10 qualification.
# Reuses vLLM Engram dequant semantics (fp8_e4m3fn + ue8m0 per-32 scales) from
# the image-pinned DeepSeek V4.1 runtime. Not copied from TonoKen3 source.
"""Disk-backed Engram table with bounded staging (no full-table residency)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger("vllm.engram_disk")

# Checkpoint key patterns after HF index (pre-mapper names).
_WEIGHT_KEY = "layers.{layer}.engram.embed.weight"
_SCALE_KEY = "layers.{layer}.engram.embed.scale"


def engram_disk_backed_enabled(engram_config=None) -> bool:
    env = os.environ.get("VLLM_ENGRAM_DISK_BACKED", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if engram_config is not None and getattr(engram_config, "disk_backed", False):
        return True
    return False


def should_skip_engram_embed_tensor(name: str) -> bool:
    """Skip full materialization of the huge embed tables when disk-backed."""
    if not engram_disk_backed_enabled():
        return False
    n = name.replace("model.", "")
    return n.endswith("engram.embed.weight") or n.endswith("engram.embed.scale")


def _resolve_model_dir(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    for key in ("VLLM_ENGRAM_MODEL_DIR", "MODEL_DIR", "MODEL"):
        val = os.environ.get(key, "").strip()
        if val and Path(val).exists():
            # MODEL may be the in-container mount `/models`
            p = Path(val)
            if (p / "model.safetensors.index.json").exists() or (
                p / "config.json"
            ).exists():
                return p.resolve()
    try:
        from vllm.config import get_current_vllm_config

        model = get_current_vllm_config().model_config.model
        p = Path(model)
        if p.exists():
            return p.resolve()
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError(
        "disk-backed Engram requires a local model directory "
        "(set VLLM_ENGRAM_MODEL_DIR or pass model_dir)"
    )


def _assert_local_nvme(path: Path) -> None:
    """Fail closed on NFS/USB/WiFi-style mounts for Engram backing."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["findmnt", "-no", "FSTYPE,SOURCE,TARGET", "-T", str(path)],
            text=True,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("findmnt failed for %s: %s", path, exc)
        return
    logger.info("Engram backing filesystem: %s for %s", out, path)
    fstype = out.split()[0].lower() if out else ""
    source = out.split()[1] if out and len(out.split()) > 1 else ""
    forbidden = ("nfs", "nfs4", "cifs", "smb", "fuse.sshfs", "fuse.rclone")
    if fstype in forbidden or source.startswith(("//", "nfs:")):
        raise RuntimeError(
            f"Engram disk-backed path must be node-local NVMe, not {out}"
        )
    # USB heuristic: removable block devices under /media or /mnt with usb
    resolved = str(path)
    if "/media/" in resolved or "/mnt/usb" in resolved or "usb" in source.lower():
        raise RuntimeError(
            f"Refusing USB/media Engram backing: {out} path={path}"
        )


def _fadvise_dontneed(fd: int, offset: int, length: int) -> None:
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


class DiskBackedEngramTable:
    """Node-local safetensors-backed Engram rows with bounded GPU staging."""

    def __init__(
        self,
        *,
        layer_id: int,
        vocab_start: int,
        vocab_end: int,
        dim: int,
        block_size: int = 32,
        model_dir: str | None = None,
        max_stage_rows: int = 8192,
        device: torch.device | None = None,
    ) -> None:
        self.layer_id = layer_id
        self.vocab_start = int(vocab_start)
        self.vocab_end = int(vocab_end)
        self.dim = int(dim)
        self.block_size = int(block_size)
        self.max_stage_rows = int(max_stage_rows)
        self.model_dir = _resolve_model_dir(model_dir)
        _assert_local_nvme(self.model_dir)

        index_path = self.model_dir / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        w_key = _WEIGHT_KEY.format(layer=layer_id)
        s_key = _SCALE_KEY.format(layer=layer_id)
        if w_key not in weight_map or s_key not in weight_map:
            raise KeyError(
                f"Engram keys missing from index: {w_key} / {s_key}"
            )
        self.weight_key = w_key
        self.scale_key = s_key
        self.weight_file = self.model_dir / weight_map[w_key]
        self.scale_file = self.model_dir / weight_map[s_key]
        if not self.weight_file.is_file() or not self.scale_file.is_file():
            raise FileNotFoundError(
                f"Engram shard files missing under {self.model_dir}"
            )

        self.device = device or torch.device("cuda", torch.cuda.current_device())
        # Bounded staging only — never the full table.
        self._host_w = torch.empty(
            self.max_stage_rows,
            self.dim,
            dtype=torch.float8_e4m3fn,
            device="cpu",
            pin_memory=True,
        )
        self._host_s = torch.empty(
            self.max_stage_rows,
            self.dim // self.block_size,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        self._gpu_w = torch.empty(
            self.max_stage_rows,
            self.dim,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        self._gpu_s = torch.empty(
            self.max_stage_rows,
            self.dim // self.block_size,
            dtype=torch.uint8,
            device=self.device,
        )
        self._handle_w = None
        self._handle_s = None
        self._slice_w = None
        self._slice_s = None
        self._fd_w = None
        self._fd_s = None
        self.rows_fetched_total = 0
        self.lookups_total = 0
        self._open()
        nbytes = (
            self.max_stage_rows
            * (self.dim + self.dim // self.block_size)
            * 2  # host+gpu
        )
        logger.info(
            "Disk-backed Engram layer=%s rows=[%s,%s) files=%s/%s "
            "stage_rows=%s staging_bytes≈%.2f MiB",
            layer_id,
            self.vocab_start,
            self.vocab_end,
            self.weight_file.name,
            self.scale_file.name,
            self.max_stage_rows,
            nbytes / 1024**2,
        )

    def _open(self) -> None:
        from safetensors import safe_open

        # Keep handles open for the process lifetime; pages are not forced
        # resident — we only touch requested row ranges.
        self._handle_w = safe_open(str(self.weight_file), framework="pt", device="cpu")
        self._handle_s = safe_open(str(self.scale_file), framework="pt", device="cpu")
        self._slice_w = self._handle_w.get_slice(self.weight_key)
        self._slice_s = self._handle_s.get_slice(self.scale_key)
        shape_w = tuple(self._slice_w.get_shape())
        shape_s = tuple(self._slice_s.get_shape())
        if shape_w[1] != self.dim:
            raise ValueError(f"weight dim mismatch {shape_w} vs dim={self.dim}")
        if shape_s[1] != self.dim // self.block_size:
            raise ValueError(f"scale dim mismatch {shape_s}")
        self._fd_w = os.open(str(self.weight_file), os.O_RDONLY)
        self._fd_s = os.open(str(self.scale_file), os.O_RDONLY)

    def close(self) -> None:
        for attr in ("_handle_w", "_handle_s"):
            h = getattr(self, attr, None)
            if h is not None:
                try:
                    h.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, attr, None)
        for fd_attr in ("_fd_w", "_fd_s"):
            fd = getattr(self, fd_attr, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, fd_attr, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def fetch_unique_local_rows(
        self, local_row_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch unique local rows into GPU staging.

        Returns (gpu_weight[:U], gpu_scale[:U], inverse) where inverse maps
        each input local_row_id position to [0, U).
        """
        if local_row_ids.numel() == 0:
            empty_w = self._gpu_w[:0]
            empty_s = self._gpu_s[:0]
            return empty_w, empty_s, local_row_ids.to(dtype=torch.long)

        flat = local_row_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        # Exclude padding / invalid (-1) from unique fetch set.
        valid_mask = flat >= 0
        if not bool(valid_mask.any()):
            empty_w = self._gpu_w[:0]
            empty_s = self._gpu_s[:0]
            return empty_w, empty_s, torch.zeros_like(flat, device=self.device)

        valid_ids = flat[valid_mask]
        unique_valid = torch.unique(valid_ids, sorted=True)
        u = int(unique_valid.numel())
        if u > self.max_stage_rows:
            raise RuntimeError(
                f"Engram staging overflow: need {u} unique rows, "
                f"max_stage_rows={self.max_stage_rows}"
            )

        # Map every flat position -> staged slot (or -1).
        # unique_valid[i] lives at staging slot i.
        inverse = torch.full((flat.numel(),), -1, dtype=torch.long)
        # Build a dict-like via searchsorted for valid entries.
        pos = torch.searchsorted(unique_valid, flat.clamp(min=0))
        in_range = valid_mask & (pos < u) & (unique_valid[pos.clamp(max=u - 1)] == flat)
        inverse[in_range] = pos[in_range]

        # Correctness-first: gather rows via safetensors slices.
        owned = self.vocab_end - self.vocab_start
        for i in range(u):
            row = int(unique_valid[i].item())
            if row < 0 or row >= owned:
                self._host_w[i].zero_()
                self._host_s[i].zero_()
                continue
            global_row = self.vocab_start + row
            w = self._slice_w[global_row : global_row + 1]
            s = self._slice_s[global_row : global_row + 1]
            if s.dtype == torch.float8_e8m0fnu:
                s = s.view(torch.uint8)
            self._host_w[i].copy_(w.view(torch.float8_e4m3fn).reshape(self.dim))
            self._host_s[i].copy_(s.reshape(self.dim // self.block_size))

        self._gpu_w[:u].copy_(self._host_w[:u], non_blocking=True)
        self._gpu_s[:u].copy_(self._host_s[:u], non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        # Page-cache policy: only the unique rows above were touched. We
        # intentionally do not posix_fadvise the whole 95 GiB shard (that would
        # thrash unrelated checkpoint pages). Repeated lookups stay bounded by
        # staging size; OS reclaim handles cold file pages under pressure.

        self.rows_fetched_total += u
        self.lookups_total += 1
        return self._gpu_w[:u], self._gpu_s[:u], inverse.to(device=self.device)
