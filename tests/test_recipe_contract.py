#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Import recipe helpers directly from scripts/ without external dependencies.
import sys
sys.path.insert(0, str(SCRIPTS))
from runtime_lock import load_lock  # noqa: E402
from validate_pack import validate_pack  # noqa: E402


class RecipeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = load_lock(ROOT / "runtime.lock.json")

    def test_dockerfile_defaults_match_lock(self) -> None:
        dockerfile = (ROOT / "Dockerfile.spark").read_text()

        def arg(name: str) -> str:
            match = re.search(rf"^ARG {re.escape(name)}=(.+)$", dockerfile, re.M)
            self.assertIsNotNone(match, f"missing Docker ARG {name}")
            return match.group(1).strip()

        self.assertEqual(arg("BASE_IMAGE"), self.lock["base_image"]["ref"])
        self.assertEqual(arg("VLLM_EXL3_REPO"), self.lock["vllm_exl3"]["repo"])
        self.assertEqual(arg("VLLM_EXL3_REF"), self.lock["vllm_exl3"]["commit"])
        self.assertEqual(arg("EXLLAMAV3_REF"), self.lock["exllamav3"]["commit"])
        self.assertEqual(arg("TORCH_CUDA_ARCH_LIST"), self.lock["torch_cuda_arch_list"])

    def test_first_boot_and_capability_contract(self) -> None:
        self.assertEqual(
            self.lock["first_boot"],
            {
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.75,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 1024,
                "text_only": True,
                "dspark": False,
                "eager": True,
                "native_moe": False,
            },
        )
        self.assertEqual(
            self.lock["capabilities"]["accepted_exl3_config_k"],
            [2, 3, 4, 5, 6, 7, 8],
        )
        self.assertFalse(
            self.lock["capabilities"]["tensor_level_mixed_k_within_layer"]
        )

    def test_build_wrapper_forwards_every_locked_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bindir = tmp / "bin"
            bindir.mkdir()
            log = tmp / "docker.log"
            docker = bindir / "docker"
            docker.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%q ' \"$@\" >> \"$DOCKER_LOG\"\n"
                "printf '\\\\n' >> \"$DOCKER_LOG\"\n"
                "if [[ \"$1\" == image && \"$2\" == inspect ]]; then "
                "echo 'ID=fake RepoDigests=[]'; fi\n"
                "exit 0\n"
            )
            docker.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{bindir}:{env['PATH']}"
            env["DOCKER_LOG"] = str(log)
            subprocess.run(
                ["bash", str(SCRIPTS / "build_runtime.sh")],
                cwd=ROOT,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            text = log.read_text()
            expected = [
                f"BASE_IMAGE={self.lock['base_image']['ref']}",
                f"VLLM_EXL3_REPO={self.lock['vllm_exl3']['repo']}",
                f"VLLM_EXL3_REF={self.lock['vllm_exl3']['commit']}",
                f"EXLLAMAV3_REF={self.lock['exllamav3']['commit']}",
                f"TORCH_CUDA_ARCH_LIST={self.lock['torch_cuda_arch_list']}",
            ]
            for item in expected:
                self.assertIn(item, text)

    @staticmethod
    def _base_config() -> dict:
        return {
            "architectures": ["DeepseekV41ForCausalLM"],
            "model_type": "deepseek_v41",
            "quantization_config": {
                "quant_method": "exl3",
                "codebook": "mcg",
                "mtp_experts": "source",
                "non_routed_quantization": {
                    "quant_method": "deepseek_v4_fp8",
                    "weight_block_size": [32, 32],
                    "activation_scheme": "dynamic",
                },
            },
        }

    @classmethod
    def _write_pack(cls, root: Path, tensors: dict[str, tuple[str, list[int]]]) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(cls._base_config()))
        shard = root / "model-00001-of-00001.safetensors"
        offset = 0
        header: dict[str, dict] = {}
        payload = bytearray()
        sizes = {"I16": 2, "F16": 2, "F32": 4}
        for name, (dtype, shape) in tensors.items():
            count = 1
            for dim in shape:
                count *= dim
            size = count * sizes[dtype]
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            payload.extend(bytes(size))
            offset += size
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * ((-len(raw)) % 8)
        shard.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: shard.name for name in tensors}})
        )
        return shard

    def test_validator_accepts_layer_uniform_k7(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            self._write_pack(
                root,
                {
                    "model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 112]),
                    "model.layers.0.ffn.experts.0.w2_trellis": ("I16", [1, 1, 112]),
                },
            )
            report = validate_pack(root, "tp2", 0)
            self.assertTrue(report["deployable_with_current_pinned_loader"], report)
            self.assertEqual(report["trellis_k_histogram"], {"7": 2})
            self.assertEqual(report["mixed_k_layers"], {})
            self.assertEqual(report["tensor_size_mismatches"], [])
            self.assertEqual(len(report["shard_header_sha256"]), 1)

    def test_validator_rejects_mixed_k_within_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            self._write_pack(
                root,
                {
                    "model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 32]),
                    "model.layers.0.ffn.experts.0.w2_trellis": ("I16", [1, 1, 48]),
                },
            )
            report = validate_pack(root, "tp2", 0)
            self.assertFalse(report["deployable_with_current_pinned_loader"])
            self.assertTrue(report["mixed_k_layers"])
            self.assertTrue(
                any("multiple physical K widths" in error for error in report["errors"])
            )

    def test_validator_rejects_pointer_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            shard = self._write_pack(
                root,
                {"model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 32])},
            )
            shard.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:" + "0" * 64 + "\nsize 999\n"
            )
            report = validate_pack(root, "tp2", 0)
            self.assertFalse(report["deployable_with_current_pinned_loader"])
            self.assertEqual(report["bad_shards"][0]["status"], "pointer")

    def test_validator_rejects_dtype_shape_byte_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pack"
            shard = self._write_pack(
                root,
                {"model.layers.0.ffn.experts.0.w1_trellis": ("I16", [1, 1, 32])},
            )
            raw = shard.read_bytes()
            header_len = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8 : 8 + header_len])
            key = next(iter(header))
            header[key]["data_offsets"] = [0, 32]
            new_header = json.dumps(header, separators=(",", ":")).encode()
            new_header += b" " * ((-len(new_header)) % 8)
            shard.write_bytes(struct.pack("<Q", len(new_header)) + new_header + bytes(64))
            report = validate_pack(root, "tp2", 0)
            self.assertFalse(report["deployable_with_current_pinned_loader"])
            self.assertTrue(report["tensor_size_mismatches"])

    def test_cluster_launcher_safety_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bindir = tmp / "bin"
            bindir.mkdir()
            log = tmp / "docker.log"
            docker = bindir / "docker"
            docker.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%q ' \"$@\" >> \"$DOCKER_LOG\"\n"
                "printf '\\\\n' >> \"$DOCKER_LOG\"\n"
                "if [[ \"$1\" == image && \"$2\" == inspect ]]; then exit 0; fi\n"
                "if [[ \"$1\" == container && \"$2\" == inspect ]]; then "
                "[[ \"${FAKE_CONTAINER_EXISTS:-0}\" == 1 ]] && exit 0 || exit 1; fi\n"
                "if [[ \"$1\" == run ]]; then echo fake-container-id; fi\n"
                "exit 0\n"
            )
            docker.chmod(0o755)
            sleep = bindir / "sleep"
            sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
            sleep.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{bindir}:{env['PATH']}"
            env["DOCKER_LOG"] = str(log)
            env["HEAD_IP"] = "10.0.0.1"
            env["NODE_IP"] = "10.0.0.1"
            rdma = tmp / "infiniband"
            rdma.touch()

            auto = env | {"RDMA_DEVICE": str(rdma), "ENABLE_RDMA": "auto"}
            subprocess.run(
                ["bash", str(SCRIPTS / "start_cluster.sh"), "head"],
                cwd=ROOT, env=auto, check=True, capture_output=True, text=True,
            )
            self.assertIn("--device", log.read_text())

            log.write_text("")
            disabled = env | {"RDMA_DEVICE": str(rdma), "ENABLE_RDMA": "0"}
            subprocess.run(
                ["bash", str(SCRIPTS / "start_cluster.sh"), "head"],
                cwd=ROOT, env=disabled, check=True, capture_output=True, text=True,
            )
            self.assertNotIn("--device", log.read_text())

            rdma.unlink()
            required = env | {"RDMA_DEVICE": str(rdma), "ENABLE_RDMA": "1"}
            proc = subprocess.run(
                ["bash", str(SCRIPTS / "start_cluster.sh"), "head"],
                cwd=ROOT, env=required, capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)

            existing = env | {"ENABLE_RDMA": "0", "FAKE_CONTAINER_EXISTS": "1"}
            proc = subprocess.run(
                ["bash", str(SCRIPTS / "start_cluster.sh"), "head"],
                cwd=ROOT, env=existing, capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("REPLACE_CONTAINER=1", proc.stderr)

    def test_no_stale_runtime_facts(self) -> None:
        # Split old SHAs so this test does not match its own source text.
        stale = [
            "8f4517e80416466fa4a3ad2eb28685021" + "d39e95f",
            "21fa627a3933d80de2d1030e732354d8" + "c3cd761e",
        ]
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix not in {".md", ".py", ".sh", ".json", ".yml", ".yaml", ""}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if path == Path(__file__):
                continue
            for value in stale:
                if value in text:
                    offenders.append(f"{path.relative_to(ROOT)}: stale plugin SHA {value}")
            if "128-expert envelope" in text:
                offenders.append(f"{path.relative_to(ROOT)}: stale total-expert ceiling wording")
            if "GPU_MEMORY_UTILIZATION:-0.90" in text:
                offenders.append(f"{path.relative_to(ROOT)}: stale 0.90 first-boot default")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_required_files_exist(self) -> None:
        required = [
            "runtime.lock.json",
            "docs/TP4.md",
            "docs/TP2.md",
            "docs/TROUBLESHOOTING.md",
            "scripts/runtime_lock.py",
            "scripts/validate_pack.py",
            "scripts/doctor.py",
            "scripts/doctor.sh",
            "scripts/materialize_model.sh",
            "scripts/tp4_min_fit.sh",
            "scripts/watch_ray_memory.py",
            "scripts/image_fingerprint.sh",
            "scripts/smoke_test.sh",
            "profiles/tp4-minfit.env",
            "profiles/tp4-64k.env",
            "profiles/tp2-minfit.env",
            "profiles/tp2-32k.env",
            "THIRD_PARTY_NOTICES.md",
            "AGENTS.md",
        ]
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
