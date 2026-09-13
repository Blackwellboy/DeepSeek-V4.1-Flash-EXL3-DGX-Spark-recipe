from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiskEngramContractTests(unittest.TestCase):
    def test_derived_image_enforces_safe_loader(self) -> None:
        text = (ROOT / "Dockerfile.disk-engram").read_text()
        self.assertIn(
            "disk-backed Engram requires the per-tensor safetensors iterator",
            text,
        )
        self.assertIn('safetensors_load_strategy in {"eager", "torchao", "prefetch"}', text)
        self.assertIn(
            "should_skip_engram_embed_tensor('layers.1.engram.embed.weight') is True",
            text,
        )

    def test_disk_env_reaches_ray_and_vllm(self) -> None:
        cluster = (ROOT / "scripts/start_cluster.sh").read_text()
        serve = (ROOT / "scripts/serve.sh").read_text()
        for key in (
            "VLLM_ENGRAM_DISK_BACKED",
            "VLLM_ENGRAM_MODEL_DIR",
            "VLLM_EXL3_MODEL_DIR",
        ):
            self.assertIn(key, cluster)
            self.assertIn(key, serve)

    def test_all_node_probe_requires_exact_topology(self) -> None:
        text = (ROOT / "scripts/check_disk_engram_cluster.py").read_text()
        self.assertIn("if len(nodes) != expected:", text)
        self.assertIn("arena_prescan_guard_installed", text)
        self.assertIn("should_skip_engram_embed_tensor", text)

    def test_oom_guard_targets_one_exact_container(self) -> None:
        guard = (ROOT / "scripts/oom_guard.sh").read_text()
        wrapper = (ROOT / "scripts/watch_oom_guard.sh").read_text()
        self.assertIn("OOM_GUARD_CONTAINER_NAME", guard)
        self.assertIn('grep -Fx "$TARGET_CONTAINER"', guard)
        self.assertNotIn("grep -E '^(dsv41|deepseek-v41)'", guard)
        self.assertIn("OOM_GUARD_CONTAINER_NAME", wrapper)
        self.assertNotIn("OOM_GUARD_CONTAINER_MATCH", wrapper)

    def test_resident_engram_requires_explicit_retest(self) -> None:
        text = (ROOT / "scripts/tp4_min_fit.sh").read_text()
        self.assertIn("RESIDENT_ENGRAM_TP4=CAPACITY_FAIL", text)
        self.assertIn("ALLOW_RESIDENT_ENGRAM_RETEST", text)
        self.assertIn("tp4_disk_engram_min_fit.sh", text)

    def test_disk_first_boot_is_eager_correctness_profile(self) -> None:
        text = (ROOT / "scripts/tp4_disk_engram_min_fit.sh").read_text()
        for expected in (
            "MAX_MODEL_LEN=8192",
            "MAX_NUM_SEQS=1",
            "MAX_NUM_BATCHED_TOKENS=1024",
            "TEXT_ONLY=1",
            "DSPARK=0",
            "EAGER=1",
            "NATIVE_MOE=0",
        ):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
