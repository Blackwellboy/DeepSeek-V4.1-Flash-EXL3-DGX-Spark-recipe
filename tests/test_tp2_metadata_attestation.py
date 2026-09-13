from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import attested_pack  # noqa: E402


_METADATA_ERRORS = [
    "non_routed_quantization is missing",
    "non_routed_quantization.quant_method must be deepseek_v4_fp8",
    "non_routed_quantization.weight_block_size must be [32, 32]",
    "mtp_experts must be 'source'",
]


def _base_result() -> dict:
    return {
        "errors": list(_METADATA_ERRORS),
        "warnings": [],
        "strict_locked_snapshot": True,
        "index_tensor_count": 3,
        "materialized_shard_bytes": 1234,
        "trellis_k_histogram": {"2": 2, "8": 1},
        "codebook_marker_counts": {"mul1": 3},
        "shard_header_sha256": {"model-00001-of-00001.safetensors": "abc"},
        "deployable_with_current_pinned_loader": False,
    }


class MetadataAttestationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, dict]:
        config = b'{"quantization_config":{"quant_method":"exl3"}}\n'
        (root / "config.json").write_bytes(config)
        base = _base_result()
        attestation = {
            "model_repo": "repo/model",
            "revision": "deadbeef",
            "config_sha256": hashlib.sha256(config).hexdigest(),
            "index_tensor_count": base["index_tensor_count"],
            "materialized_shard_bytes": base["materialized_shard_bytes"],
            "trellis_k_histogram": base["trellis_k_histogram"],
            "codebook_marker_counts": base["codebook_marker_counts"],
            "shard_header_sha256": base["shard_header_sha256"],
            "runtime_hf_overrides": {
                "quantization_config": {
                    "quant_method": "exl3",
                    "non_routed_quantization": {
                        "quant_method": "deepseek_v4_fp8",
                        "activation_scheme": "dynamic",
                        "weight_block_size": [32, 32],
                    },
                    "mtp_experts": "source",
                }
            },
        }
        return base, attestation

    def test_exact_locked_snapshot_can_supply_runtime_only_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base, attestation = self._fixture(root)
            lock = {
                "models": {"tp2": {"repo_id": "repo/model", "revision": "deadbeef"}}
            }
            with (
                mock.patch.object(attested_pack, "validate_pack", return_value=base),
                mock.patch.object(
                    attested_pack, "load_runtime_metadata_attestation", return_value=attestation
                ),
                mock.patch.object(attested_pack, "load_lock", return_value=lock),
            ):
                result = attested_pack.validate_pack_with_locked_metadata(
                    root, "tp2", 32.0, strict_locked_snapshot=True
                )
            self.assertTrue(result["metadata_attestation_match"])
            self.assertEqual(result["metadata_contract_source"], "locked_snapshot_attestation")
            self.assertEqual(result["errors"], [])
            self.assertTrue(result["deployable_with_current_pinned_loader"])
            self.assertEqual(
                result["runtime_hf_overrides"]["quantization_config"]["mtp_experts"],
                "source",
            )
            # Attestation never mutates canonical model metadata.
            self.assertEqual(
                json.loads((root / "config.json").read_text())["quantization_config"],
                {"quant_method": "exl3"},
            )

    def test_changed_header_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base, attestation = self._fixture(root)
            base["shard_header_sha256"] = {
                "model-00001-of-00001.safetensors": "changed"
            }
            lock = {
                "models": {"tp2": {"repo_id": "repo/model", "revision": "deadbeef"}}
            }
            with (
                mock.patch.object(attested_pack, "validate_pack", return_value=base),
                mock.patch.object(
                    attested_pack, "load_runtime_metadata_attestation", return_value=attestation
                ),
                mock.patch.object(attested_pack, "load_lock", return_value=lock),
            ):
                result = attested_pack.validate_pack_with_locked_metadata(
                    root, "tp2", 32.0, strict_locked_snapshot=True
                )
            self.assertFalse(result["metadata_attestation_match"])
            self.assertFalse(result["deployable_with_current_pinned_loader"])
            self.assertTrue(
                any("shard_header_sha256" in x for x in result["metadata_attestation_mismatches"])
            )
            self.assertEqual(result["runtime_hf_overrides"], {})

    def test_attestation_requires_strict_locked_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base, attestation = self._fixture(root)
            base["strict_locked_snapshot"] = False
            with (
                mock.patch.object(attested_pack, "validate_pack", return_value=base),
                mock.patch.object(
                    attested_pack, "load_runtime_metadata_attestation", return_value=attestation
                ),
            ):
                result = attested_pack.validate_pack_with_locked_metadata(
                    root, "tp2", 32.0, strict_locked_snapshot=False
                )
            self.assertFalse(result["deployable_with_current_pinned_loader"])
            self.assertIn(
                "locked metadata attestation requires --strict-locked-snapshot",
                result["metadata_attestation_mismatches"],
            )

    def test_structural_error_is_never_removed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base, attestation = self._fixture(root)
            base["errors"].append("1 shard tensors are absent from model.safetensors.index.json")
            lock = {
                "models": {"tp2": {"repo_id": "repo/model", "revision": "deadbeef"}}
            }
            with (
                mock.patch.object(attested_pack, "validate_pack", return_value=base),
                mock.patch.object(
                    attested_pack, "load_runtime_metadata_attestation", return_value=attestation
                ),
                mock.patch.object(attested_pack, "load_lock", return_value=lock),
            ):
                result = attested_pack.validate_pack_with_locked_metadata(
                    root, "tp2", 32.0, strict_locked_snapshot=True
                )
            self.assertTrue(result["metadata_attestation_match"])
            self.assertEqual(
                result["errors"],
                ["1 shard tensors are absent from model.safetensors.index.json"],
            )
            self.assertFalse(result["deployable_with_current_pinned_loader"])

    def test_checked_in_attestation_is_bound_to_runtime_lock(self):
        lock = json.loads((ROOT / "runtime.lock.json").read_text(encoding="utf-8"))
        contract = lock["models"]["tp2"]
        att_path = ROOT / contract["runtime_metadata_attestation"]
        attestation = json.loads(att_path.read_text(encoding="utf-8"))
        self.assertEqual(attestation["model_repo"], contract["repo_id"])
        self.assertEqual(attestation["revision"], contract["revision"])
        self.assertEqual(len(attestation["shard_header_sha256"]), contract["expected_shards"])
        self.assertEqual(attestation["index_tensor_count"], 188245)
        self.assertEqual(attestation["materialized_shard_bytes"], 446440212472)
        qcfg = attestation["runtime_hf_overrides"]["quantization_config"]
        self.assertEqual(qcfg["quant_method"], "exl3")
        self.assertEqual(qcfg["mtp_experts"], "source")
        self.assertEqual(
            qcfg["non_routed_quantization"],
            {
                "quant_method": "deepseek_v4_fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
            },
        )


class RuntimeWiringTests(unittest.TestCase):
    def test_serve_uses_hf_overrides_as_one_array_argument(self):
        text = (SCRIPTS / "serve.sh").read_text(encoding="utf-8")
        self.assertIn('ARGS+=( --hf-overrides "$HF_OVERRIDES_JSON" )', text)
        self.assertIn("Runtime HF override:", text)

    def test_tp2_serve_requires_strict_attestation(self):
        text = (SCRIPTS / "serve_tp2.sh").read_text(encoding="utf-8")
        self.assertIn("--strict-locked-snapshot", text)
        self.assertIn("--print-hf-overrides", text)
        self.assertIn("HF_OVERRIDES_JSON", text)


if __name__ == "__main__":
    unittest.main()
