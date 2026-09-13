from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_serve_supports_guarded_tp2_moe_tp_ab():
    text = (ROOT / "scripts" / "serve.sh").read_text()
    assert 'MOE_PARALLEL_MODE="${MOE_PARALLEL_MODE:-ep}"' in text
    assert '--enable-expert-parallel --enable-ep-weight-filter' in text
    assert 'ALLOW_EXPERIMENTAL_TP4_MOE_TP' in text
    assert '576->640' in text


def test_tp2_disk_engram_profile_and_launcher_exist():
    profile = (ROOT / "profiles" / "tp2-disk-engram.env").read_text()
    launcher = (ROOT / "scripts" / "tp2_disk_engram_min_fit.sh").read_text()
    assert "VLLM_ENGRAM_DISK_BACKED=1" in profile
    assert "MOE_PARALLEL_MODE=ep" in profile
    assert 'check_disk_engram_cluster.py 2' in launcher
    assert 'check_oom_guards.sh" 2' in launcher
    assert "MAX_MODEL_LEN=8192" in launcher
    assert "DSPARK=0" in launcher
    assert "EAGER=1" in launcher


def test_tp2_docs_do_not_claim_mixed_k_is_loader_incompatible():
    serve = (ROOT / "scripts" / "serve_tp2.sh").read_text()
    docs = (ROOT / "docs" / "TP2.md").read_text()
    stale = "mixed K2-K8 is incompatible with the pinned layer-uniform loader"
    assert stale not in serve
    assert stale not in docs
    assert "5120 × 1152" in docs
    assert "890 bytes" in docs


def test_context_and_kernel_receipt_helpers_are_checked_in():
    context = (ROOT / "scripts" / "v41_context_receipt.py").read_text()
    kernel = (ROOT / "scripts" / "kernel_dispatch_receipt.sh").read_text()
    assert "validate_v41_context_scaling" in context
    assert "capacity_claim_allowed" in context
    assert "runtime_diagnostics" in kernel
    assert "MXFP8" in kernel
