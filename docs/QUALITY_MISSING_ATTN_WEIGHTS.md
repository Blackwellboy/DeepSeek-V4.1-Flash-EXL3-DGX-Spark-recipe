# Quality: incomplete non-routed tensors in 4.75bpw pack

## Symptom

TP4 serves `/v1/models` but greedy output is finite deterministic garbage.
`max_tokens=1` already wrong (`"ican"` / id 13484) → MLA decode is not primary.

## Broken HF revision

```text
vcruz305/DSV4.1-Flash-EXL3-4.75bpw
BROKEN_HF_REVISION=e971fd559aed165dc9db20c75948b964cef81410
HF_PACK_CHANGED_SINCE_E971FD55=NO (still current as of 2026-09-14)
```

HF card claims `Status: complete` and that protected non-expert tensors
(attn / shared / DSpark / vision / head) were copied. The published
`model.safetensors.index.json` contradicts that for backbone layers.

## Static contract vs base `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a`

Required non-routed tensors (everything in base except routed experts): **1621**

| Family | Present in pack | Missing |
|---|---:|---:|
| ATTN (layers.*.attn* / attn_norm) | 0 | **603** |
| SHARED (layers.*.ffn.shared_experts) | 0 | **240** |
| DSPARK (hc_*) | 0 | **240** |
| GATE (ffn.gate.*) | 0 | **120** |
| NORMS_FFN (ffn_norm) | 0 | **40** |
| VISION / MTP / ENGRAM / TOP | present | 0 |

```text
MISSING_REQUIRED_FAMILIES=ATTN+SHARED+DSPARK+GATE+FFN_NORMS
UNIQUE_SOURCE_SHARDS=40 (~275 GiB download if naively pulled; extract-and-delete OK)
```

## Live CUDA corroboration

- EXL3 routed trellis parity: PASS (540/540)
- embed / head: PASS
- live `layers.*.attn.wq_b/wo_b` FP8 hashes **identical across layers** (uninit)

## Validator

```bash
python3 scripts/validate_model_weight_contract.py /path/to/pack \
  --base-index /path/to/DeepSeek-V4.1-Flash/model.safetensors.index.json
```

Broken pack → `FAIL MISSING_BACKBONE_ATTENTION ...`

## Repair direction

Supplemental native FP8/BF16 repair shards from base revision `dba1be0a`, merged
local index, provenance manifest. Do not requantize. See
`scripts/build_nonrouted_repair_pack.py`.
