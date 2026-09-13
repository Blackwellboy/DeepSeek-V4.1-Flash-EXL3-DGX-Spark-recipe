# Quality: missing backbone attention weights in 4.75bpw pack

## Symptom

TP4 serves `/v1/models` but greedy output is finite deterministic garbage.
`max_tokens=1` already wrong (`"ican"`), so sparse-MLA decode is not primary.

## Live CUDA proof (2026-09-13/14, 4× DGX Spark)

| Family | Result |
|---|---|
| EXL3 routed trellis (all ranks, mixed K) | PASS 540/540 |
| `embed_tokens` ↔ `embed.weight` (vocab TP) | PASS |
| `lm_head` ↔ `head.weight` (vocab TP) | PASS |
| Tokenizer `e971fd55` | MATCH |
| `layers.*.attn.*` in safetensors index | **0 keys** |
| Live attn FP8 hashes across layers | **identical** (uninit pattern) |

## Config vs artifact

```text
quantization_config.scope = deepseek_v41_routed_experts
non_routed_quantization.quant_method = deepseek_v4_fp8
```

Non-routed FP8 attn/shared tensors are expected but not present under `layers.*.attn.*`
(only `vision.*attn` and `mtp.*attn` appear). Pack README claims protected attn
tensors were copied — the published index contradicts that.

## Required fix (model pack / hybrid load)

1. Publish a complete pack including backbone `layers.*.attn.*` (+ shared experts) FP8 tables; or
2. Recipe hybrid loader: EXL3 routed experts from this pack + non-routed from
   `deepseek-ai/DeepSeek-V4.1-Flash` (or an FP8 non-routed sidecar).

Until then: `GREEDY_SMOKE=FAIL`, Sixcat blocked.
