# SM12x sparse-MLA page contract (DeepSeek V4.1 Flash TP4)

Minimal port of the SM12x page/indexer contract required for FlashInfer SM120
sparse-MLA decode with local heads=16 and topk=1152 on GB10.

## Contract

| knob | value |
|---|---|
| FlashInfer | 0.7.0rc1 (`07869c61`) with SM120 DSV4 runtime topk |
| Global vLLM `--block-size` | 128 |
| Compressed KV states/page | 64 |
| Indexer states/page | 64 (DeepGEMM `block_kv` ∈ {32,64}) |
| SWA backend page | 64 |

## Provenance

See `PROVENANCE.md`. Adapted from
`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark@fc725ec` (Apache-2.0;
Tech2Wild/Kai). Not a wholesale file replace of unrelated Engram/boot fixes.

## Mounts

Bind-mount the three Python files over the image-pinned V4.1 tree using
`mounts.txt` paths under site-packages `vllm/`.
