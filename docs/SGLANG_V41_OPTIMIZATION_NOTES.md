# SGLang DeepSeek V4.1 optimization notes

Reference reviewed 2026-09-12:

- https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization?v=4

This recipe does **not** copy SGLang kernels. The article is used as an architecture/performance reference for experiments that remain independently implemented in the vLLM/EXL3 stack.

## Lessons adopted

1. **V4.1 cache accounting is not the old V4 43-layer MLA formula.** V4.1 has four persistent global-cache producers (layers 2, 8, 14, 20), with the first three pooled 2:1. The logical global KV+index floor is 890 bytes per original token. Actual backend allocation must still be measured before making a capacity claim.
2. **Verify actual kernel dispatch.** The article's largest plain-decode jump came from reaching the intended Blackwell MXFP8 path after fixing scale layout. `scripts/kernel_dispatch_receipt.sh` therefore records actual runtime/dispatch evidence instead of assuming a fast path from configuration.
3. **MoE TP deserves an A/B against EP.** SGLang reduced rank-wait skew with MoE TP4 after padding 576->640. For our two-Spark candidate, 2304/2=1152 is already exactly 128-aligned, so pure MoE TP2 is exposed as an experimental A/B with no width padding.
4. **Speculative verify is a separate kernel regime.** DSpark stays disabled during first correctness/capacity qualification. Mixed-K grouped/fused execution and verify-specific small-row optimization come later.
5. **Do not cargo-cult GB300 results onto Spark.** Their topology/performance numbers are hypotheses to test on GB10, not expected Spark throughput.

## Not yet implemented

- grouped/fused heterogeneous mixed-K MoE dispatch;
- CUDA-graph qualification for heterogeneous mixed-K;
- TP4 576->640 EXL3 padding;
- DSpark verify-specific mixed-K optimization.

Those remain follow-up projects after the TP2 EP2 and pure-MoE-TP2 correctness/capacity A/B is measured on real hardware.
