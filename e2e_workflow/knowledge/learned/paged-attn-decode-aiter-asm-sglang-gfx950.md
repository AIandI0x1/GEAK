---
key: paged decode attention (fp8-KV, aiter asm ll4mi) · gfx950 · sglang
type: routing
confidence: ★★
effect: dominant head (54.2% GPU on Qwen3-14B-FP8 decode); live=non-editable aiter asm → op bake-off is server-flag delegated, so the op-level lever is a Tier-C author-Triton paged-decode; e2e transfer PENDING (author loop not yet gated)
last_seen: 2026-07-18
---
# sglang fp8-KV paged DECODE attention → live path is aiter's asm ll4mi kernel (non-editable) → author Triton
- lever: the live decode seam is `aiter.ops.attention:paged_attention_ragged` which, under fp8 KV,
  dispatches the aiter **asm** `paged_attention_ll4mi_QKV_mfma16` kernel (NHD paged layout, block_size=1,
  partition_size=256, per-tensor k/v scale=1.0). It is NON-editable asm, so there is NO op-level env/flag
  winner — the cross-backend swap (`--attention-backend {aiter,triton,ck}`) is a SERVER flag = Config
  Tuner's job. `op_bench.py:bench_attn` only validates the oracle (harness_suspect=false is EXPECTED, not a
  fault; produces NO isolated ms). Run the immutable `unittest.py` directly for the baseline bar.
- apply: Tier-C author, **Triton route=author** (no editable incumbent → mode=author, target_language=triton):
  a fresh paged fp8-KV decode kernel judged vs the frozen aiter-asm oracle. Must win/not-regress BOTH decode
  M-buckets {1,256} and stay HIP-graph-capturable (server replays decode under a HIP graph; the unittest
  enforces a capture/replay bundle). Knobs: BLOCK_N (kv tile), num_kv_splits/partition, num_warps {4,8},
  num_stages, waves_per_eu, matrix_instr_nonkdim {16}, fuse the per-load fp8 KV dequant into the dot.
- verify: greedy temp=0 e2e parity (bf16 argmax tie-break risk on a cross-impl swap; oracle tol=0.03).
  Amdahl: 54.2% head → a 1.2x kernel ≈ +10% e2e ceiling — large; worth the author budget.
- caution: decode is launch/HBM-bound (memory-bound op); a host-heavy Triton rewrite can win isolated yet
  net ~0 e2e — confirm at the e2e gate, and keep the steady-state call sync-free/compile-free for HIP-graph
  capture. ckProfiler absent on this image → CK instance-sweep + CK author unavailable (advisory only).
- source: test_results/Qwen3-14B-FP8_24h/.../geak/e2e_cycle0 (paged_attention_ll4mi_QKV_mfma16_decode), 2026-07-18
