---
key: fp8_a8w8_blockscale dense GEMM · gfx950 · sglang0.5.12 CK-bpreshuffle live path
type: lever
confidence: ★★★ (iso tune) · ★★★ (e2e: +8-15% verified by clean warm A/B @conc=256)
effect: iso 1.49-1.75× on prefill M≥4096 (cktile). Clean warm same-session A/B (2026-07-17, gfx950 cu_num=256): +8.35% @conc256/ISL1K, +15.0% @conc256/ISL8K — scales with prefill share. Re-confirmed 2026-07-18 (Qwen3-14B-FP8, e2e_cycle0): aiter --compare tuner aggregate 1.52× @M4096, 1.57× @M8192 (gate_up 34816×5120 @M8192 = 1.73×); decode M∈{1,256} & down/o_proj small-N ~1.0× (default optimal, auto-skipped). All tuned rows errRatio≤2e-4.
confirms: 7
last_seen: 2026-07-18
source: Qwen3-14B-FP8 / Qwen3.5-27B-FP8 TP=1 gfx950 cu_num=256 (MI355X)
status: active
---
> 🔴 **MANDATE — any CK `gemm_a8w8` tune on gfx950 MUST follow this skill.**
> On gfx950 (MI355X) serving FP8 a8w8 block-scale, the live GEMM IS CK `gemm_a8w8_blockscale_bpreshuffle`
> (**it is CK from the start — NOT a Triton→CK switch, you are only tuning the tile config**). If the served
> (M,N,K) is not in a shipped tuned CSV, aiter falls back to the **default (untuned)** config (server.log:
> `use default config`). This op is ≈84% of GPU compute → the #1 lever, and the tune is **env-only, ZERO HBM,
> no code patch**, so it is always worth doing.
> ⚠️ But whether it converts to e2e depends on the served M-regime → **do NOT KEEP on iso speedup alone; it
> must pass a clean e2e A/B.**

---

## PLAYBOOK — tune / test / deploy

**0. Confirm the live kernel.** server.log showing `use default config` = untuned bpreshuffle-CK.
(Why: in `fp8_utils.py`, hip≥7.2.0 → `_use_aiter_bpreshuffle_gfx95=True`; served (N,K) not in
`use_aiter_triton_gemm_w8a8_tuned_gfx950` list → `use_triton=False` → runs CK bpreshuffle.)

**1. Capture the real served shapes → write an untuned CSV** (header EXACTLY `gfx,cu_num,M,N,K`; for MI355X use `gfx950,cu_num=256`).
- (N,K) = model linear dims; Qwen3-14B TP=1: `(34816,5120) gate_up / (5120,17408) down / (7168,5120) qkv / (5120,5120) o_proj`.
- M = tokens per forward: decode {1,64,128,256} + prefill {1024,4096,...,16384}.
- **Best: grab the real M from a live run** → `export AITER_LOG_TUNED_CONFIG=1`, then
  `grep -oE 'shape is M:[0-9]+' server.log | sort -u`. Tune the exact M you observe so nothing pads back to default.

**2. Run the CK tuner** (aiter's own; no ckProfiler / hipblaslt-bench needed; wall ~35min/GPU):
```
cd /sgl-workspace/aiter
python csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i <untuned.csv> -o <tuned.csv> --preshuffle --libtype both --compare --mp 1
```
`--preshuffle` (server preshuffles weights) · `--libtype both` (ck for small-M / cktile for large-M) · `--compare` (untuned vs tuned, the trustworthy iso verdict).

**3. Verify the tuner output.** All rows errRatio ≤1e-4. Expected regime split: prefill M≥4096 → **1.49–1.75×**
(cktile `...intrawave`); decode M∈{1,64} and prefill M=1024 → within ±3% (default already optimal, tuner auto-skips).
⚠️ Ignore op_bench's plain-weight bpreshuffle race (rel_err ~48) = harness artifact.

**4. Deploy (no code overlay on gfx950 — live path is already bpreshuffle-CK).**
- Merge CSV: `cat <shipped bpreshuffle csv> <tuner-output rows> > merged.csv` (keep a single header).
- **(A) env (preferred, toggleable for A/B):** `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=<abs>/merged.csv`.
- **(B) persist:** merge tuned rows into `/sgl-workspace/aiter/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv`
  (auto-merged at import). ⚠️ Once written you can't get a clean baseline in the same env → for A/B use (A) and toggle by unsetting the env.

---

## 🧪 E2E A/B (MANDATORY BEFORE KEEP)

The big iso wins are only on large-M prefill; decode (M≤conc) is ≈1.0×. **The e2e gain ∝ the served prefill
share** → measure at YOUR real (conc, ISL, OSL); don't cherry-pick a prefill-heavy shape to inflate the number.

**Clean single-variable A/B (same warm server, the only difference between arms = the tuned CSV):**
- Pin GPU: `export ROCR_VISIBLE_DEVICES=<id>` (do NOT set HIP_VISIBLE_DEVICES); `export AITER_LOG_TUNED_CONFIG=1`.
- Server: `python -m sglang.launch_server --model-path <M> --tp 1 --mem-fraction-static 0.8
  --disable-radix-cache --context-length <ctx> --trust-remote-code` (ctx must fit ISL+OSL+headroom).
- Bench: `python -m sglang.bench_serving --backend sglang --model <M> --host 127.0.0.1 --port <P>
  --dataset-name random --random-input-len <ISL> --random-output-len 1024 --random-range-ratio 1.0
  --max-concurrency 256 --num-prompts 512` (run a num-prompts=64 warmup first and discard it).
- baseline arm: env UNSET → server.log `not found tuned config`. opt arm: env → merged.csv → server.log `... is tuned on cu_num=256 ...`.
- **Compare the output-tok/s RATIO within each shape** (don't compare absolute values across ISL).

**Verified results: ISL1K/OSL1K/conc256 = 8051→8723 (+8.35%); ISL8K/OSL1K/conc256 = 1734→1994 (+15.0%).**
Gain grows with prefill share (conc↑, ISL↑ → bigger; decode-heavy / high OSL → dilutes toward ~1.0×).

**Three false-verdict traps (guard against all):**
1. **M-regime mismatch:** wins only on large-M prefill → test at your real (conc,ISL,OSL); a big iso number need not convert.
2. **Cold-vs-warm baseline:** never compare a fresh warm tuned run against a baseline from another phase (cold / pre-JIT-warm / pre-rebuild); drift can reach ±15%.
3. **Baseline contamination:** aiter auto-merges tuned CSVs from its config dir → a leftover CSV silently makes the baseline arm tuned → flat A/B.

**VERDICT:** KEEP only if a same-session, warm, single-variable-toggle A/B at the real workload clears the threshold. Here +8.35%/+15.0% → **KEEP**.

> ⚠️ caution: the gfx942 sibling ships a Triton→CK fp8_utils OVERLAY — **do NOT copy it to gfx950**; here the live path is already CK.
> History: GEAK once reported `no_win` (understated), Hyperloom once reported `+16%` (cold-baseline artifact); the clean A/B settled the real gain at +8-15%.
