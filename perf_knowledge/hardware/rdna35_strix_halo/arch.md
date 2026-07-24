---
title: RDNA 3.5 / Strix Halo (gfx1151) - architecture overview
kind: hardware
gens: [gfx1151]
dtypes: [fp32, fp16, bf16, fp8_e4m3, fp8_e5m2, int8, int4]
regimes: [both]
updated: 2026-07-23
sources:
  - https://www.amd.com/en/products/processors/laptop/ryzen-ai-300-series.html
  - https://github.com/ROCm/ROCm documentation for gfx1151
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch.html
---

# RDNA 3.5 / Strix Halo (gfx1151) - architecture overview

> Target: **AMD Radeon 8060S Graphics** (Strix Halo APU), RDNA 3.5, ISA **gfx1151**.
> This is a consumer APU integrated GPU, NOT an Instinct datacenter card.
> Key differences from CDNA: unified memory (no HBM), wave32 default, smaller CU count,
> no XCD chiplets, no Infinity Cache, no MFMA matrix units (uses WMMA instead).

## TL;DR
> The Strix Halo iGPU is a **single monolithic GPU** with 40 CUs, wave32 default (wave64 optional),
> 128GB unified memory (8GB dedicated VRAM + 119GB GTT via system RAM), and **WMMA** matrix
> instructions (not MFMA). Launch **~320+ workgroups** to fill all CUs, use **wave32** by default
> for compute, and remember most inference kernels are **memory-bandwidth-bound** on the
> unified memory path (~256 GB/s peak, much lower than HBM3's 5.3 TB/s).

## The one-screen cheat sheet
| Fact | Value | Why it matters |
|---|---|---|
| Wavefront | **32 lanes** (default), 64 optional | RDNA defaults to wave32, unlike CDNA's wave64 |
| CUs | **40** (20 SIMDs reported by torch) | grid >= 320 workgroups to fill (8 waves/CU) |
| XCDs | **0** (monolithic) | no chiplet locality concerns |
| SIMD/CU | **2** (SIMD32) | occupancy is per-SIMD, different from CDNA's 4 |
| Wave slots | **~10/SIMD -> ~20/CU** | check rocminfo for exact values |
| Peak clock | **~2.6 GHz** (boost) | basis of all peak math |
| VGPR | **512 per SIMD**, 24-granule | occupancy killer, same pattern as CDNA |
| LDS | **64 KiB/CU**, 32 banks | bank conflicts mod 32 |
| L1 | **16 KiB/CU** (vector), 128 B line | smaller than CDNA's 32 KiB |
| L2 | **8 MiB** (shared across all CUs) | monolithic, no per-XCD partitioning |
| Infinity Cache | **none** (no MALL) | all misses go to unified memory |
| Memory | **128 GB unified** (8GB VRAM + 119GB GTT) | system RAM, NOT HBM3 |
| Memory bandwidth | **~256 GB/s** (LPDDR5X-8533 quad-channel) | ~20x slower than MI300X HBM3 |
| Matrix peak | FP16/BF16 ~50 TF (WMMA), INT8 ~100 TF | WMMA, not MFMA; different instruction set |
| TDP | **~120W** (total APU shared with CPU) | thermal/clock budget shared with CPU |

## Key differences from CDNA (MI300X/MI355X)

### 1. Wave32 default (CRITICAL)
RDNA 3.5 defaults to **wave32** (32 lanes/wavefront), NOT wave64 like CDNA.
This affects:
- All divergence and coalescing windows are 32-wide, not 64-wide
- Triton's `num_warps` maps to wave32 wavefronts (not wave64)
- Subgroup operations use 32-lane width
- To force wave64: `__attribute__((amdgpu_flat_work_group_size(64, 64)))` or compile with
  `-mwavefrontwidth=64` (but this is non-default and may not work on all RDNA 3.5 paths)

**Pitfall**: Writing/tuning as if wave64 (CDNA habit) - all lane math is **32-wide** on RDNA by default.
If your kernel was written for CDNA wave64 and uses wave64-dependent patterns (e.g., 64-lane shuffles),
it MUST be adapted for wave32 or explicitly forced to wave64.

### 2. WMMA instead of MFMA
RDNA 3.5 uses **WMMA** (Warp Matrix Multiply-Accumulate) instructions, not CDNA's MFMA.
- WMMA operates on 16x16 tiles (FP16/BF16) vs MFMA's 16x16/32x32 variants
- Different register allocation patterns (VGPRs, not AGPRs)
- Triton-AMD backend supports WMMA on RDNA
- HIP kernels need `wmma_f16_16x16x16` instructions, not `mfma_f16_16x16x16`

### 3. Unified memory (no HBM)
- 128GB total: 8GB dedicated VRAM + 119GB GTT (system RAM mapped to GPU)
- Bandwidth is ~256 GB/s (LPDDR5X), vs 5.3 TB/s on MI300X
- GTT allocations have higher latency than VRAM
- Memory-bound kernels will be **much slower** than on Instinct
- Use `HSA_ENABLE_SDMA=0` for small transfers (force CPU copy over SDMA)
- Large model weights spill into GTT automatically

### 4. No XCD chiplets
- Single monolithic GPU - no cross-chiplet locality concerns
- L2 cache is shared across all CUs (8 MiB total)
- No Infinity Cache / MALL
- Simpler memory hierarchy than CDNA

### 5. Smaller CU count
- 40 CUs vs 304 on MI300X
- Grid sizing: aim for ~320+ workgroups (8 waves/CU) instead of 1024+
- Decode (skinny GEMV) needs fewer SPLIT_K partitions to fill CUs
- Fewer CUs = less parallelism, but also less scheduling overhead

## Concepts

### Memory hierarchy
```
  [GPU CUs] -> [L1 vector cache 16 KiB/CU] -> [L2 cache 8 MiB shared] -> [Unified Memory 128GB]
                                                                      |
                                          [8GB dedicated VRAM (fast)] | [119GB GTT (system RAM, slow)]
```

### APU memory model
The Strix Halo APU shares system RAM between CPU and GPU:
- **Dedicated VRAM** (8GB): Fast, allocated first for GPU contexts
- **GTT** (119GB): System RAM pages mapped to GPU via GART, slower access
- The GPU driver automatically spills to GTT when VRAM is full
- For LLM inference: model weights >8GB will spill to GTT
- Use `--gpu-memory-utilization 0.85` (not 0.95) to leave headroom for system

### Build targets
```bash
# HIP / llama.cpp
cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_HIP_ARCHITECTURES=gfx1151

# PyTorch
export PYTORCH_ROCM_ARCH=gfx1151

# Triton-AMD
# Triton auto-detects from rocminfo, but can be pinned:
export TRITON_AMD_ARCH=gfx1151
```

## Verify
- `rocminfo | grep gfx` -> should show gfx1151
- `rocm-smi --showproductname` -> Card Series: AMD Radeon 8060S Graphics, SKU: STRXLGEN
- `python3 -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"` -> gfx1151
- `python3 -c "import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)"` -> 20 (SIMDs)
- `python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1024**3)"` -> ~124 GB
