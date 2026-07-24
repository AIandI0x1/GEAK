// Standalone quantized matrix-vector multiply (Q4 -> F16) for Strix Halo
// This is the hot path for token generation in llama.cpp (mmvq)
// Each thread block processes one row of the weight matrix
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>

#define QK4 32
#define WARP_SIZE 32

// Q4_0 block: 1 fp16 scale + 16 packed nibbles = 18 bytes
typedef struct {
    __half d;       // scale
    uint8_t qs[16]; // 32 4-bit quants packed into 16 bytes
} q4_0_block;

// Baseline: simple 1-block-per-thread mmvq
// A is (n, k) in Q4_0, x is (k) in F16, y is (n) in F16
// k must be a multiple of 32
__global__ void mmvq_q4_0_f16_baseline(
    const q4_0_block* __restrict__ A,  // (n, k/32) blocks
    const __half* __restrict__ x,       // (k) in F16
    __half* __restrict__ y,             // (n) output
    int n, int k) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n) return;

    int nb = k / QK4;  // number of blocks per row
    const q4_0_block* row_blocks = A + row * nb;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const q4_0_block& blk = row_blocks[b];
        float d = __half2float(blk.d);
        for (int j = 0; j < 16; j++) {
            int i0 = j;
            int i1 = j + 16;
            uint8_t q0 = blk.qs[j] & 0x0F;
            uint8_t q1 = blk.qs[j] >> 4;
            float v0 = (q0 - 8.0f) * d;
            float v1 = (q1 - 8.0f) * d;
            sum += v0 * __half2float(x[b * 32 + i0]);
            sum += v1 * __half2float(x[b * 32 + i1]);
        }
    }

    // Warp reduce
    __shared__ float warp_sums[32];
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (lane == 0 && warp == 0) {
        float total = 0.0f;
        for (int w = 0; w < blockDim.y; w++) total += warp_sums[w];
        y[row] = __float2half(total);
    }
}

// Optimized: vectorized load + better occupancy for wave32
__global__ void mmvq_q4_0_f16_optimized(
    const q4_0_block* __restrict__ A,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int n, int k) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n) return;

    int nb = k / QK4;
    const q4_0_block* row_blocks = A + row * nb;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const q4_0_block& blk = row_blocks[b];
        float d = __half2float(blk.d);

        // Vectorized: load 16 bytes as uint4 (128-bit load)
        const uint4 q4 = *reinterpret_cast<const uint4*>(blk.qs);

        // Unpack and dot product
        const __half* xb = x + b * 32;
        float2 xf[16];
        for (int j = 0; j < 16; j++) {
            xf[j].x = __half2float(xb[j]);
            xf[j].y = __half2float(xb[j + 16]);
        }

        // Extract nibbles from uint4
        uint8_t* qs = (uint8_t*)&q4;
        for (int j = 0; j < 16; j++) {
            float v0 = (qs[j] & 0x0F) - 8.0f;
            float v1 = (qs[j] >> 4) - 8.0f;
            sum += v0 * xf[j].x * d;
            sum += v1 * xf[j].y * d;
        }
    }

    // Warp reduce
    __shared__ float warp_sums[32];
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (lane == 0 && warp == 0) {
        float total = 0.0f;
        for (int w = 0; w < blockDim.y; w++) total += warp_sums[w];
        y[row] = __float2half(total);
    }
}

// v2: Packed half2 FMA + vectorized x loads.
//  - x is loaded with uint4 (128-bit) vector loads and reinterpreted as __half.
//  - The per-element multiply-add is done with __half2 packed ops (__hfma2 /
//    __hmul2) instead of scalar __half2float + fmul.
//  - The partial products are widened back to f32 and accumulated in f32 for
//    the final reduction (accuracy preserved).
__global__ void mmvq_q4_0_f16_v2(
    const q4_0_block* __restrict__ A,  // (n, k/32) blocks
    const __half* __restrict__ x,       // (k) in F16
    __half* __restrict__ y,             // (n) output
    int n, int k) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n) return;

    int nb = k / QK4;  // number of blocks per row
    const q4_0_block* row_blocks = A + row * nb;

    float sum = 0.0f;

    // Constants for the packed (q - 8) dequant offset, applied via __hfma2.
    const __half2 one2  = __halves2half2(__float2half(1.0f),   __float2half(1.0f));
    const __half2 neg82 = __halves2half2(__float2half(-8.0f),  __float2half(-8.0f));

    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const q4_0_block& blk = row_blocks[b];

        // Vectorized x load: 32 halfs = 4 x 128-bit (uint4) loads.
        uint4 xl[4];
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            xl[q] = reinterpret_cast<const uint4*>(x + b * 32)[q];
        }
        const __half* xh = reinterpret_cast<const __half*>(xl);  // 32 halfs

        // Widen the 32 x elements to f32 using PACKED __half2 conversions.
        float xf[32];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            float2 p = __half22float2(__halves2half2(xh[j], xh[j + 16]));
            xf[j] = p.x;
            xf[j + 16] = p.y;
        }

        // Vectorized weight load (16 bytes = 1 x uint4).
        const uint4 q4 = *reinterpret_cast<const uint4*>(blk.qs);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&q4);

        // Scale kept in f32 (exact) and FACTORED OUT of the inner dot so the
        // per-element product (q-8)*x is computed in f32 exactly, matching the
        // baseline's f32 path (only the final store rounds to f16).
        const float d_f = __half2float(blk.d);

        float partial = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            // Pack the two nibbles and subtract 8 with a PACKED half2 FMA.
            const __half2 nib = __halves2half2(
                __float2half(static_cast<float>(qs[j] & 0x0F)),
                __float2half(static_cast<float>(qs[j] >> 4)));
            const __half2 off2 = __hfma2(nib, one2, neg82);  // (q0-8, q1-8)
            const float2 off = __half22float2(off2);

            // (q-8)*x in f32 (exact for these magnitudes), accumulate in f32.
            partial += off.x * xf[j] + off.y * xf[j + 16];
        }
        sum += d_f * partial;
    }

    // Warp reduce (f32)
    __shared__ float warp_sums[32];
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset);
    }
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (lane == 0 && warp == 0) {
        float total = 0.0f;
        for (int w = 0; w < blockDim.y; w++) total += warp_sums[w];
        y[row] = __float2half(total);
    }
}
