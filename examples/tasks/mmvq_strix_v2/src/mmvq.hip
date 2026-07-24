// Standalone quantized matrix-vector multiply (Q4 -> F16) for Strix Halo
// This is the hot path for token generation in llama.cpp (mmvq)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>

static constexpr int QK4 = 32;
static constexpr int WARP_SIZE = 32;

typedef struct {
    __half d;
    uint8_t qs[16];
} q4_0_block;

// Baseline: simple 1-block-per-thread mmvq
__global__ void mmvq_q4_0_f16_baseline(
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

// === Round 1: Vectorize weight + activation loads (uint4 qs, half2 x) to widen memory transactions ===
__global__ void mmvq_q4_0_f16_r1(
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

        // (1) Single 128-bit load of the 16 packed nibble bytes.
        const uint4 qbytes = *reinterpret_cast<const uint4*>(blk.qs);
        const uint32_t qw[4] = { qbytes.x, qbytes.y, qbytes.z, qbytes.w };

        // (2) Vectorized activation load: 16 half2 = 32 __half in one sweep.
        const __half2* xh2 = reinterpret_cast<const __half2*>(x + b * 32);
        __half hx[32];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            const __half2 p = xh2[j];
            hx[2 * j]     = __low2half(p);
            hx[2 * j + 1] = __high2half(p);
        }

        float d = __half2float(blk.d);

        // Exact same arithmetic order as baseline (pair j, j+16).
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            const uint32_t word = qw[j >> 2];
            const uint8_t byte_j = static_cast<uint8_t>((word >> (8 * (j & 3))) & 0xFF);
            const uint8_t q0 = byte_j & 0x0F;
            const uint8_t q1 = static_cast<uint8_t>(byte_j >> 4);
            const float v0 = (q0 - 8.0f) * d;
            const float v1 = (q1 - 8.0f) * d;
            sum += v0 * __half2float(hx[j]);
            sum += v1 * __half2float(hx[j + 16]);
        }
    }

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

// === Round 2: Double-buffered weight prefetch + software pipelining to raise achieved weight bandwidth ===
__global__ void mmvq_q4_0_f16_r2(
    const q4_0_block* __restrict__ A,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int n, int k) {

    int lane = threadIdx.x;                 // 0..31 (wave32)
    int warp = threadIdx.y;                 // row offset within block
    const int QK = 32;                      // weights per q4_0 block
    const int stride = blockDim.x;          // 32 lanes per warp

    int row = blockIdx.x * blockDim.y + warp;
    if (row >= n) return;

    int nb = k / QK;
    const q4_0_block* row_blocks = A + row * nb;

    // Double-buffered weight staging: two ping-pong register slots so the
    // global load of the NEXT block overlaps the dequant/accumulate of the
    // CURRENT block (software pipelining).
    struct WBuf { uint4 qs; float d; };
    WBuf buf[2];
    int cur = 0, nxt = 1;

    // Preload the first block this lane owns into buf[cur].
    int b0 = lane;
    if (b0 < nb) {
        const q4_0_block* b0p = row_blocks + b0;
        buf[cur].qs = *reinterpret_cast<const uint4*>(b0p->qs);  // single 128-bit coalesced load
        buf[cur].d  = __half2float(b0p->d);
    }

    float sum = 0.0f;

    // Software-pipelined main loop: issue the next block's weight load UP FRONT
    // (before the current block's compute) so the memory pipe stays full.
    for (int b = b0; b < nb; b += stride) {
        int b_next = b + stride;

        // --- EARLY PREFETCH of next block's weights ---
        if (b_next < nb) {
            const q4_0_block* bnp = row_blocks + b_next;
            buf[nxt].qs = *reinterpret_cast<const uint4*>(bnp->qs);
            buf[nxt].d  = __half2float(bnp->d);
        }

        // --- COMPUTE current block from buf[cur] ---
        uint4 qs = buf[cur].qs;
        float d = buf[cur].d;
        int xbase = b * QK;

        // Unpack the 16-byte qs (4 x uint32 = 32 nibbles = 32 weights).
        // Each byte -> low nibble (vand) and high nibble (vshr) -> float2 of weights.
        uint32_t w32[4] = { qs.x, qs.y, qs.z, qs.w };
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t word = w32[w];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                uint32_t byte = (word >> (q * 8)) & 0xFFu;
                uint32_t lo = byte & 0x0Fu;            // vand: low nibble
                uint32_t hi = byte >> 4;               // vshr: high nibble
                int bi = w * 4 + q;                    // block byte index
                float v0 = ((float)lo - 8.0f) * d;     // dequant low nibble
                float v1 = ((float)hi - 8.0f) * d;     // dequant high nibble
                sum += v0 * __half2float(x[xbase + bi]);
                sum += v1 * __half2float(x[xbase + bi + 16]);
            }
        }

        // Ping-pong swap.
        int t = cur; cur = nxt; nxt = t;
    }

    // ---- Warp reduction (f32) via butterfly __shfl_xor ----
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset);
    }

    // ---- Cross-warp reduction via fixed shared memory ----
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (lane == 0 && warp == 0) {
        float total = 0.0f;
        for (int w = 0; w < blockDim.y; ++w) total += warp_sums[w];
        y[row] = __float2half(total);
    }
}
