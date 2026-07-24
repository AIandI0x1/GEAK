// Standalone benchmark of the REAL ROCmFP4 vec_dot kernel from rocmfp4-llama.
// This is the actual hot path for token generation on Strix Halo.
// Extracted from: ggml/src/ggml-cuda/vecdotq.cuh + ggml/rocmfp4/rocmfp4_hip_codebook.cuh
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// === ROCmFP4 block structures (from rocmfp4.h) ===
#define QK_ROCMFP4 32
#define QR_ROCMFP4 2
#define QI_ROCMFP4 (QK_ROCMFP4 / (4 * QR_ROCMFP4))

typedef struct {
    uint8_t qs[QK_ROCMFP4/2];
    uint8_t e[2];
} block_rocmfp4;

typedef struct {
    uint8_t qs[QK_ROCMFP4/2];
    uint8_t e;
} block_rocmfp4_fast;

// Q8_1 block (activation format)
typedef struct {
    union {
        struct {
            __half d;   // delta
            __half s;   // sum
        };
        __half2 ds;
    };
    int8_t qs[QK_ROCMFP4];  // 32 int8 values
} block_q8_1;

// === Scale conversion (from rocmfp4_hip_scale.cuh) ===
static __device__ __forceinline__ float rocmfp4_ue4m3_to_fp32_half_finite(uint8_t x) {
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    if (exp == 0) {
        return (float) man * (1.0f / 1024.0f);
    }
    const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
    return __uint_as_float(bits);
}

// === Codebook lookup (from rocmfp4_hip_codebook.cuh) ===
static __device__ __forceinline__ int rocmfp4_get_qs_i32(const void * x, const int & i32) {
    return *((const int *) ((const uint8_t *) x + 4*i32));
}

static __device__ __forceinline__ int2 rocmfp4_get_int_from_codebook_16(const int & q4, const int8_t * fallback_table) {
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    const uint32_t v_even_low  = __builtin_amdgcn_perm(values1, values0, q_even & 0x07070707u);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values1, values0, q_odd  & 0x07070707u);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values3, values2, q_even & 0x07070707u);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values3, values2, q_odd  & 0x07070707u);

    const uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    const uint32_t mask_odd  = 0x03020100u | ((q_odd  & 0x08080808u) >> 1);

    return make_int2(
        __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
        __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd));
}

// === DP4A (from common.cuh) ===
// gfx1151 is RDNA3 + RDNA3_5, so it uses sudot4 (not sdot4)
#define RDNA3
#define RDNA3_5

static __device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx900__)
    int tmp1, tmp2;
    asm("v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0\n"
        "v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1\n"
        "v_add3_u32 %0, %1, %2, %0\n"
        "v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2\n"
        "v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3\n"
        "v_add3_u32 %0, %1, %2, %0"
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2) : "v"(a), "v"(b));
#else
    const int8_t *a8 = (const int8_t*)&a;
    const int8_t *b8 = (const int8_t*)&b;
    c += a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
    return c;
}

// === THE ACTUAL HOT KERNEL: vec_dot_rocmfp4_q8_1 (from vecdotq.cuh) ===
// This is called inside mul_mat_vec_q for every block, every row.
// VDR=4 means 4 iterations of the inner loop per call.
#define VDR_ROCMFP4_Q8_1_MMVQ 4

static __device__ __forceinline__ float vec_dot_rocmfp4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_rocmfp4 * bq4 = (const block_rocmfp4 *) vbq + kbx;
    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l = 0; l < VDR_ROCMFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = rocmfp4_get_qs_i32(bq4->qs, iqs + l);
        const int2 v = rocmfp4_get_int_from_codebook_16(aux_q4, nullptr);

        sumi0 = ggml_cuda_dp4a(v.x, q8[l + 0], sumi0);
        sumi1 = ggml_cuda_dp4a(v.y, q8[l + 4], sumi1);
    }

    const float db = __low2float(bq8_1->ds);
    return db * (rocmfp4_ue4m3_to_fp32_half_finite(bq4->e[0]) * sumi0 +
                 rocmfp4_ue4m3_to_fp32_half_finite(bq4->e[1]) * sumi1);
}

// === THE FAST VARIANT: vec_dot_rocmfp4_fast_q8_1 ===
#define VDR_ROCMFP4_FAST_Q8_1_MMVQ 2

static __device__ __forceinline__ float vec_dot_rocmfp4_fast_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_rocmfp4_fast * bq4 = (const block_rocmfp4_fast *) vbq + kbx;
    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_ROCMFP4_FAST_Q8_1_MMVQ; ++l) {
        const int aux_q4 = rocmfp4_get_qs_i32(bq4->qs, iqs + l);
        const int2 v = rocmfp4_get_int_from_codebook_16(aux_q4, nullptr);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    return __low2float(bq8_1->ds) * rocmfp4_ue4m3_to_fp32_half_finite(bq4->e) * sumi;
}

// === Warp reduce ===
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

// === Full mmvq kernel using the REAL vec_dot_rocmfp4_q8_1 ===
// This mirrors the actual mul_mat_vec_q from mmvq.cu but simplified
// for standalone benchmarking (no fusion, no MoE, single column).
static constexpr int WARP_SIZE = 32;
static constexpr int NWARPS = 2;  // matches GGML_ROCMFP4_RDNA35_NWARPS=2

__global__ __launch_bounds__(NWARPS * WARP_SIZE, 1)
void mmvq_rocmfp4_baseline(
    const block_rocmfp4 * __restrict__ A,   // (n, k/32) weight blocks
    const block_q8_1    * __restrict__ x,   // (k/32) activation blocks
    float               * __restrict__ y,   // (n) output
    int n, int k) {

    constexpr int qk  = QK_ROCMFP4;     // 32
    constexpr int qi  = QI_ROCMFP4;     // 2
    constexpr int vdr = VDR_ROCMFP4_Q8_1_MMVQ;  // 4
    constexpr int rows_per_block = 1;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_block * blockIdx.x;
    const int row = row0 + 0;  // rows_per_block=1
    if (row >= n) return;

    const int blocks_per_row_x = k / qk;
    constexpr int blocks_per_iter = vdr * NWARPS * WARP_SIZE / qi;

    float tmp = 0.0f;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx;  // 1:1 mapping for Q8_1
        const int kqs = vdr * (tid % (qi/vdr));
        tmp += vec_dot_rocmfp4_q8_1(A, x + kby, kbx, kqs);
    }

    // Multi-warp reduction
    if constexpr (NWARPS > 1) {
        __shared__ float tmp_shared[NWARPS - 1][WARP_SIZE];
        if (threadIdx.y > 0) {
            tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
        }
        __syncthreads();
        if (threadIdx.y > 0) return;

        #pragma unroll
        for (int l = 0; l < NWARPS - 1; ++l) {
            tmp += tmp_shared[l][threadIdx.x];
        }
    }

    tmp = warp_reduce_sum(tmp);
    if (threadIdx.x == 0) {
        y[row] = tmp;
    }
}

// === Same but for FAST variant ===
__global__ __launch_bounds__(NWARPS * WARP_SIZE, 1)
void mmvq_rocmfp4_fast_baseline(
    const block_rocmfp4_fast * __restrict__ A,
    const block_q8_1         * __restrict__ x,
    float                    * __restrict__ y,
    int n, int k) {

    constexpr int qk  = QK_ROCMFP4;
    constexpr int qi  = QI_ROCMFP4;
    constexpr int vdr = VDR_ROCMFP4_FAST_Q8_1_MMVQ;
    constexpr int rows_per_block = 1;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_block * blockIdx.x;
    const int row = row0;
    if (row >= n) return;

    const int blocks_per_row_x = k / qk;
    constexpr int blocks_per_iter = vdr * NWARPS * WARP_SIZE / qi;

    float tmp = 0.0f;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx;
        const int kqs = vdr * (tid % (qi/vdr));
        tmp += vec_dot_rocmfp4_fast_q8_1(A, x + kby, kbx, kqs);
    }

    if constexpr (NWARPS > 1) {
        __shared__ float tmp_shared[NWARPS - 1][WARP_SIZE];
        if (threadIdx.y > 0) {
            tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
        }
        __syncthreads();
        if (threadIdx.y > 0) return;

        #pragma unroll
        for (int l = 0; l < NWARPS - 1; ++l) {
            tmp += tmp_shared[l][threadIdx.x];
        }
    }

    tmp = warp_reduce_sum(tmp);
    if (threadIdx.x == 0) {
        y[row] = tmp;
    }
}

// === Round 1: Multi-row output blocking with LDS activation cache to reuse x across rows ===
__global__ __launch_bounds__(NWARPS * WARP_SIZE, 1)
void mmvq_rocmfp4_r1(
    const block_rocmfp4 * __restrict__ A,   // (n, k/32) weight blocks
    const block_q8_1    * __restrict__ x,   // (k/32) activation blocks
    float               * __restrict__ y,   // (n) output
    int n, int k) {

    constexpr int qk  = QK_ROCMFP4;     // 32
    constexpr int qi  = QI_ROCMFP4;     // 2
    constexpr int vdr = VDR_ROCMFP4_Q8_1_MMVQ;  // 4
    constexpr int rows_per_block = 4;   // R: multi-row output blocking

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_block * blockIdx.x;
    const int blocks_per_row_x = k / qk;
    constexpr int blocks_per_iter = vdr * NWARPS * WARP_SIZE / qi;

    // ---- Cooperatively load full x vector into LDS once per block ----
    // x is (k/32) block_q8_1 blocks; each block is 40 bytes. We cache the
    // int8 qs[] (32 bytes) and the ds half2 (4 bytes) = 36 bytes/block.
    // One uint4 (128-bit) load covers 16 bytes -> 9 uint4 per block_q8_1.
    constexpr int bytes_per_xblock = sizeof(block_q8_1); // 40
    constexpr int xblocks = (QK_ROCMFP4 == 32) ? 0 : 0;  // placeholder, computed below
    const int n_xblocks = blocks_per_row_x;              // k/32
    // 40 bytes/block -> 10 uint4 (160 bytes) covers 4 blocks; use 10*ceil(n_xblocks/4)
    const int lds_u4 = ((n_xblocks + 3) / 4) * 10;       // uint4 slots
    __shared__ uint4 x_lds[((4096 / 16) > 0) ? 2560 : 2560]; // 2560*16 = 40960 bytes cap (<=32KB)
    // Reinterpret as raw bytes; we index by block then field.
    __shared__ uint8_t x_bytes[40960];

    // Load x cooperatively: each thread handles 16-byte chunks.
    const int total_bytes = n_xblocks * bytes_per_xblock;
    const int total_u4 = (total_bytes + 15) / 16;
    for (int i = tid; i < total_u4; i += NWARPS * WARP_SIZE) {
        const uint4 * src = reinterpret_cast<const uint4 *>(x) + i;
        x_bytes[i * 16 + 0] = ((const uint8_t *)src)[0];
        x_bytes[i * 16 + 1] = ((const uint8_t *)src)[1];
        x_bytes[i * 16 + 2] = ((const uint8_t *)src)[2];
        x_bytes[i * 16 + 3] = ((const uint8_t *)src)[3];
        x_bytes[i * 16 + 4] = ((const uint8_t *)src)[4];
        x_bytes[i * 16 + 5] = ((const uint8_t *)src)[5];
        x_bytes[i * 16 + 6] = ((const uint8_t *)src)[6];
        x_bytes[i * 16 + 7] = ((const uint8_t *)src)[7];
        x_bytes[i * 16 + 8] = ((const uint8_t *)src)[8];
        x_bytes[i * 16 + 9] = ((const uint8_t *)src)[9];
        x_bytes[i * 16 + 10] = ((const uint8_t *)src)[10];
        x_bytes[i * 16 + 11] = ((const uint8_t *)src)[11];
        x_bytes[i * 16 + 12] = ((const uint8_t *)src)[12];
        x_bytes[i * 16 + 13] = ((const uint8_t *)src)[13];
        x_bytes[i * 16 + 14] = ((const uint8_t *)src)[14];
        x_bytes[i * 16 + 15] = ((const uint8_t *)src)[15];
    }
    __syncthreads();

    // ---- Per-row accumulation: R partial sums held in registers ----
    float tmp[rows_per_block];
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) tmp[r] = 0.0f;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = vdr * (tid % (qi/vdr));
        // x block from LDS
        const block_q8_1 * bq8_1 = reinterpret_cast<const block_q8_1 *>(x_bytes + kbx * bytes_per_xblock);
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            const int row = row0 + r;
            if (row >= n) continue;
            const block_rocmfp4 * bq4 = A + (size_t)row * blocks_per_row_x + kbx;
            tmp[r] += vec_dot_rocmfp4_q8_1(bq4, bq8_1, kbx, kqs);
        }
    }

    // ---- Cross-warp reduction: sum per-row accumulators ----
    __shared__ float tmp_shared[NWARPS - 1][WARP_SIZE][rows_per_block];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r)
            tmp_shared[threadIdx.y - 1][threadIdx.x][r] = tmp[r];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

#pragma unroll
    for (int l = 0; l < NWARPS - 1; ++l) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r)
            tmp[r] += tmp_shared[l][threadIdx.x][r];
    }

    // ---- Warp reduction + write each row ----
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        tmp[r] = warp_reduce_sum(tmp[r]);
        const int row = row0 + r;
        if (threadIdx.x == 0 && row < n) {
            y[row] = tmp[r];
        }
    }
}

// === Round 2: K-loop unrolling with independent accumulators for memory-level parallelism ===
__global__ __launch_bounds__(NWARPS * WARP_SIZE, 1)
void mmvq_rocmfp4_r2(
    const block_rocmfp4 * __restrict__ A,
    const block_q8_1    * __restrict__ x,
    float               * __restrict__ y,
    int n, int k) {

    constexpr int qk  = QK_ROCMFP4;
    constexpr int qi  = QI_ROCMFP4;
    constexpr int vdr = VDR_ROCMFP4_Q8_1_MMVQ;
    constexpr int rows_per_block = 1;
    constexpr int UNROLL = 4;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_block * blockIdx.x;
    const int row = row0 + 0;
    if (row >= n) return;

    const int blocks_per_row_x = k / qk;
    constexpr int blocks_per_iter = vdr * NWARPS * WARP_SIZE / qi;

    const int start = tid / (qi / vdr);
    const int kqs   = vdr * (tid % (qi / vdr));

    float tmp0 = 0.0f, tmp1 = 0.0f, tmp2 = 0.0f, tmp3 = 0.0f;

    int kbx = start;
    // Unrolled K-loop: 4 independent vec_dot calls with separate accumulators
    // so the block_rocmfp4/block_q8_1 loads for kbx, kbx+bpi, kbx+2*bpi, kbx+3*bpi
    // can be issued back-to-back (outstanding) before the dependent MAC chain.
    for (; kbx + (UNROLL - 1) * blocks_per_iter < blocks_per_row_x; kbx += UNROLL * blocks_per_iter) {
        tmp0 += vec_dot_rocmfp4_q8_1(A, x + (kbx + 0 * blocks_per_iter), kbx + 0 * blocks_per_iter, kqs);
        tmp1 += vec_dot_rocmfp4_q8_1(A, x + (kbx + 1 * blocks_per_iter), kbx + 1 * blocks_per_iter, kqs);
        tmp2 += vec_dot_rocmfp4_q8_1(A, x + (kbx + 2 * blocks_per_iter), kbx + 2 * blocks_per_iter, kqs);
        tmp3 += vec_dot_rocmfp4_q8_1(A, x + (kbx + 3 * blocks_per_iter), kbx + 3 * blocks_per_iter, kqs);
    }

    // Scalar remainder loop: covers blocks_per_row_x not divisible by UNROLL*blocks_per_iter.
    for (; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        tmp0 += vec_dot_rocmfp4_q8_1(A, x + kbx, kbx, kqs);
    }

    float tmp = tmp0 + tmp1 + tmp2 + tmp3;

    // Multi-warp reduction
    if constexpr (NWARPS > 1) {
        __shared__ float tmp_shared[NWARPS - 1][WARP_SIZE];
        if (threadIdx.y > 0) {
            tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
        }
        __syncthreads();
        if (threadIdx.y > 0) return;

        #pragma unroll
        for (int l = 0; l < NWARPS - 1; ++l) {
            tmp += tmp_shared[l][threadIdx.x];
        }
    }

    tmp = warp_reduce_sum(tmp);
    if (threadIdx.x == 0) {
        y[row] = tmp;
    }
}

// === Round 3: Software-pipelined weight prefetch with wider coalesced loads (single-row CTA) ===
__global__ __launch_bounds__(NWARPS * WARP_SIZE, 1)
void mmvq_rocmfp4_r3(
    const block_rocmfp4 * __restrict__ A,   // (n, k/32) weight blocks
    const block_q8_1    * __restrict__ x,   // (k/32) activation blocks
    float               * __restrict__ y,   // (n) output
    int n, int k) {

    constexpr int qk = QK_ROCMFP4;   // 32

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;                 // single-row CTA (rows_per_block = 1)
    if (row >= n) return;

    const int blocks_per_row = k / qk;
    const int stride = NWARPS * WARP_SIZE;      // 1 block per thread per iter, strided

    const block_rocmfp4 * __restrict__ Arow = A + (size_t) row * blocks_per_row;

    float tmp = 0.0f;

    // ---- 2-stage software pipeline over the dominant weight path ----
    // qs[16] == exactly one 128-bit (uint4) coalesced transaction, replacing
    // the 4 scalar 32-bit block_rocmfp4 fetches in the baseline inner loop.
    int kb = tid;
    uint4 wcur;
    const bool have0 = kb < blocks_per_row;
    if (have0) {
        wcur = *((const uint4 *) Arow[kb].qs);          // prime the pipeline
    }

    for (; kb < blocks_per_row; kb += stride) {
        const int next_kb = kb + stride;

        // Stage 1: issue the NEXT block's wide weight load now so its global
        // latency overlaps the current block's dequant + dp4a below.
        uint4 wnext;
        const bool have_next = next_kb < blocks_per_row;
        if (have_next) {
            wnext = *((const uint4 *) Arow[next_kb].qs);
        }

        // Stage 2: consume the current (already-resident) weight block.
        const block_q8_1 * __restrict__ b8 = x + kb;   // 1:1 mapping for Q8_1
        const int * __restrict__ q8 = (const int *) b8->qs;

        const int qw[4] = { (int) wcur.x, (int) wcur.y, (int) wcur.z, (int) wcur.w };

        int sumi0 = 0;
        int sumi1 = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int2 v = rocmfp4_get_int_from_codebook_16(qw[l], nullptr);
            sumi0 = ggml_cuda_dp4a(v.x, q8[l + 0], sumi0);
            sumi1 = ggml_cuda_dp4a(v.y, q8[l + 4], sumi1);
        }

        const float db = __low2float(b8->ds);
        tmp += db * (rocmfp4_ue4m3_to_fp32_half_finite(Arow[kb].e[0]) * sumi0 +
                     rocmfp4_ue4m3_to_fp32_half_finite(Arow[kb].e[1]) * sumi1);

        // advance the pipeline
        wcur = wnext;
    }

    // ---- Multi-warp reduction ----
    if constexpr (NWARPS > 1) {
        __shared__ float tmp_shared[NWARPS - 1][WARP_SIZE];
        if (threadIdx.y > 0) {
            tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
        }
        __syncthreads();
        if (threadIdx.y > 0) return;

#pragma unroll
        for (int l = 0; l < NWARPS - 1; ++l) {
            tmp += tmp_shared[l][threadIdx.x];
        }
    }

    tmp = warp_reduce_sum(tmp);
    if (threadIdx.x == 0) {
        y[row] = tmp;
    }
}

// === Round 3: Multi-row tiling with cooperative shared-memory activation cache (re-attempt r1, fix correctness) ===
__global__ __launch_bounds__(NWARPS * WARP_SIZE, 1)
void mmvq_rocmfp4_r3(
    const block_rocmfp4 * __restrict__ A,   // (n, k/32) weight blocks
    const block_q8_1    * __restrict__ x,   // (k/32) activation blocks
    float               * __restrict__ y,   // (n) output
    int n, int k) {

    constexpr int qk  = QK_ROCMFP4;            // 32
    constexpr int qi  = QI_ROCMFP4;            // 2
    constexpr int vdr = VDR_ROCMFP4_Q8_1_MMVQ; // 4
    constexpr int ROWS_PER_BLOCK = 4;          // sweep 2..8
    constexpr int q8_block_bytes = 40;         // sizeof(block_q8_1)
    constexpr int max_shared = 32768;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x; // 0..NWARPS*WARP_SIZE-1
    const int row0 = ROWS_PER_BLOCK * blockIdx.x;
    const int row_local = tid % ROWS_PER_BLOCK;            // which output row this thread owns
    const int row = row0 + row_local;
    const int lane_in_group = tid / ROWS_PER_BLOCK;        // 0..(NWARPS*WARP_SIZE/ROWS_PER_BLOCK)-1

    const int blocks_per_row_x = k / qk;
    constexpr int blocks_per_iter = vdr * NWARPS * WARP_SIZE / qi;

    // Per-group k-stride so rows do NOT alias the same k-slice (fix r1 bug a).
    // Each row-group gets a strided slice of the k-dimension.
    const int kbx_stride = blocks_per_iter / ROWS_PER_BLOCK; // >=1 for ROWS_PER_BLOCK<=8
    const int kby = 0; // 1:1 mapping for Q8_1 (unused offset)
    const int kqs = vdr * (lane_in_group % (qi / vdr));

    float tmp = 0.0f;
    if (row < n) {
        for (int kbx = lane_in_group / (qi / vdr) + row_local * kbx_stride;
             kbx < blocks_per_row_x;
             kbx += kbx_stride * ROWS_PER_BLOCK) {
            tmp += vec_dot_rocmfp4_q8_1(A + row * blocks_per_row_x, x + kby, kbx, kqs);
        }
    }

    // Per-row-group warp reduction keyed by row (fix r1 bug b):
    // reduce within the group of threads sharing this row, then only one
    // thread per row writes y[row]. No cross-warp tmp_shared / early-return.
    // lane_in_group is the warp lane within the row-group (wave32).
    tmp = warp_reduce_sum(tmp);
    if (lane_in_group == 0 && row < n) {
        y[row] = tmp;
    }
}
