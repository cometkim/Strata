// src/kernels/cuda/kv_q4.cu - see include/strata/kernels/kv_q4.hpp. Q4_0 KV with Walsh-Hadamard rotation
// (from PR #21 by code-martin; KV-streaming integration and the deterministic group maximum added on merge).
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fast Walsh-Hadamard Transform for N = 256: one warp per row, 8 values per lane in registers.
// Orthonormal (scale 1/sqrt(256) = 1/16), so it is its own inverse.
__global__ void fwht256_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t n_rows, float scale) {
    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size;   // 8

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) return;

    const float* row_src = src + r * N;
    float* row_dst = dst + r * N;

    float reg[el_w];
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * warp_size + lane] * scale;

    // the low 5 index bits live across lanes
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    // the high 3 bits across each lane's registers
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) row_dst[i * warp_size + lane] = reg[i];
}

// One 32-value group in one warp (lane = element), ggml's q4_0: d = (the value of largest |x|) / -8,
// q = clamp(trunc(x / d + 8.5), 0, 15). Returns the scale bits; `byte` is lane t's packed byte for t < 16
// (element t in the low nibble, t + 16 in the high one). Ties in |x| resolve to the larger value in EVERY lane,
// so all lanes agree on d (a plain `a > amax` could leave lanes with opposite signs and one block two scales).
__device__ __forceinline__ uint16_t q4_group(float x, int lane, uint8_t& byte) {
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rz(x * id + 8.5f);
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint8_t qhi = __shfl_down_sync(0xffffffffu, qc, 16);
    byte = (uint8_t) (qc | (qhi << 4));
    (void) lane;
    return f16_from_f32(d);
}

__device__ __forceinline__ void q4_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}

// ggml_fp32_to_ue4m3, bit for bit: saturating at 448, the mantissa rounded half up from the FP32 bits, subnormals
// (below 2^-6) as round(x * 2^9) clamped to 1..7, anything not > 0 as 0.
__device__ __forceinline__ uint8_t ue4m3_from_f32(float x) {
    if (!(x > 0.0f)) return 0;
    if (x > 448.0f) x = 448.0f;
    const uint32_t bits = __float_as_uint(x);
    int e = (int) ((bits >> 23) & 0xFF) - 127 + 7;
    if (e <= 0) {
        const int m = (int) __fadd_rn(__fmul_rn(x, 512.0f), 0.5f);
        return m < 1 ? 0 : (uint8_t) (m > 7 ? 7 : m);
    }
    if (e >= 15) return 0x7E;
    int m = (int) ((bits >> 20) & 0x7) + (int) ((bits >> 19) & 1);
    if (m > 7) {
        m = 0;
        if (++e >= 15) return 0x7E;
    }
    return (uint8_t) ((e << 3) | m);
}

// The E2M1 code nearest xs at the (halved) scale d, and its squared error. The magnitude is the number of midpoints
// between the doubled values (0.5, 1.5, 2.5, 3.5, 5, 7, 10) times d that |xs| exceeds: every such product is exact in
// FP32 (at most 7 significant bits), so a value on a midpoint takes the smaller magnitude, as ggml's best_index_mxfp4
// takes the first of equals, with no rounding in the comparison. Zero is the positive code, a negative value the
// code with bit 3 set. Seven comparisons instead of ggml's sixteen-way search.
__device__ __forceinline__ int nvfp4_nearest(float xs, float d, float& sq) {
    const float a = fabsf(xs);
    const int mag = (a > __fmul_rn(0.5f, d)) + (a > __fmul_rn(1.5f, d)) + (a > __fmul_rn(2.5f, d)) +
                    (a > __fmul_rn(3.5f, d)) + (a > __fmul_rn(5.0f, d)) + (a > __fmul_rn(7.0f, d)) +
                    (a > __fmul_rn(10.0f, d));
    const int code = mag | (mag != 0 && xs < 0.0f ? 8 : 0);
    const float e = __fsub_rn(__fmul_rn((float) nvfp4_kvalue2(code), d), xs);
    sq = __fmul_rn(e, e);
    return code;
}

// One 32-value group as two NVFP4 sub-blocks of 16 (lanes 0-15 and 16-31) of x / KV_NVFP4_GLOBAL. The scale is the
// UE4M3 value of least squared error among four: ggml's (quantize_row_nvfp4_ref: amax / 6 rounded to UE4M3), one
// code above it and two below (UE4M3 codes are ordered by value; a code outside 1..0x7E is not tried), each with its
// nearest codes (nvfp4_nearest). Measured on Gaussian rows (what the rotation makes of K/V): relative error 0.086
// instead of ggml's 0.095 - q4_0's 0.086 - in the same format, which every reader decodes unchanged. The half's error
// sum is the one its first lane holds (a butterfly in a fixed order), so all 16 lanes choose alike; ties keep ggml's
// scale. Returns the lane's sub-block scale; `byte` is lane t's packed byte for (t & 15) < 8 - value t in the low
// nibble, t + 8 in the high one, ggml's layout.
__device__ __forceinline__ uint16_t nvfp4_group(float x, int lane, uint8_t& byte) {
    const float xs = x * (1.0f / KV_NVFP4_GLOBAL);   // a power of two: exact
    float amax = fabsf(xs);
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const int ue0 = ue4m3_from_f32(__fdiv_rn(amax, 6.0f));
    int best_ue = ue0, best_code = 0;
    float best_sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int cand = ue0 + (k == 0 ? 0 : k == 1 ? 1 : 1 - k);   // ue0, ue0 + 1, ue0 - 1, ue0 - 2
        const bool ok = k == 0 || (cand >= 1 && cand <= 0x7E);      // the same in all 16 lanes of the half
        float sq;
        const int code = nvfp4_nearest(xs, nvfp4_ue4m3_half((uint8_t) (ok ? cand : ue0)), sq);
        float sum = sq;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
        sum = __shfl_sync(0xffffffffu, sum, lane & 16);
        if (k == 0 || (ok && sum < best_sum)) { best_sum = sum; best_ue = cand; best_code = code; }
    }
    const int hi = __shfl_down_sync(0xffffffffu, best_code, 8);
    byte = (uint8_t) (best_code | (hi << 4));
    return (uint16_t) best_ue;
}

// One NVFP4 block (64 values) per 64 threads: warp w quantizes values 32w..32w+31 (sub-blocks 2w and 2w + 1) into the
// block's 36 bytes in shared memory, and threads 0-8 then store them as nine aligned 32-bit words to each destination.
// Wide stores matter for the host copy (KV streaming), which is written over PCIe: the 1-byte stores of two warps per
// block made the prompt path's append 7x slower there than q4_0's (38 vs 5.4 ms per 8,192 cells, RTX 5090).
__device__ __forceinline__ void nvfp4_block(float x, uint32_t* words) {
    const int t = threadIdx.x, w = t >> 5, lane = t & 31;
    uint8_t byte;
    const uint16_t ue = nvfp4_group(x, lane, byte);
    uint8_t* wb = reinterpret_cast<uint8_t*>(words);
    const int sub = w * 2 + lane / 16, j = lane % 16;
    if (j == 0) wb[sub] = (uint8_t) ue;
    if (j < 8) wb[4 + sub * 8 + j] = byte;
    __syncthreads();
}
__device__ __forceinline__ void nvfp4_put(uint8_t* pool, long long row, int blk, const uint32_t* words) {
    uint8_t* at = pool + row * (long long) sizeof(block_nvfp4_kv) * 4 + blk * (int) sizeof(block_nvfp4_kv);
    if (threadIdx.x < 9) reinterpret_cast<uint32_t*>(at)[threadIdx.x] = words[threadIdx.x];
}

// The NVFP4 append (as kv_append_q4_kernel): one block = one 64-value NVFP4 block of one KV head of K or V, 64 threads.
__global__ void kv_append_nvfp4_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                       const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                       int step_stride, int planes, const float* __restrict__ kcur,
                                       const float* __restrict__ vcur, int kv_heads, int head_dim, int page_size,
                                       KvHostPools host) {
    __shared__ uint32_t words[9];
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, blk = blockIdx.y;
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    nvfp4_block((is_v ? vcur : kcur)[step_off + h * head_dim + blk * QK_NVFP4_KV + threadIdx.x], words);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) nvfp4_put(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), blk, words);
    if (host.k_q4 != nullptr)
        nvfp4_put(is_v ? host.v_q4 : host.k_q4, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), blk,
                  words);
}

// The prompt path's NVFP4 append (as kv_append_q4_batch_kernel): grid (T, kv_heads, blocks), 64 threads.
__global__ void kv_append_nvfp4_batch_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                             const int32_t* __restrict__ table, int64_t pos0,
                                             const float* __restrict__ K, const float* __restrict__ V, int kv_heads,
                                             int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                             KvHostPools stage) {
    __shared__ uint32_t words[9];
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, blk = blockIdx.z;
    const bool is_v = is_v_grid != 0;
    nvfp4_block((is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + blk * QK_NVFP4_KV + threadIdx.x], words);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) nvfp4_put(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), blk, words);
    if (host.k_q4 != nullptr) nvfp4_put(is_v ? host.v_q4 : host.k_q4, row_id, blk, words);
    if (stage.k_q4 != nullptr) nvfp4_put(is_v ? stage.v_q4 : stage.k_q4, row_id, blk, words);
}

// One block = one 32-value group of one KV head of K (plane 0) or V (plane 1) for token step_idx; 32 threads.
__global__ void kv_append_q4_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                    int step_stride, int planes,
                                    const float* __restrict__ kcur, const float* __restrict__ vcur,
                                    int kv_heads, int head_dim, int page_size, KvHostPools host) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    const float x = (is_v ? vcur : kcur)[step_off + h * head_dim + b * QK4_0 + t];
    uint8_t byte;
    const uint16_t d = q4_group(x, t, byte);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, byte);
    if (host.k_q4 != nullptr)
        q4_store(is_v ? host.v_q4 : host.k_q4, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t,
                 d, byte);
}

// The prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool (identity layout) when given.
__global__ void kv_append_q4_batch_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                          const int32_t* __restrict__ table, int64_t pos0,
                                          const float* __restrict__ K, const float* __restrict__ V,
                                          int kv_heads, int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                          KvHostPools stage) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, b = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK4_0 + th];
    uint8_t byte;
    const uint16_t d = q4_group(x, th, byte);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, byte);
    if (host.k_q4 != nullptr) q4_store(is_v ? host.v_q4 : host.k_q4, row_id, b, th, d, byte);
    if (stage.k_q4 != nullptr) q4_store(is_v ? stage.v_q4 : stage.k_q4, row_id, b, th, d, byte);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths)
__global__ void kv_gather_q4_kernel(const uint8_t* __restrict__ k_q4, const uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / QK4_0;                        // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0);    // 144
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;

    const int t = threadIdx.x;
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
    const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
    const float kd = f32_from_f16(k_blk->d);
    const float vd = f32_from_f16(v_blk->d);
    const int j = t < 16 ? t : (t - 16);
    const uint8_t k_byte = k_blk->qs[j];
    const uint8_t v_byte = v_blk->qs[j];
    const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
    const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);
    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4_0 + t);
    k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
    v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
}

// The NVFP4 gather: the same grid as kv_gather_q4_kernel (32 threads per 32-value group). Every decoded value is
// exact in FP32 (code x scale has at most 6 significant bits, the global scale is a power of two).
__global__ void kv_gather_nvfp4_kernel(const uint8_t* __restrict__ k_q4, const uint8_t* __restrict__ v_q4,
                                       const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                       const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                       uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int groups = head_dim / 32;
    const int bytes_per_head = (head_dim / QK_NVFP4_KV) * (int) sizeof(block_nvfp4_kv);
    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= n_ids * kv_heads * groups) return;
    const int t = threadIdx.x;
    const long long id = blk_idx / (kv_heads * groups);
    const int rem = (int) (blk_idx % (kv_heads * groups));
    const int h = rem / groups, b = rem % groups;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int sub = (b % 2) * 2 + t / 16, j = t % 16;
    const block_nvfp4_kv* kb = reinterpret_cast<const block_nvfp4_kv*>(k_q4 + row * bytes_per_head) + b / 2;
    const block_nvfp4_kv* vb = reinterpret_cast<const block_nvfp4_kv*>(v_q4 + row * bytes_per_head) + b / 2;
    const uint8_t kbyte = kb->qs[sub * 8 + (j & 7)], vbyte = vb->qs[sub * 8 + (j & 7)];
    const int kc = j < 8 ? (kbyte & 0x0F) : (kbyte >> 4), vc = j < 8 ? (vbyte & 0x0F) : (vbyte >> 4);
    const long long dst = (id * kv_heads + h) * head_dim + b * 32 + t;
    k_scratch[dst] = f16_from_f32((float) nvfp4_kvalue2(kc) * nvfp4_ue4m3_half(kb->d[sub]) * KV_NVFP4_GLOBAL);
    v_scratch[dst] = f16_from_f32((float) nvfp4_kvalue2(vc) * nvfp4_ue4m3_half(vb->d[sub]) * KV_NVFP4_GLOBAL);
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    fwht256_kernel<<<dim3((unsigned) num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        src, dst, n_rows, 1.0f / 16.0f);
    check("fwht256 launch");
}

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q4");
    const int planes = (v_q4 == nullptr || (v_q4 == k_q4 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0), (unsigned) (n_steps * planes));
    kv_append_q4_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_q4 launch");
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    kv_append_q4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_q4_batch_kernel<<<grid, 32, 0, cs>>>(k_q4, v_q4, page_table, pos0, K, V, (int) s.n_head_kv,
                                                       (int) s.head_dim, (int) s.page_size, is_v, h, st);
    check("kv_append_q4 batch launch");
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    kv_gather_q4_kernel<<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_q4 launch");
}

void kv_append_nvfp4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                           int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                           void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_nvfp4");
    const int planes = (v_q4 == nullptr || (v_q4 == k_q4 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK_NVFP4_KV), (unsigned) (n_steps * planes));
    kv_append_nvfp4_kernel<<<grid, QK_NVFP4_KV, 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_nvfp4 launch");
}

void kv_append_nvfp4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                          const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                          const KvHostPools* host) {
    kv_append_nvfp4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_nvfp4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                     const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                     const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_nvfp4");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK_NVFP4_KV));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_nvfp4_batch_kernel<<<grid, QK_NVFP4_KV, 0, cs>>>(k_q4, v_q4, page_table, pos0, K, V,
                                                                   (int) s.n_head_kv, (int) s.head_dim,
                                                                   (int) s.page_size, is_v, h, st);
    check("kv_append_nvfp4 batch launch");
}

void kv_gather_nvfp4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                          const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                          uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int64_t total_blocks = max_ids * s.n_head_kv * (s.head_dim / 32);
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    kv_gather_nvfp4_kernel<<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_nvfp4 launch");
}

}  // namespace strata::kernels
