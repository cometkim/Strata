// include/strata/kernels/kv_q4.hpp - Q4_0 KV storage with Walsh-Hadamard rotation for the QSA layers (`--kv q4_0`).
//
// From PR #21 (code-martin). K and V are rotated by the orthonormal 256-point Hadamard matrix H before they are
// quantized to ggml's q4_0 (32 values per block: an fp16 scale + 16 bytes of 4-bit codes, 144 B per head per cell,
// 576 B per cell for K and V of both heads, vs 1,056 B in INT8). The rotation spreads a head's outlier channels
// over all 256 dimensions, which is what makes 4 bits usable. The query is rotated the same way, so
// <Hq, Hk> = <q, k> and the scores are unchanged, and the attention output (a mix of rotated values) is rotated
// back with H (self-inverse). What is lossy is the 4-bit rounding itself: measured, not assumed, in
// bench/results/2026-09-27-kv-q4.
//
// With KV streaming (--kv-resident) the codes live in the host copy and the resident slots like INT8's (KvHostPools
// k_q4/v_q4, the same block granule), so both options combine.
//
// `--kv nvfp4` (opt-in) stores the same rotated K and V as NVFP4 in the same pools: ggml's block_nvfp4 (64 values =
// four 16-value sub-blocks, one UE4M3 scale each, then 32 bytes of E2M1 codes), 36 B per 64 values, so 144 B per head
// and cell exactly as q4_0 - the pools, the streaming, the MTP ring and the snapshots move the same bytes. After a
// fixed power-of-two global scale (KV_NVFP4_GLOBAL, which moves the UE4M3 scales' range onto the rotated K/V's
// magnitudes), each 16 values get the UE4M3 scale of least squared error among ggml's (amax / 6) and its neighbours,
// and their nearest E2M1 codes (kv_q4.cu). The kv_*_nvfp4* functions below are the q4_0 ones' NVFP4 twins (same
// arguments, same pools); the q4_0 functions are unchanged.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_KV_HD __host__ __device__
#else
#define STRATA_KV_HD
#endif

namespace strata::kernels {

inline constexpr int QK4_0 = 32;

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;             // scale (fp16 bits)
    uint8_t qs[QK4_0 / 2];  // 32 4-bit codes: element j in the low nibble of qs[j], j + 16 in the high one
};
#pragma pack(pop)

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

inline constexpr int QK_NVFP4_KV = 64;       // values per block
inline constexpr int QK_NVFP4_KV_SUB = 16;   // values per UE4M3 scale

/// ggml's block_nvfp4: sub-block s holds values 16s..16s+15, value 16s+j (j < 8) in the low nibble of qs[8s+j] and
/// 16s+8+j in its high nibble.
struct block_nvfp4_kv {
    uint8_t d[QK_NVFP4_KV / QK_NVFP4_KV_SUB];   // UE4M3 scales, one per 16 values
    uint8_t qs[QK_NVFP4_KV / 2];                 // E2M1 codes (sign bit 3)
};

static_assert(sizeof(block_nvfp4_kv) == 36, "block_nvfp4_kv must be 36 bytes");
static_assert((256 / QK_NVFP4_KV) * sizeof(block_nvfp4_kv) == (256 / QK4_0) * sizeof(block_q4_0),
              "an NVFP4 head must take the q4_0 head's bytes: both formats share the 4-bit pools");

/// The global scale: stored = value / KV_NVFP4_GLOBAL, so the UE4M3 scales (normal from 2^-6 to 448) cover block
/// maxima from 6 * 2^-6 * G to 2688 * G. A power of two, so the scaling is exact. Measured, not assumed: the
/// rotated K/V of Qwen3.8-Flash-Next (bench/results/2026-10-14-kv-nvfp4).
inline constexpr float KV_NVFP4_GLOBAL = 1.0f / 16.0f;

/// E2M1 code -> value, doubled (ggml's kvalues_fp4: 0, 1, 2, 3, 4, 6, 8, 12, then the negatives).
STRATA_KV_HD inline int nvfp4_kvalue2(int code) {
    const int v = (int) ((0x0C08060403020100ull >> ((code & 7) * 8)) & 0xFF);   // the magnitudes as a byte table
    return (code & 8) ? -v : v;
}

/// UE4M3 -> value, halved to pair with the doubled codes (ggml's ggml_ue4m3_to_fp32; 0 and 0x7F give 0).
STRATA_KV_HD inline float nvfp4_ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t e = (x >> 3) & 0xF, m = x & 0x7;
    // every UE4M3 value is exact in FP32: subnormal m * 2^-10, normal 2^(e-8) * (1 + m/8) - built from its bits
    // (FP32 exponent field e + 119, mantissa m << 20), as iq_kernels.cu's ue4m3_to_f32_half
    if (e == 0) return (float) m * 0.0009765625f;
    const uint32_t bits = ((e + 119u) << 23) | (m << 20);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(bits);
#else
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
#endif
}

#if defined(__CUDACC__) || defined(__HIPCC__)
/// Four E2M1 codes (one per byte, 0..15) as four int8 doubled values: the magnitudes by the low 3 bits through one
/// byte permute, then negated per byte where bit 3 (the sign) is set.
__device__ __forceinline__ uint32_t nvfp4_codes_to_i8x4(uint32_t x) {
    const uint32_t sel = (x & 7u) | ((x >> 4) & 0x70u) | ((x >> 8) & 0x700u) | ((x >> 12) & 0x7000u);
    const uint32_t mag = __byte_perm(0x03020100u, 0x0C080604u, sel);
    const uint32_t m = ((x >> 3) & 0x01010101u) * 0xFFu;   // 0xFF in each negative byte
    return __vsub4(mag ^ m, m);                              // per byte: ~mag + 1 where negative
}
#endif

/// Bytes per cell and KV head: 8 blocks of 32 for head_dim 256 -> 144.
inline uint64_t kv_q4_bytes_per_head(int head_dim) { return (uint64_t) (head_dim / QK4_0) * sizeof(block_q4_0); }

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_q4_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_q4_bytes_per_head((int) s.head_dim) * 2;
}

/// Orthonormal Fast Walsh-Hadamard Transform of rows of 256 floats (scale 1/16): its own inverse.
void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream);
inline void fwht256_inplace_cuda(float* data, int64_t n_rows, void* stream) { fwht256_cuda(data, data, n_rows, stream); }

/// Append the (already rotated) cell at step[kStepPos]. With a host copy (KV streaming) it is written there too,
/// and to VRAM only if its block is resident.
void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host = nullptr);

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host = nullptr);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into `stage` (identity
/// layout, the one-layer staging pool of a streamed session) when given.
void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host = nullptr,
                  const KvHostPools* stage = nullptr);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated).
void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream);

/// `--kv nvfp4`: the four above with NVFP4 blocks in the same pools (CUDA; the HIP build compiles them too).
void kv_append_nvfp4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                          const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                          const KvHostPools* host = nullptr);
void kv_append_nvfp4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                           int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                           void* stream, const KvHostPools* host = nullptr);
void kv_append_nvfp4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                     const float* V, const QsaShapes& s, void* stream, const KvHostPools* host = nullptr,
                     const KvHostPools* stage = nullptr);
void kv_gather_nvfp4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                          const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                          uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
