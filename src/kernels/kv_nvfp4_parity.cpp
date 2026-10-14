// src/kernels/kv_nvfp4_parity.cpp - `--kv nvfp4` (kv_q4.hpp) against ggml (GPU, synthetic, no model).
//
// The bytes are ggml's block_nvfp4 and ggml's to_float decodes them. The quantizer picks each 16 values' scale by
// least squared error among ggml's (amax / 6) and its neighbours, so its oracle is a host transcription of that
// search on ggml's rounding of the scale (ggml_fp32_to_ue4m3) and ggml's nearest code (best_index_mxfp4, in exact
// arithmetic), and ggml_quantize_chunk(GGML_TYPE_NVFP4) is the baseline it must not do worse than. Checks:
//   0. the round trip: the search's error against ggml's quantizer's, on Gaussian rows in the global scale's range;
//   1. kv_append_nvfp4_step writes the reference bytes, bit for bit, through a shuffled page table - over cells
//      whose magnitudes span the UE4M3 range (subnormal scales, saturation at 448, zeros, exact ties);
//   2. kv_append_nvfp4_steps and the prompt path's kv_append_nvfp4 write the same bytes as the per-cell append;
//   3. kv_gather_nvfp4_step gives fp16 of ggml's values times the global scale;
//   4. qsa_decode_attn_step / _batch over the NVFP4 pools (KV_MODE 4) equal, bit for bit, the same kernels over
//      FP16 pools that hold those values (every NVFP4 value is exact in FP16, and load8 hands the attention the
//      same floats in the same order);
//   5. the tensor-core prompt kernel (qsa_prompt_attn's NVFP4 kernel) against the same FP16-pool attention, within FP32-level
//      accuracy (its summation order differs, as for every tensor-core mode).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// ggml_fp32_to_ue4m3 (ggml-impl.h), transcribed
uint8_t ue4m3_ref(float x) {
    if (!(x > 0.0f)) return 0;
    if (x > 448.0f) x = 448.0f;
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    const int fp32_exp = (int) ((bits >> 23) & 0xFF) - 127, fp32_man = (int) ((bits >> 20) & 0x7);
    int e = fp32_exp + 7;
    if (e <= 0) {
        int man = (int) (x * 512.0f + 0.5f);
        if (man > 7) man = 7;
        return man < 1 ? 0 : (uint8_t) man;
    }
    if (e >= 15) return 0x7E;
    int m = fp32_man + (int) ((bits >> 19) & 1);
    if (m > 7) {
        m = 0;
        if (++e >= 15) return 0x7E;
    }
    return (uint8_t) ((e << 3) | m);
}

// ggml_ue4m3_to_fp32 (halved, as ggml's), written independently of kv_q4.hpp
float ue4m3_half_ref(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 0x7;
    return e == 0 ? std::ldexp((float) m, -9) * 0.5f : std::ldexp(1.0f + (float) m / 8.0f, e - 7) * 0.5f;
}

const int8_t kKv2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};   // ggml's kvalues_fp4

// products and differences rounded separately (as the kernel's __fmul_rn / __fsub_rn): no contraction to an FMA
float mul_rn(float a, float b) { volatile float r = a * b; return r; }
float sub_rn(float a, float b) { volatile float r = a - b; return r; }
float add_rn(float a, float b) { volatile float r = a + b; return r; }

// The nearest code as kv_q4.cu's nvfp4_nearest, written as ggml's best_index_mxfp4 is: the first code of least
// |value - x| over all 16, in exact (double) arithmetic - every value, x and difference here is exact in a double,
// so this is the mathematical nearest with ties to the smaller magnitude and +0 for zero, which the kernel's midpoint
// count must equal; then the squared error as the kernel rounds it
int nearest_ref(float x, float d, float& sq) {
    int best = 0;
    double best_err = std::fabs(0.0 - (double) x);
    for (int i = 1; i < 16; ++i) {
        const double err = std::fabs((double) kKv2[i] * (double) d - (double) x);
        if (err < best_err) { best = i; best_err = err; }
    }
    const float e = sub_rn(mul_rn((float) kKv2[best], d), x);
    sq = mul_rn(e, e);
    return best;
}

// One 16-value sub-block as kv_q4.cu's nvfp4_group: ggml's scale, one code above, two below; the error sums in the
// kernel's butterfly order (lane 0's); ties keep the earlier candidate
uint8_t quant_sub_ref(const float* xs, int codes[16]) {
    float amax = 0.0f;
    for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(xs[j]));
    const int ue0 = ue4m3_ref(amax / 6.0f);
    int best_ue = ue0;
    float best_sum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const int cand = ue0 + (k == 0 ? 0 : k == 1 ? 1 : 1 - k);
        const bool ok = k == 0 || (cand >= 1 && cand <= 0x7E);
        if (!ok) continue;
        const float d = ue4m3_half_ref((uint8_t) cand);
        int c[16];
        float v[16];
        for (int j = 0; j < 16; ++j) c[j] = nearest_ref(xs[j], d, v[j]);
        for (int o = 8; o > 0; o >>= 1) {
            float w[16];
            for (int j = 0; j < 16; ++j) w[j] = add_rn(v[j], v[j ^ o]);
            std::memcpy(v, w, sizeof v);
        }
        if (k == 0 || v[0] < best_sum) {
            best_sum = v[0];
            best_ue = cand;
            std::memcpy(codes, c, sizeof c);
        }
    }
    return (uint8_t) best_ue;
}

// A row of D values (already divided by the global scale) as D / 64 block_nvfp4 (ggml's layout)
void quant_row_ref(const float* xs, int D, uint8_t* out) {
    for (int b = 0; b < D / 64; ++b)
        for (int u = 0; u < 4; ++u) {
            int codes[16];
            uint8_t* blk = out + b * 36;
            blk[u] = quant_sub_ref(xs + b * 64 + u * 16, codes);
            for (int j = 0; j < 8; ++j) blk[4 + u * 8 + j] = (uint8_t) (codes[j] | (codes[j + 8] << 4));
        }
}

}  // namespace

int main() {
    std::printf("=== Running kv_nvfp4_parity test ===\n");
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size;
    const int pages = 24, cells = pages * P;
    const int64_t row_bytes = (int64_t) k::kv_q4_bytes_per_head(D);   // 144, the NVFP4 head too
    if (ggml_row_size(GGML_TYPE_NVFP4, D) != (size_t) row_bytes) {
        std::fprintf(stderr, "FAIL: ggml's NVFP4 row is %zu bytes, the pools hold %lld\n",
                     ggml_row_size(GGML_TYPE_NVFP4, D), (long long) row_bytes);
        return 1;
    }
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 7 + 5) % pages;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");

    // K/V [cell][head][dim]; each (cell, head) row has its own magnitude, 2^-14 .. 2^10, so the UE4M3 scales run
    // from subnormal (and flushed to zero) to saturated; some rows are special cases.
    std::vector<float> K((size_t) cells * H * D), V((size_t) cells * H * D);
    std::uniform_int_distribution<int> mag(-14, 10);
    for (std::vector<float>* X : {&K, &V})
        for (int r = 0; r < cells * H; ++r) {
            const float m = std::ldexp(1.0f, mag(rng));
            float* x = X->data() + (size_t) r * D;
            const int kind = r % 23;
            for (int d = 0; d < D; ++d) {
                if (kind == 0) x[d] = 0.0f;                                   // all zero
                else if (kind == 1) x[d] = d % 2 ? -0.0f : 0.0f;              // signed zeros
                else if (kind == 2) x[d] = (float) ((d % 13) - 6) * 0.25f * m;   // exact grid points and ties
                else if (kind == 3) x[d] = d % 16 == 3 ? 3000.0f : nd(rng);  // one outlier per sub-block
                else x[d] = nd(rng) * m;
            }
        }

    // ---- the oracle: the search's NVFP4 of x / G (host), ggml's quantizer as the baseline, both decoded by ggml
    const float invG = 1.0f / k::KV_NVFP4_GLOBAL;
    std::vector<uint8_t> refK((size_t) cells * H * row_bytes), refV((size_t) cells * H * row_bytes);
    std::vector<float> valK(K.size()), valV(V.size()), ggmlK(K.size());
    {
        std::vector<float> scaled(K.size());
        std::vector<uint8_t> gq((size_t) cells * H * row_bytes);
        const ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
        for (int pass = 0; pass < 2; ++pass) {
            const std::vector<float>& X = pass ? V : K;
            std::vector<uint8_t>& R = pass ? refV : refK;
            std::vector<float>& Y = pass ? valV : valK;
            for (size_t i = 0; i < X.size(); ++i) scaled[i] = X[i] * invG;
            for (int r = 0; r < cells * H; ++r) {
                quant_row_ref(scaled.data() + (size_t) r * D, D, R.data() + (size_t) r * row_bytes);
                tr->to_float(R.data() + (size_t) r * row_bytes, Y.data() + (size_t) r * D, D);
                for (int d = 0; d < D; ++d) Y[(size_t) r * D + d] *= k::KV_NVFP4_GLOBAL;
            }
            if (pass == 0) {   // the baseline: ggml's own quantizer on the same K
                ggml_quantize_chunk(GGML_TYPE_NVFP4, scaled.data(), gq.data(), 0, (int64_t) cells * H, D, nullptr);
                for (int r = 0; r < cells * H; ++r) {
                    tr->to_float(gq.data() + (size_t) r * row_bytes, ggmlK.data() + (size_t) r * D, D);
                    for (int d = 0; d < D; ++d) ggmlK[(size_t) r * D + d] *= k::KV_NVFP4_GLOBAL;
                }
            }
        }
    }

    // ---- 0. the round trip: Gaussian rows whose RMS lies in the range the global scale is set for (2^-9 .. 2^4;
    // below it sub-blocks flush to zero, above it the scale saturates). ggml's quantizer measures a relative error of
    // 0.095 there, the scale search 0.086 (q4_0's, numpy); the search never loses to ggml's scale on any sub-block.
    {
        double sum_s = 0.0, sum_g = 0.0;
        int n = 0, worse = 0;
        for (int r = 0; r < cells * H; ++r) {
            double e2s = 0.0, e2g = 0.0, x2 = 0.0;
            for (int d = 0; d < D; ++d) {
                const double x = K[(size_t) r * D + d], ys = valK[(size_t) r * D + d], yg = ggmlK[(size_t) r * D + d];
                e2s += (x - ys) * (x - ys);
                e2g += (x - yg) * (x - yg);
                x2 += x * x;
            }
            if (e2s > e2g * (1.0 + 1e-6) + 1e-30) ++worse;
            const double rms = std::sqrt(x2 / D);
            if (r % 23 < 4 || rms < std::ldexp(1.0, -9) || rms >= std::ldexp(1.0, 5)) continue;   // special / out of range
            sum_s += std::sqrt(e2s / x2);
            sum_g += std::sqrt(e2g / x2);
            ++n;
        }
        const double rel = sum_s / n, rel_g = sum_g / n;
        const bool ok = rel < 0.09 && rel < rel_g && worse == 0;
        std::printf("[0/5] round trip over %d rows: relative error %.4f (ggml's quantizer %.4f), %d rows worse: %s\n", n,
                    rel, rel_g, worse, ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    // ---- 1. the per-cell append, through the shuffled table
    const size_t pool = (size_t) cells * H * row_bytes;
    uint8_t *d_k = dalloc<uint8_t>(pool), *d_v = dalloc<uint8_t>(pool);
    float* d_K = dalloc<float>(K.size());
    float* d_V = dalloc<float>(V.size());
    ck(cudaMemcpy(d_K, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
    ck(cudaMemcpy(d_V, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");
    std::vector<int32_t> steps((size_t) cells * k::kStepCount);
    for (int c = 0; c < cells; ++c) k::qsa_step_fill(&steps[(size_t) c * k::kStepCount], c, s);
    int32_t* d_steps = dalloc<int32_t>(steps.size());
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");
    for (int c = 0; c < cells; ++c)
        k::kv_append_nvfp4_step(d_k, d_v, d_table, d_steps + (size_t) c * k::kStepCount, d_K + (size_t) c * H * D,
                                d_V + (size_t) c * H * D, s, nullptr, nullptr);
    ck(cudaDeviceSynchronize(), "append");
    std::vector<uint8_t> hk(pool), hv(pool);
    ck(cudaMemcpy(hk.data(), d_k, pool, cudaMemcpyDeviceToHost), "hk");
    ck(cudaMemcpy(hv.data(), d_v, pool, cudaMemcpyDeviceToHost), "hv");
    int bad = 0, shown = 0;
    for (int c = 0; c < cells; ++c)
        for (int h = 0; h < H; ++h) {
            const size_t r = ((size_t) table[c / P] * H + h) * P + c % P, l = (size_t) c * H + h;
            for (int pass = 0; pass < 2; ++pass) {
                const uint8_t* got = (pass ? hv : hk).data() + r * row_bytes;
                const uint8_t* want = (pass ? refV : refK).data() + l * row_bytes;
                if (std::memcmp(got, want, (size_t) row_bytes) != 0) {
                    ++bad;
                    if (shown++ < 4) {
                        int b = 0;
                        while (got[b] == want[b]) ++b;
                        std::fprintf(stderr, "  cell %d head %d %c: byte %d is 0x%02x, reference 0x%02x\n", c, h,
                                     pass ? 'V' : 'K', b, got[b], want[b]);
                    }
                }
            }
        }
    std::printf("[1/5] append vs the reference quantizer: %s (%d of %d rows differ)\n", bad ? "FAIL" : "PASS", bad,
                cells * H * 2);
    int fails = bad != 0;

    // ---- 2. the batched append (verify / MTP) and the prompt path's append write the same bytes
    {
        uint8_t *d_k2 = dalloc<uint8_t>(pool), *d_v2 = dalloc<uint8_t>(pool);
        uint8_t *d_k3 = dalloc<uint8_t>(pool), *d_v3 = dalloc<uint8_t>(pool);
        for (int c0 = 0; c0 < cells; c0 += 5) {
            const int n = std::min(5, cells - c0);
            k::kv_append_nvfp4_steps(d_k2, d_v2, d_table, d_steps + (size_t) c0 * k::kStepCount, k::kStepCount, n,
                                     d_K + (size_t) c0 * H * D, d_V + (size_t) c0 * H * D, s, nullptr, nullptr);
        }
        k::kv_append_nvfp4(d_k3, d_v3, d_table, 0, cells, d_K, d_V, s, nullptr, nullptr, nullptr);
        ck(cudaDeviceSynchronize(), "append2");
        std::vector<uint8_t> a(pool);
        int b2 = 0;
        for (uint8_t* p : {d_k2, d_v2, d_k3, d_v3}) {
            ck(cudaMemcpy(a.data(), p, pool, cudaMemcpyDeviceToHost), "a");
            const std::vector<uint8_t>& ref = (p == d_k2 || p == d_k3) ? hk : hv;
            if (std::memcmp(a.data(), ref.data(), pool) != 0) ++b2;
        }
        std::printf("[2/5] kv_append_nvfp4_steps and kv_append_nvfp4 vs the per-cell append: %s\n", b2 ? "FAIL" : "PASS");
        fails += b2 != 0;
    }

    // ---- 3. the gather: fp16 of ggml's values
    std::vector<int32_t> ids(cells);
    for (int i = 0; i < cells; ++i) ids[i] = (i * 37 + 11) % cells;   // every cell once, out of order
    int32_t* d_ids = dalloc<int32_t>(cells);
    ck(cudaMemcpy(d_ids, ids.data(), cells * 4, cudaMemcpyHostToDevice), "ids");
    int32_t step[k::kStepCount];
    k::qsa_step_fill(step, cells - 1, s);
    step[k::kStepWidth] = cells;
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);
    ck(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step");
    {
        uint16_t *d_ksc = dalloc<uint16_t>(K.size()), *d_vsc = dalloc<uint16_t>(V.size());
        k::kv_gather_nvfp4_step(d_k, d_v, d_table, d_ids, d_step, cells, s, d_ksc, d_vsc, nullptr);
        ck(cudaDeviceSynchronize(), "gather");
        std::vector<uint16_t> gk(K.size()), gv(V.size());
        ck(cudaMemcpy(gk.data(), d_ksc, gk.size() * 2, cudaMemcpyDeviceToHost), "gk");
        ck(cudaMemcpy(gv.data(), d_vsc, gv.size() * 2, cudaMemcpyDeviceToHost), "gv");
        int b3 = 0;
        for (int i = 0; i < cells; ++i)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    const size_t o = ((size_t) i * H + h) * D + d, src = ((size_t) ids[i] * H + h) * D + d;
                    if (gk[o] != k::f16_from_f32(valK[src]) || gv[o] != k::f16_from_f32(valV[src])) ++b3;
                    if (k::f32_from_f16(k::f16_from_f32(valK[src])) != valK[src]) ++b3;   // exact in fp16
                }
        std::printf("[3/5] gather vs ggml's values: %s (%d bad)\n", b3 ? "FAIL" : "PASS", b3);
        fails += b3 != 0;
    }

    // ---- 4. the decode attention over NVFP4 (KV_MODE 4) vs FP16 pools holding the same values
    {
        std::vector<uint16_t> fk((size_t) cells * H * D), fv((size_t) cells * H * D);
        for (int c = 0; c < cells; ++c)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    const size_t r = ((size_t) table[c / P] * H + h) * P + c % P, l = ((size_t) c * H + h) * D + d;
                    fk[r * D + d] = k::f16_from_f32(valK[l]);
                    fv[r * D + d] = k::f16_from_f32(valV[l]);
                }
        uint16_t *d_fk = dalloc<uint16_t>(fk.size()), *d_fv = dalloc<uint16_t>(fv.size());
        ck(cudaMemcpy(d_fk, fk.data(), fk.size() * 2, cudaMemcpyHostToDevice), "fk");
        ck(cudaMemcpy(d_fv, fv.data(), fv.size() * 2, cudaMemcpyHostToDevice), "fv");
        k::QsaAttnPools p4, p16;
        p4.k_q4 = d_k; p4.v_q4 = d_v; p4.nvfp4 = true; p4.page_table = d_table;
        p16.k_pool = d_fk; p16.v_pool = d_fv; p16.page_table = d_table;
        const int nq = 3;
        std::vector<float> q((size_t) nq * s.n_head * D);
        for (auto& x : q) x = nd(rng) * 4.0f;   // sharp enough that the softmax does not average everything out
        float* d_q = dalloc<float>(q.size());
        ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q");
        std::vector<int32_t> ids3((size_t) nq * cells), steps3((size_t) nq * k::kStepCount);
        for (int i = 0; i < nq; ++i) {
            for (int c = 0; c < cells; ++c) ids3[(size_t) i * cells + c] = ids[c];
            std::memcpy(&steps3[(size_t) i * k::kStepCount], step, sizeof step);
            steps3[(size_t) i * k::kStepCount + k::kStepWidth] = cells - 9 * i;
        }
        int32_t *d_ids3 = dalloc<int32_t>(ids3.size()), *d_steps3 = dalloc<int32_t>(steps3.size());
        ck(cudaMemcpy(d_ids3, ids3.data(), ids3.size() * 4, cudaMemcpyHostToDevice), "ids3");
        ck(cudaMemcpy(d_steps3, steps3.data(), steps3.size() * 4, cudaMemcpyHostToDevice), "steps3");
        const size_t sf = k::qsa_decode_attn_scratch_floats(cells, s);
        float* d_scr = dalloc<float>(sf * nq);
        const size_t out = (size_t) nq * s.n_head * D;
        float *d_o4 = dalloc<float>(out), *d_o16 = dalloc<float>(out);
        std::vector<float> o4(out), o16(out);
        int b4 = 0;
        // one query (the decode step) ...
        k::qsa_decode_attn_step(d_q, p4, d_ids, d_step, cells, s, d_scr, d_o4, nullptr);
        k::qsa_decode_attn_step(d_q, p16, d_ids, d_step, cells, s, d_scr, d_o16, nullptr);
        ck(cudaDeviceSynchronize(), "attn step");
        ck(cudaMemcpy(o4.data(), d_o4, (size_t) s.n_head * D * 4, cudaMemcpyDeviceToHost), "o4");
        ck(cudaMemcpy(o16.data(), d_o16, (size_t) s.n_head * D * 4, cudaMemcpyDeviceToHost), "o16");
        b4 += std::memcmp(o4.data(), o16.data(), (size_t) s.n_head * D * 4) != 0;
        // ... and a batch (the prompt's fallback, verify windows)
        k::qsa_decode_attn_batch(d_q, p4, d_ids3, d_steps3, cells, s, d_scr, d_o4, nq, nullptr);
        k::qsa_decode_attn_batch(d_q, p16, d_ids3, d_steps3, cells, s, d_scr, d_o16, nq, nullptr);
        ck(cudaDeviceSynchronize(), "attn batch");
        ck(cudaMemcpy(o4.data(), d_o4, out * 4, cudaMemcpyDeviceToHost), "o4b");
        ck(cudaMemcpy(o16.data(), d_o16, out * 4, cudaMemcpyDeviceToHost), "o16b");
        b4 += std::memcmp(o4.data(), o16.data(), out * 4) != 0;
        double mx = 0.0;
        for (size_t i = 0; i < out; ++i) mx = std::max(mx, (double) std::fabs(o4[i] - o16[i]));
        std::printf("[4/5] decode attention, NVFP4 pools vs FP16 pools of the same values: %s (max |diff| %g)\n",
                    b4 ? "FAIL" : "PASS", mx);
        fails += b4 != 0;

        // 5. the tensor-core prompt path takes the same pools (its NVFP4 kernel); o16 holds the FP16-pool batch above
        if (!k::qsa_prompt_attn_batch(d_q, p4, d_ids3, d_steps3, cells, s, d_o4, nq, nullptr)) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
            std::printf("[5/5] qsa_prompt_attn NVFP4: PASS (refused on HIP - the decode kernel runs)\n");
#else
            int major = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
            const bool want = major >= 8;   // sm_80+ only (cp.async; as q4_0's mode 4)
            std::printf("[5/5] qsa_prompt_attn NVFP4: %s (refused)\n", want ? "FAIL" : "PASS");
            fails += want;
#endif
        } else {
            ck(cudaDeviceSynchronize(), "prompt attn");
            ck(cudaMemcpy(o4.data(), d_o4, out * 4, cudaMemcpyDeviceToHost), "o4p");
            double big = 0.0, diff = 0.0;
            for (size_t i = 0; i < out; ++i) {
                big = std::max(big, (double) std::fabs(o16[i]));
                diff = std::max(diff, (double) std::fabs(o4[i] - o16[i]));
            }
            const bool ok = diff <= 1e-4 * big;
            std::printf("[5/5] qsa_prompt_attn NVFP4 (tensor cores) vs FP16 pools: %s (max |diff| %.3g of max |out| %.3g)\n",
                        ok ? "PASS" : "FAIL", diff, big);
            fails += !ok;
        }
    }

    std::printf(fails ? "kv_nvfp4_parity: FAIL\n" : "kv_nvfp4_parity: PASS\n");
    return fails ? 1 : 0;
}
