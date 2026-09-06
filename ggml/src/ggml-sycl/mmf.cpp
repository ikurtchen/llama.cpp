#include "mmf.hpp"
#include "dequantize.hpp"
#include "fwht.hpp"
#include "mmid.hpp"

#ifdef GGML_SYCL_USE_ONEMKL
#include <oneapi/mkl.hpp>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

struct ggml_sycl_mmf_src1_f16_scratch {
    sycl::half * ptr = nullptr;
    size_t       capacity = 0;
};

struct ggml_sycl_mmf_f32_scratch {
    float * ptr = nullptr;
    size_t  capacity = 0;
};

enum ggml_sycl_mmf_f32_scratch_slot {
    GGML_SYCL_MMF_SRC0_F32_SCRATCH = 0,
    GGML_SYCL_MMF_SRC1_F32_SCRATCH = 1,
    GGML_SYCL_MMF_DST_F32_SCRATCH  = 2,
    GGML_SYCL_MMF_F32_SCRATCH_COUNT,
};

static ggml_sycl_mmf_src1_f16_scratch & ggml_sycl_mmf_get_src1_f16_scratch(int device) {
    static std::array<ggml_sycl_mmf_src1_f16_scratch, GGML_SYCL_MAX_DEVICES> scratch = {};
    GGML_ASSERT(device >= 0 && device < GGML_SYCL_MAX_DEVICES);
    return scratch[device];
}

static ggml_sycl_mmf_f32_scratch & ggml_sycl_mmf_get_f32_scratch(int device, ggml_sycl_mmf_f32_scratch_slot slot) {
    static std::array<std::array<ggml_sycl_mmf_f32_scratch, GGML_SYCL_MMF_F32_SCRATCH_COUNT>, GGML_SYCL_MAX_DEVICES> scratch = {};
    GGML_ASSERT(device >= 0 && device < GGML_SYCL_MAX_DEVICES);
    return scratch[device][slot];
}

static sycl::half * ggml_sycl_mmf_reserve_src1_f16_scratch(ggml_backend_sycl_context & ctx, size_t nelements) {
    auto & scratch = ggml_sycl_mmf_get_src1_f16_scratch(ctx.device);
    sycl::queue & q = ctx.stream();

    if (scratch.capacity >= nelements) {
        return scratch.ptr;
    }

    SYCL_CHECK(q.wait_and_throw());
    if (scratch.ptr != nullptr) {
        sycl::free(scratch.ptr, q);
    }

    scratch.ptr = sycl::malloc_device<sycl::half>(std::max<size_t>(nelements, 1), q);
    GGML_ASSERT(scratch.ptr != nullptr);
    scratch.capacity = nelements;
    return scratch.ptr;
}

static float * ggml_sycl_mmf_reserve_f32_scratch(
        ggml_backend_sycl_context & ctx,
        ggml_sycl_mmf_f32_scratch_slot slot,
        size_t nelements) {
    auto & scratch = ggml_sycl_mmf_get_f32_scratch(ctx.device, slot);
    sycl::queue & q = ctx.stream();

    if (scratch.capacity >= nelements) {
        return scratch.ptr;
    }

    SYCL_CHECK(q.wait_and_throw());
    if (scratch.ptr != nullptr) {
        sycl::free(scratch.ptr, q);
    }

    scratch.ptr = sycl::malloc_device<float>(std::max<size_t>(nelements, 1), q);
    GGML_ASSERT(scratch.ptr != nullptr);
    scratch.capacity = nelements;
    return scratch.ptr;
}

static bool ggml_sycl_mmf_has_packed_rows(const ggml_tensor * src0) {
    return src0->nb[0] == ggml_type_size(src0->type) &&
           src0->nb[1] == ggml_row_size(src0->type, src0->ne[0]);
}

static sycl::event ggml_sycl_convert_src1_f32_to_f16(
        sycl::queue & q,
        const ggml_tensor * src1,
        sycl::half * dst_f16) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const float * src = static_cast<const float *>(src1->data);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);

    const int64_t ne = ggml_nelements(src1);
    const int64_t block = 256;
    const int64_t nblk = (ne + block - 1) / block;

    return q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int64_t i = item.get_global_id(0);
                if (i >= ne) {
                    return;
                }

                int64_t tmp = i;
                const int64_t i13 = tmp / (ne10 * ne11 * ne12);
                tmp -= i13 * ne10 * ne11 * ne12;
                const int64_t i12 = tmp / (ne10 * ne11);
                tmp -= i12 * ne10 * ne11;
                const int64_t i11 = tmp / ne10;
                const int64_t i10 = tmp - i11 * ne10;

                dst_f16[i] = sycl::half(src[i10 + i11 * s11 + i12 * s12 + i13 * s13]);
            });
}

#ifdef GGML_SYCL_USE_ONEMKL
template <typename src0_t, typename src1_t>
static sycl::event ggml_sycl_mmf_gemm(
        sycl::queue & q,
        const src0_t * src0,
        const src1_t * src1,
        float * dst,
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t lda,
        int64_t ldb,
        int64_t ldc,
        const std::vector<sycl::event> & deps = {}) {
    return oneapi::mkl::blas::column_major::gemm(
            q,
            oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans,
            m, n, k,
            1.0f,
            src0, lda,
            src1, ldb,
            0.0f,
            dst, ldc,
            deps);
}

template <typename src0_t, typename src1_t>
static sycl::event ggml_sycl_mmf_gemm_batch(
        sycl::queue & q,
        const src0_t * src0,
        const src1_t * src1,
        float * dst,
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t lda,
        int64_t stride_a,
        int64_t ldb,
        int64_t stride_b,
        int64_t ldc,
        int64_t stride_c,
        int64_t batch_size,
        const std::vector<sycl::event> & deps = {}) {
    return oneapi::mkl::blas::column_major::gemm_batch(
            q,
            oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans,
            m, n, k,
            1.0f,
            src0, lda, stride_a,
            src1, ldb, stride_b,
            0.0f,
            dst, ldc, stride_c,
            batch_size,
            deps);
}
#endif

static bool ggml_sycl_mmf_can_use_strided_batch(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return ggml_is_contiguous_2(src0) && ggml_is_contiguous_2(src1) && ggml_is_contiguous_2(dst);
}

// Decode-path fast path for Q8_0 x F32 mul_mat: n (number of RHS columns) is small (batch=1
// decode, or a handful of speculative-decode/beam candidates). The general path dequantizes the
// whole weight row block to F32 and calls oneMKL GEMM, which for n this small is dominated by the
// cost of writing+reading the ne00*ne01 F32 scratch buffer rather than by useful FLOPs. This kernel
// instead reads the quantized bytes once and accumulates the dot product directly in registers, so
// memory traffic drops from O(ne00*ne01 f32) to O(ne00*ne01 q8_0 bytes) -- about a 4x cut in bytes
// moved, on top of removing a whole kernel launch (dequantize) and the oneMKL GEMM call.
constexpr int GGML_SYCL_MMVQ_MAX_N_COLS = 8;
constexpr int GGML_SYCL_MMVQ_WG_SIZE    = 256;

static void ggml_sycl_mul_mat_vec_q8_0_kernel(
        const block_q8_0 * __restrict__ x,
        const float * __restrict__ y,
        float * __restrict__ dst,
        int64_t nblocks_per_row,
        int64_t s1,
        int64_t d1,
        int n_cols,
        sycl::nd_item<1> item) {
    const int64_t row = item.get_group(0);
    const int tid = (int) item.get_local_id(0);
    const int wg_size = (int) item.get_local_range(0);

    const block_q8_0 * row_x = x + row * nblocks_per_row;

    float sums[GGML_SYCL_MMVQ_MAX_N_COLS] = {0.0f};

    for (int64_t ib = tid; ib < nblocks_per_row; ib += wg_size) {
        const block_q8_0 & blk = row_x[ib];
        const float d = (float) blk.d;
        const int64_t base = ib * QK8_0;
        for (int col = 0; col < n_cols; ++col) {
            const float * yv = y + col * s1 + base;
            float partial = 0.0f;
#pragma unroll
            for (int l = 0; l < QK8_0; ++l) {
                partial += (float) blk.qs[l] * yv[l];
            }
            sums[col] += partial * d;
        }
    }

    for (int col = 0; col < n_cols; ++col) {
        const float total = sycl::reduce_over_group(item.get_group(), sums[col], sycl::plus<float>());
        if (tid == 0) {
            dst[col * d1 + row] = total;
        }
    }
}

static sycl::event ggml_sycl_mul_mat_vec_q8_0(
        sycl::queue & q,
        const block_q8_0 * src0,
        const float * src1,
        float * dst,
        int64_t nrows_x,
        int64_t ncols_x,
        int64_t s1,
        int64_t d1,
        int n_cols) {
    GGML_ASSERT(n_cols <= GGML_SYCL_MMVQ_MAX_N_COLS);
    const int64_t nblocks_per_row = ncols_x / QK8_0;
    const sycl::range<1> local(GGML_SYCL_MMVQ_WG_SIZE);
    const sycl::range<1> global(nrows_x * GGML_SYCL_MMVQ_WG_SIZE);
    return q.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) {
                ggml_sycl_mul_mat_vec_q8_0_kernel(src0, src1, dst, nblocks_per_row, s1, d1, n_cols, item);
            });
}

static void ggml_sycl_op_mul_mat_q8_0(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#ifndef GGML_SYCL_USE_ONEMKL
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("GGML_SYCL q8 mul_mat requires oneMKL");
#else
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    sycl::queue & q = ctx.stream();
    const to_fp32_sycl_t dequantize = ggml_get_to_fp32_sycl(src0->type);
    GGML_ASSERT(dequantize != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);

    const int64_t d1 = dst->nb[1] / sizeof(float);
    const int64_t d2 = dst->nb[2] / sizeof(float);
    const int64_t d3 = dst->nb[3] / sizeof(float);

    const char * src0_base = (const char *) src0->data;
    const float * src1_base = (const float *) src1->data;
    float * dst_base = (float *) dst->data;

    // Decode-dominated case: n RHS columns is small, so skip the F32 dequant scratch + GEMM and
    // read the quantized weights once via a fused dot-product kernel (see
    // ggml_sycl_mul_mat_vec_q8_0 above). Falls back to the general dequant+GEMM path for large n
    // (prefill/batched compute, where oneMKL's tiled GEMM is the better fit) or non-packed rows.
    if (ne11 <= GGML_SYCL_MMVQ_MAX_N_COLS && ggml_sycl_mmf_has_packed_rows(src0) && ne00 % QK8_0 == 0) {
        for (int64_t i3 = 0; i3 < ne13; ++i3) {
            for (int64_t i2 = 0; i2 < ne12; ++i2) {
                const int64_t i03 = i3 / r3;
                const int64_t i02 = i2 / r2;

                const block_q8_0 * src0_i =
                        (const block_q8_0 *) (src0_base + (ptrdiff_t) i02 * src0->nb[2] + (ptrdiff_t) i03 * src0->nb[3]);
                const float * src1_i = src1_base + i2 * s12 + i3 * s13;
                float * dst_i = dst_base + i2 * d2 + i3 * d3;

                ggml_sycl_mul_mat_vec_q8_0(q, src0_i, src1_i, dst_i, ne01, ne00, s11, d1, (int) ne11);
            }
        }
        return;
    }

    float * src0_f32 = ggml_sycl_mmf_reserve_f32_scratch(ctx, GGML_SYCL_MMF_SRC0_F32_SCRATCH, (size_t) (ne00 * ne01));

    int64_t cached_i02 = -1;
    int64_t cached_i03 = -1;

    for (int64_t i3 = 0; i3 < ne13; ++i3) {
        for (int64_t i2 = 0; i2 < ne12; ++i2) {
            const int64_t i03 = i3 / r3;
            const int64_t i02 = i2 / r2;

            if (i02 != cached_i02 || i03 != cached_i03) {
                const void * src0_i = src0_base + (ptrdiff_t) i02 * src0->nb[2] + (ptrdiff_t) i03 * src0->nb[3];
                dequantize(src0_i, src0_f32, ne00 * ne01, q);
                cached_i02 = i02;
                cached_i03 = i03;
            }

            const float * src1_i = src1_base + i2 * s12 + i3 * s13;
            float * dst_i = dst_base + i2 * d2 + i3 * d3;

            ggml_sycl_mmf_gemm(q, src0_f32, src1_i, dst_i, ne01, ne11, ne00, ne00, s11, d1);
        }
    }
#endif
}

static void ggml_sycl_op_mul_mat_id_f32(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#ifndef GGML_SYCL_USE_ONEMKL
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("GGML_SYCL mul_mat_id requires oneMKL");
#else
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    sycl::queue & q = ctx.stream();
    const to_fp32_sycl_t src0_to_f32 = ggml_get_to_fp32_sycl(src0->type);
    GGML_ASSERT(src0_to_f32 != nullptr);

    const int64_t k = src0->ne[0];
    const int64_t m = src0->ne[1];
    const int64_t n_experts = src0->ne[2];
    const int64_t n_src1_cols = src1->ne[1];
    const int64_t n_used = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const int64_t ne_get_rows = n_used * n_tokens;

    std::vector<float> src1_host((size_t) ggml_nelements(src1));
    SYCL_CHECK(q.memcpy(src1_host.data(), src1->data, ggml_nbytes(src1)).wait());

    int32_t * ids_src1 = sycl::malloc_shared<int32_t>(std::max<int64_t>(ne_get_rows, 1), q);
    int32_t * ids_dst = sycl::malloc_shared<int32_t>(std::max<int64_t>(ne_get_rows, 1), q);
    int32_t * expert_bounds = sycl::malloc_shared<int32_t>(std::max<int64_t>(n_experts + 1, 1), q);
    GGML_ASSERT(ids_src1 != nullptr);
    GGML_ASSERT(ids_dst != nullptr);
    GGML_ASSERT(expert_bounds != nullptr);

    const int si1 = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    GGML_ASSERT(sis1 > 0);

    ggml_sycl_launch_mm_ids_helper(
            ctx,
            (const int32_t *) ids->data,
            ids_src1,
            ids_dst,
            expert_bounds,
            (int) n_experts,
            (int) n_tokens,
            (int) n_used,
            (int) n_src1_cols,
            si1,
            sis1,
            false);

    size_t max_group_cols = 0;
    for (int64_t expert = 0; expert < n_experts; ++expert) {
        const int64_t begin = expert_bounds[expert];
        const int64_t end = expert_bounds[expert + 1];
        GGML_ASSERT(end >= begin);
        max_group_cols = std::max(max_group_cols, (size_t) (end - begin));
    }

    std::vector<float> dst_host((size_t) ggml_nelements(dst), 0.0f);
    std::vector<float> src1_group((size_t) k * std::max<size_t>(max_group_cols, 1));
    std::vector<float> dst_group((size_t) m * std::max<size_t>(max_group_cols, 1));

    float * src0_f32 = ggml_sycl_mmf_reserve_f32_scratch(ctx, GGML_SYCL_MMF_SRC0_F32_SCRATCH, (size_t) (k * m));
    float * src1_f32 = ggml_sycl_mmf_reserve_f32_scratch(ctx, GGML_SYCL_MMF_SRC1_F32_SCRATCH, (size_t) (k * std::max<size_t>(max_group_cols, 1)));
    float * dst_f32  = ggml_sycl_mmf_reserve_f32_scratch(ctx, GGML_SYCL_MMF_DST_F32_SCRATCH,  (size_t) (m * std::max<size_t>(max_group_cols, 1)));

    const int64_t dst_s1 = dst->nb[1] / sizeof(float);
    const int64_t dst_s2 = dst->nb[2] / sizeof(float);

    for (int64_t expert = 0; expert < n_experts; ++expert) {
        const int64_t begin = expert_bounds[expert];
        const int64_t end = expert_bounds[expert + 1];
        const size_t group_size = (size_t) (end - begin);
        if (group_size == 0) {
            continue;
        }

        for (size_t col = 0; col < group_size; ++col) {
            const int64_t src1_idx = ids_src1[begin + (int64_t) col];
            const float * src1_col = src1_host.data() + src1_idx * k;
            std::memcpy(src1_group.data() + col * k, src1_col, (size_t) k * sizeof(float));
        }

        SYCL_CHECK(q.memcpy(src1_f32, src1_group.data(), group_size * (size_t) k * sizeof(float)).wait());

        const void * src0_expert = (const char *) src0->data + (ptrdiff_t) expert * src0->nb[2];
        src0_to_f32(src0_expert, src0_f32, k * m, q);
        SYCL_CHECK(ggml_sycl_mmf_gemm(q, src0_f32, src1_f32, dst_f32, m, (int64_t) group_size, k, k, k, m).wait_and_throw());
        SYCL_CHECK(q.memcpy(dst_group.data(), dst_f32, group_size * (size_t) m * sizeof(float)).wait());

        for (size_t col = 0; col < group_size; ++col) {
            const int64_t dst_idx = ids_dst[begin + (int64_t) col];
            const int64_t row = dst_idx / n_used;
            const int64_t slot = dst_idx % n_used;
            float * dst_col = dst_host.data() + slot * dst_s1 + row * dst_s2;
            std::memcpy(dst_col, dst_group.data() + col * m, (size_t) m * sizeof(float));
        }
    }

    SYCL_CHECK(q.memcpy(dst->data, dst_host.data(), ggml_nbytes(dst)).wait());
    sycl::free(ids_src1, q);
    sycl::free(ids_dst, q);
    sycl::free(expert_bounds, q);
#endif
}

template <typename src0_t, typename src1_t>
static void ggml_sycl_op_mul_mat_impl(
        ggml_backend_sycl_context & ctx,
        const src0_t * src0_ptr,
        const src1_t * src1_ptr,
        ggml_tensor * dst,
        int64_t s01,
        int64_t s02,
        int64_t s03,
        int64_t s11,
        int64_t s12,
        int64_t s13) {
#ifndef GGML_SYCL_USE_ONEMKL
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0_ptr);
    GGML_UNUSED(src1_ptr);
    GGML_UNUSED(dst);
    GGML_UNUSED(s01);
    GGML_UNUSED(s02);
    GGML_UNUSED(s03);
    GGML_UNUSED(s11);
    GGML_UNUSED(s12);
    GGML_UNUSED(s13);
    GGML_ABORT("GGML_SYCL mul_mat requires oneMKL");
#else
    sycl::queue & q = ctx.stream();

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t d1 = dst->nb[1] / sizeof(float);
    const int64_t d2 = dst->nb[2] / sizeof(float);
    const int64_t d3 = dst->nb[3] / sizeof(float);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    float * dst_ptr = static_cast<float *>(dst->data);

    std::vector<sycl::event> deps;

    if (r2 == 1 && r3 == 1 && ggml_sycl_mmf_can_use_strided_batch(src0, src1, dst)) {
        const int64_t stride_a = ne02 == 1 ? s03 : s02;
        const int64_t stride_b = ne12 == 1 ? s13 : s12;
        const int64_t stride_c = ne12 == 1 ? d3 : d2;
        deps.push_back(ggml_sycl_mmf_gemm_batch(
                q,
                src0_ptr,
                src1_ptr,
                dst_ptr,
                ne01,
                ne11,
                ne00,
                s01,
                stride_a,
                s11,
                stride_b,
                d1,
                stride_c,
                ne12 * ne13));
    } else {
        for (int64_t i3 = 0; i3 < ne13; ++i3) {
            for (int64_t i2 = 0; i2 < ne12; ++i2) {
                const int64_t i03 = i3 / r3;
                const int64_t i02 = i2 / r2;

                const src0_t * src0_i = src0_ptr + i02 * s02 + i03 * s03;
                const src1_t * src1_i = src1_ptr + i2 * s12 + i3 * s13;
                float * dst_i = dst_ptr + i2 * d2 + i3 * d3;

                deps.clear();
                deps.push_back(ggml_sycl_mmf_gemm(
                        q,
                        src0_i,
                        src1_i,
                        dst_i,
                        ne01,
                        ne11,
                        ne00,
                        s01,
                        s11,
                        d1,
                        deps));
            }
        }
    }
#endif
}

} // namespace

void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_sycl_op_fwht(ctx, src1, dst)) {
        return;
    }

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(src1->ne[2] % src0->ne[2] == 0);
    GGML_ASSERT(src1->ne[3] % src0->ne[3] == 0);

    const int64_t s11_f32 = src1->nb[1] / sizeof(float);
    const int64_t s12_f32 = src1->nb[2] / sizeof(float);
    const int64_t s13_f32 = src1->nb[3] / sizeof(float);

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const int64_t s01 = src0->nb[1] / sizeof(float);
            const int64_t s02 = src0->nb[2] / sizeof(float);
            const int64_t s03 = src0->nb[3] / sizeof(float);

            ggml_sycl_op_mul_mat_impl(
                    ctx,
                    static_cast<const float *>(src0->data),
                    static_cast<const float *>(src1->data),
                    dst,
                    s01, s02, s03,
                    s11_f32, s12_f32, s13_f32);
        } break;
        case GGML_TYPE_F16: {
            const int64_t s01 = src0->nb[1] / sizeof(sycl::half);
            const int64_t s02 = src0->nb[2] / sizeof(sycl::half);
            const int64_t s03 = src0->nb[3] / sizeof(sycl::half);

            sycl::half * src1_f16 = ggml_sycl_mmf_reserve_src1_f16_scratch(ctx, ggml_nelements(src1));
            sycl::queue & q = ctx.stream();
            ggml_sycl_convert_src1_f32_to_f16(q, src1, src1_f16);

            const int64_t s11 = src1->ne[0];
            const int64_t s12 = src1->ne[0] * src1->ne[1];
            const int64_t s13 = s12 * src1->ne[2];

            ggml_sycl_op_mul_mat_impl(
                    ctx,
                    static_cast<const sycl::half *>(src0->data),
                    src1_f16,
                    dst,
                    s01, s02, s03,
                    s11, s12, s13);
        } break;
        case GGML_TYPE_Q8_0:
            ggml_sycl_op_mul_mat_q8_0(ctx, dst);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type %s", __func__, ggml_type_name(src0->type));
    }
}

void ggml_sycl_op_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0 != nullptr);

    switch (src0->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            ggml_sycl_op_mul_mat_id_f32(ctx, dst);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type %s", __func__, ggml_type_name(src0->type));
    }
}

bool ggml_sycl_supports_mul_mat(const ggml_tensor * op) {
#ifndef GGML_SYCL_USE_ONEMKL
    GGML_UNUSED(op);
    return false;
#else
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    const bool dense_ok = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                          src1->type == GGML_TYPE_F32 &&
                          op->type == GGML_TYPE_F32 &&
                          src1->ne[0] == src0->ne[0] &&
                          src1->ne[2] % src0->ne[2] == 0 &&
                          src1->ne[3] % src0->ne[3] == 0;
    const bool q8_ok = src0->type == GGML_TYPE_Q8_0 &&
                       src1->type == GGML_TYPE_F32 &&
                       op->type == GGML_TYPE_F32 &&
                       src1->ne[0] == src0->ne[0] &&
                       src1->ne[2] % src0->ne[2] == 0 &&
                       src1->ne[3] % src0->ne[3] == 0 &&
                       ggml_sycl_mmf_has_packed_rows(src0);
    return dense_ok || q8_ok;
#endif
}

bool ggml_sycl_supports_mul_mat_id(const ggml_tensor * op) {
#ifndef GGML_SYCL_USE_ONEMKL
    GGML_UNUSED(op);
    return false;
#else
    if (op == nullptr || op->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * ids  = op->src[2];

    if (src0 == nullptr || src1 == nullptr || ids == nullptr) {
        return false;
    }

    return ggml_get_to_fp32_sycl(src0->type) != nullptr &&
           src1->type == GGML_TYPE_F32 &&
           ids->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[3] == 1 &&
           src1->ne[3] == 1 &&
           ids->ne[2] == 1 &&
           ids->ne[3] == 1 &&
           ids->ne[1] == src1->ne[2] &&
           src0->ne[0] == src1->ne[0] &&
           ids->ne[0] % src1->ne[1] == 0 &&
           ggml_sycl_mmf_has_packed_rows(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
#endif
}
