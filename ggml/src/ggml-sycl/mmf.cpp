#include "mmf.hpp"
#include "fwht.hpp"

#ifdef GGML_SYCL_USE_ONEMKL
#include <oneapi/mkl.hpp>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>

namespace {

struct ggml_sycl_mmf_src1_f16_scratch {
    sycl::half * ptr = nullptr;
    size_t       capacity = 0;
};

static ggml_sycl_mmf_src1_f16_scratch & ggml_sycl_mmf_get_src1_f16_scratch(int device) {
    static std::array<ggml_sycl_mmf_src1_f16_scratch, GGML_SYCL_MAX_DEVICES> scratch = {};
    GGML_ASSERT(device >= 0 && device < GGML_SYCL_MAX_DEVICES);
    return scratch[device];
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

    const bool src0_ok = src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
    return src0_ok &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src1->ne[0] == src0->ne[0] &&
           src1->ne[2] % src0->ne[2] == 0 &&
           src1->ne[3] % src0->ne[3] == 0;
#endif
}
