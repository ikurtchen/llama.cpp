#include "out-prod.hpp"

#ifdef GGML_SYCL_USE_ONEMKL
#include <oneapi/mkl.hpp>
#endif

#include <limits>

void ggml_sycl_out_prod(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#ifndef GGML_SYCL_USE_ONEMKL
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("GGML_SYCL out_prod requires oneMKL");
#else
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);
    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    sycl::queue & q = ctx.stream();

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float * dst_d = (float *) dst->data;

    const int64_t lda = nb01 / sizeof(float);
    const int64_t ldc = nb1 / sizeof(float);

    const bool src1_t = ggml_is_transposed(src1);
    const oneapi::mkl::transpose src1_op = src1_t ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    const int64_t ldb = (src1_t ? nb10 : nb11) / sizeof(float);
    GGML_ASSERT((src1_t ? nb11 : nb10) == sizeof(float));

    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s12 = nb12 / sizeof(float);
    const int64_t s13 = nb13 / sizeof(float);
    const int64_t s2  = nb2 / sizeof(float);
    const int64_t s3  = nb3 / sizeof(float);

    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    if (dps2 == 1 && ne2 > 1) {
        GGML_ASSERT(ne2 <= std::numeric_limits<int>::max());
        const int64_t batch_count = ne2;

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            oneapi::mkl::blas::column_major::gemm_batch(
                    q,
                    oneapi::mkl::transpose::nontrans,
                    src1_op,
                    ne0, ne1, ne01,
                    1.0f,
                    src0_d + (i3 / dps3) * s03, lda, s02,
                    src1_d + i3 * s13, ldb, s12,
                    0.0f,
                    dst_d + i3 * s3, ldc, s2,
                    batch_count);
        }
    } else {
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const float * src0_i = src0_d + (i3 / dps3) * s03 + (i2 / dps2) * s02;
                const float * src1_i = src1_d + i3 * s13 + i2 * s12;
                float * dst_i = dst_d + i3 * s3 + i2 * s2;

                oneapi::mkl::blas::column_major::gemm(
                        q,
                        oneapi::mkl::transpose::nontrans,
                        src1_op,
                        ne0, ne1, ne01,
                        1.0f,
                        src0_i, lda,
                        src1_i, ldb,
                        0.0f,
                        dst_i, ldc);
            }
        }
    }
#endif
}

bool ggml_sycl_supports_out_prod(const ggml_tensor * op) {
    return op->type == GGML_TYPE_F32 &&
           op->src[0]->type == GGML_TYPE_F32 &&
           op->src[1]->type == GGML_TYPE_F32;
}
