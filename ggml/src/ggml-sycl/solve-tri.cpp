#include "solve-tri.hpp"

#ifdef GGML_SYCL_USE_ONEMKL
#include <oneapi/mkl.hpp>
#endif

void ggml_sycl_op_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#ifndef GGML_SYCL_USE_ONEMKL
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("GGML_SYCL solve_tri requires oneMKL");
#else
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n = src0->ne[0];
    const int64_t k = src1->ne[0];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    GGML_ASSERT(src0->ne[1] == n);
    GGML_ASSERT(src1->ne[1] == n);
    GGML_ASSERT(src1->ne[2] == ne02);
    GGML_ASSERT(src1->ne[3] == ne03);

    const float * A = (const float *) src0->data;
    const float * B = (const float *) src1->data;
    float * X = (float *) dst->data;

    const int64_t s02 = src0->nb[2] / sizeof(float);
    const int64_t s03 = src0->nb[3] / sizeof(float);
    const int64_t s2 = dst->nb[2] / sizeof(float);
    const int64_t s3 = dst->nb[3] / sizeof(float);

    sycl::queue & q = ctx.stream();
    SYCL_CHECK(q.memcpy(X, B, ggml_nbytes(dst)));

    for (int64_t i3 = 0; i3 < ne03; ++i3) {
        for (int64_t i2 = 0; i2 < ne02; ++i2) {
            const float * A_i = A + i2 * s02 + i3 * s03;
            float * X_i = X + i2 * s2 + i3 * s3;

            oneapi::mkl::blas::column_major::trsm(
                    q,
                    oneapi::mkl::side::right,
                    oneapi::mkl::uplo::upper,
                    oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::diag::nonunit,
                    k, n,
                    1.0f,
                    A_i, n,
                    X_i, k);
        }
    }
#endif
}

bool ggml_sycl_supports_solve_tri(const ggml_tensor * op) {
    return op->type == GGML_TYPE_F32 &&
           op->src[0]->type == GGML_TYPE_F32 &&
           op->src[1]->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(op->src[0]) &&
           ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op);
}
