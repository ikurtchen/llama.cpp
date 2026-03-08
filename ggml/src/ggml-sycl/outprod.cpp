#include "outprod.hpp"

void ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols

    // Batch dimension broadcasting (GQA support)
    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    const float* src0_d = (const float*)src0->data;
    const float* src1_d = (const float*)src1->data;
    float* dst_d = (float*)dst->data;

    dpct::queue_ptr stream = ctx.stream();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Stride-based leading dimensions (handles non-contiguous sub-views)
    const int64_t lda = nb01 / sizeof(float);
    const int64_t ldc = nb1  / sizeof(float);

    // Handle transposition of src1
    const bool src1_T = ggml_is_transposed(src1);
    const oneapi::mkl::transpose src1_op = src1_T ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    const int64_t ldb = (src1_T ? nb10 : nb11) / sizeof(float);
    GGML_ASSERT((src1_T ? nb11 : nb10) == sizeof(float));

    // Data strides in dimensions 2/3 (in float elements)
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s12 = nb12 / sizeof(float);
    const int64_t s13 = nb13 / sizeof(float);
    const int64_t s2  = nb2  / sizeof(float);
    const int64_t s3  = nb3  / sizeof(float);

    // GQA broadcasting ratios: dst-per-src0
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    try {
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::nontrans, src1_op,
                    ne0, ne1, ne01,
                    alpha, src0_d + (i3/dps3)*s03 + (i2/dps2)*s02, lda,
                           src1_d +  i3      *s13 +  i2      *s12, ldb,
                    beta,  dst_d  +  i3      *s3  +  i2      *s2,  ldc);
            }
        }
    }
    catch (sycl::exception const& exc) {
        std::cerr << exc.what() << std::endl;
        GGML_ASSERT(false);
    }
}
