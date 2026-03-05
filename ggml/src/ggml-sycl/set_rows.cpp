#include "set_rows.hpp"

#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
#endif
using sycl_bf16 = sycl::ext::oneapi::bfloat16;

// Kernel for float dst types (F32, F16, BF16)
template <typename src_t, typename idx_t, typename dst_t>
static void k_set_rows(const src_t * __restrict__ src0,
                       const idx_t * __restrict__ src1,
                       dst_t * __restrict__ dst,
                       const int64_t ne_total,
                       const int64_t ne00, const int64_t ne01, const int64_t ne02,
                       const int64_t s01, const int64_t s02, const int64_t s03,
                       const int64_t s10, const int64_t s11, const int64_t s12,
                       const int64_t s1, const int64_t s2, const int64_t s3,
                       const int64_t ne11, const int64_t ne12,
                       const sycl::nd_item<1> & item) {
    const int64_t i = item.get_global_id(0);
    if (i >= ne_total) {
        return;
    }

    // Decompose flat index into multi-dimensional indices
    const int64_t i00 = i % ne00;
    const int64_t i01 = (i / ne00) % ne01;
    const int64_t i02 = (i / (ne00 * ne01)) % ne02;
    const int64_t i03 = i / (ne00 * ne01 * ne02);

    // Map to src1 indices with modular wrapping
    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = (dst_t) src0_row[i00];
}

// Launch helper for a specific (src_t, idx_t, dst_t) combination
template <typename src_t, typename idx_t, typename dst_t>
static void set_rows_sycl(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        sycl::queue & stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    if (ne_total <= 0 || ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne11 <= 0 || ne12 <= 0) {
        return;
    }

    const int64_t s01 = nb01 / sizeof(src_t);
    const int64_t s02 = nb02 / sizeof(src_t);
    const int64_t s03 = nb03 / sizeof(src_t);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);
    const int64_t s1  = nb1  / sizeof(dst_t);
    const int64_t s2  = nb2  / sizeof(dst_t);
    const int64_t s3  = nb3  / sizeof(dst_t);

    const int num_blocks = (ne_total + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_SET_BLOCK_SIZE, SYCL_SET_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            k_set_rows(src0_d, src1_d, dst_d, ne_total,
                       ne00, ne01, ne02,
                       s01, s02, s03,
                       s10, s11, s12,
                       s1, s2, s3,
                       ne11, ne12, item);
        });
}

// Dispatch over dst type for a given (src_t=float, idx_t)
template <typename idx_t>
static void set_rows_sycl(ggml_backend_sycl_context & ctx,
                           const ggml_tensor * src0, const ggml_tensor * src1,
                           ggml_tensor * dst) {
    const float * src0_d = (const float *) src0->data;
    const idx_t * src1_d = (const idx_t *) src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    sycl::queue & stream = *(ctx.stream());

    if (dst->type == GGML_TYPE_F32) {
        set_rows_sycl(
            src0_d, src1_d, (float *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_sycl(
            src0_d, src1_d, (sycl::half *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_sycl(
            src0_d, src1_d, (sycl_bf16 *) dst->data,
            ne00, ne01, ne02, ne03,
            ne11, ne12,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream);
    } else {
        GGML_ABORT("set_rows: unsupported dst type %s", ggml_type_name(dst->type));
    }
}

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_sycl<int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_sycl<int32_t>(ctx, src0, src1, dst);
    }
}
