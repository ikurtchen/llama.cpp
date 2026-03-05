#include "pad_reflect_1d.hpp"

#include <stdint.h>

void ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0     = dst->src[0];
    const void *        src0_dd  = src0->data;
    void *              dst_dd   = dst->data;
    dpct::queue_ptr     stream   = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int       p0   = opts[0];
    const int       p1   = opts[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne0 = dst->ne[0];

    const int64_t nb0 = dst->nb[0];
    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    // Sanity: padded length matches
    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    constexpr int64_t bx     = SYCL_PAD_REFLECT_1D_BLOCK_SIZE;
    const int64_t     tiles0 = (ne0 + bx - 1) / bx;

    // grid: dim0=i3(ne03), dim1=i2(ne02), dim2=combined i1*tiles0 (with local size bx)
    sycl::range<3> global_range(ne03, ne02, ne01 * tiles0 * bx);
    sycl::range<3> local_range(1, 1, bx);

    // Capture all tensor metadata by value for the kernel lambda.
    // Do NOT capture ggml_tensor* pointers — they are host pointers.
    const char * src0_ptr_base = (const char *) src0_dd;
    char *       dst_ptr_base  = (char *) dst_dd;

    stream->parallel_for(
        sycl::nd_range<3>(global_range, local_range),
        [=](sycl::nd_item<3> item) {
            const int64_t i3 = item.get_group(0);
            const int64_t i2 = item.get_group(1);

            const int64_t group_x = item.get_group(2);
            const int64_t i1      = group_x / tiles0;
            const int64_t tile0   = group_x % tiles0;
            const int64_t i0      = item.get_local_id(2) + tile0 * bx;

            if (i0 >= ne0) {
                return;
            }

            const char * src0_ptr = src0_ptr_base + i3 * nb03 + i2 * nb02 + i1 * nb01;
            char *       dst_ptr  = dst_ptr_base  + i3 * nb3  + i2 * nb2  + i1 * nb1;

            const int64_t rel_i0 = i0 - p0;
            int64_t       src_idx;

            if (rel_i0 < 0) {
                // Left padding — reflect
                src_idx = -rel_i0;
            } else if (rel_i0 < ne00) {
                // Middle — copy
                src_idx = rel_i0;
            } else {
                // Right padding — reflect
                src_idx = 2 * ne00 - 2 - rel_i0;
            }

            const float value               = *(const float *) (src0_ptr + src_idx * nb00);
            *(float *) (dst_ptr + i0 * nb0)  = value;
        });
}
