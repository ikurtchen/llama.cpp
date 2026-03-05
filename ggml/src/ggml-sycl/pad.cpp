#include "pad.hpp"

#include <stdint.h>

void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    // Extract padding params from op_params
    const int32_t lp0      = ((const int32_t *) dst->op_params)[0];
    const int32_t rp0      = ((const int32_t *) dst->op_params)[1];
    const int32_t lp1      = ((const int32_t *) dst->op_params)[2];
    const int32_t rp1      = ((const int32_t *) dst->op_params)[3];
    const int32_t lp2      = ((const int32_t *) dst->op_params)[4];
    const int32_t rp2      = ((const int32_t *) dst->op_params)[5];
    const int32_t lp3      = ((const int32_t *) dst->op_params)[6];
    const int32_t rp3      = ((const int32_t *) dst->op_params)[7];
    const bool    circular = (bool) ((const int32_t *) dst->op_params)[8];

    // Source strides in elements (byte strides / sizeof(float))
    const size_t s00 = src0->nb[0] / ggml_type_size(src0->type);
    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    // Output dimensions
    const int ne0 = (int) dst->ne[0];
    const int ne1 = (int) dst->ne[1];
    const int ne2 = (int) dst->ne[2];
    const int ne3 = (int) dst->ne[3];

    // Launch configuration matching CUDA: gridDim(ceil(ne0/256), ne1, ne2*ne3), blockDim=256
    const int block_size = SYCL_PAD_BLOCK_SIZE;
    const int num_blocks_x = (ne0 + block_size - 1) / block_size;

    sycl::range<3> global_range(ne2 * ne3, ne1, num_blocks_x * block_size);
    sycl::range<3> local_range(1, 1, block_size);

    stream->parallel_for(
        sycl::nd_range<3>(global_range, local_range),
        [=](sycl::nd_item<3> item) {
            const int i0 = item.get_local_id(2) + item.get_group(2) * item.get_local_range(2);
            const int i1 = item.get_group(1);
            const int i2 = item.get_group(0) % ne2;
            const int i3 = item.get_group(0) / ne2;

            if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
                return;
            }

            const int64_t dst_idx = (int64_t) i3 * (ne0 * ne1 * ne2)
                                  + (int64_t) i2 * (ne0 * ne1)
                                  + (int64_t) i1 * ne0 + i0;

            if (!circular) {
                if ((i0 >= lp0 && i0 < ne0 - rp0) &&
                    (i1 >= lp1 && i1 < ne1 - rp1) &&
                    (i2 >= lp2 && i2 < ne2 - rp2) &&
                    (i3 >= lp3 && i3 < ne3 - rp3)) {
                    const int64_t i00 = i0 - lp0;
                    const int64_t i01 = i1 - lp1;
                    const int64_t i02 = i2 - lp2;
                    const int64_t i03 = i3 - lp3;
                    const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;
                    dst_d[dst_idx] = src0_d[src_idx];
                } else {
                    dst_d[dst_idx] = 0.0f;
                }
            } else {
                const int64_t ne00 = ne0 - lp0 - rp0;
                const int64_t ne01 = ne1 - lp1 - rp1;
                const int64_t ne02 = ne2 - lp2 - rp2;
                const int64_t ne03 = ne3 - lp3 - rp3;

                // wrap_around: (coord + size) % size — handles negatives
                const int64_t i00 = ((int64_t)(i0 - lp0) + ne00) % ne00;
                const int64_t i01 = ((int64_t)(i1 - lp1) + ne01) % ne01;
                const int64_t i02 = ((int64_t)(i2 - lp2) + ne02) % ne02;
                const int64_t i03 = ((int64_t)(i3 - lp3) + ne03) % ne03;
                const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;
                dst_d[dst_idx] = src0_d[src_idx];
            }
        });
}
