#include "pad.hpp"

#include <stdint.h>

// Non-circular pad kernel: uses int dimension-product addressing (baseline-style)
// All source index math is 32-bit int for Xe2 performance.
static void pad_f32(const float * src, float * dst,
                    const int lp0, const int rp0, const int lp1, const int rp1,
                    const int lp2, const int rp2, const int lp3, const int rp3,
                    const int ne0, const int ne1, const int ne2, const int ne3,
                    sycl::nd_item<3> item_ct1) {
    int i0 = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int i1 = item_ct1.get_group(1);
    int i2 = item_ct1.get_group(0) % ne2;
    int i3 = item_ct1.get_group(0) / ne2;
    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    if ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        const int i00 = i0 - lp0;
        const int i01 = i1 - lp1;
        const int i02 = i2 - lp2;
        const int i03 = i3 - lp3;
        const int ne00 = ne0 - lp0 - rp0;
        const int ne01 = ne1 - lp1 - rp1;
        const int ne02 = ne2 - lp2 - rp2;

        const int src_idx = i03 * (ne00 * ne01 * ne02) +
                            i02 * (ne00 * ne01) + i01 * ne00 + i00;

        dst[dst_idx] = src[src_idx];
    } else {
        dst[dst_idx] = 0.0f;
    }
}

// Non-circular pad kernel with stride-based addressing for non-contiguous sources
static void pad_f32_strided(const float * src, float * dst,
                            const int lp0, const int rp0, const int lp1, const int rp1,
                            const int lp2, const int rp2, const int lp3, const int rp3,
                            const int ne0, const int ne1, const int ne2, const int ne3,
                            const int s00, const int s01, const int s02, const int s03,
                            sycl::nd_item<3> item_ct1) {
    int i0 = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int i1 = item_ct1.get_group(1);
    int i2 = item_ct1.get_group(0) % ne2;
    int i3 = item_ct1.get_group(0) / ne2;
    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    if ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        const int i00 = i0 - lp0;
        const int i01 = i1 - lp1;
        const int i02 = i2 - lp2;
        const int i03 = i3 - lp3;
        const int src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;
        dst[dst_idx] = src[src_idx];
    } else {
        dst[dst_idx] = 0.0f;
    }
}

// Circular pad kernel: uses stride-based addressing to handle non-contiguous sources
static void pad_f32_circular(const float * src, float * dst,
                             const int lp0, const int rp0, const int lp1, const int rp1,
                             const int lp2, const int rp2, const int lp3, const int rp3,
                             const int ne0, const int ne1, const int ne2, const int ne3,
                             const int s00, const int s01, const int s02, const int s03,
                             sycl::nd_item<3> item_ct1) {
    int i0 = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int i1 = item_ct1.get_group(1);
    int i2 = item_ct1.get_group(0) % ne2;
    int i3 = item_ct1.get_group(0) / ne2;
    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;

    const int ne00 = ne0 - lp0 - rp0;
    const int ne01 = ne1 - lp1 - rp1;
    const int ne02 = ne2 - lp2 - rp2;
    const int ne03 = ne3 - lp3 - rp3;

    // wrap_around: (coord + size) % size — handles negatives
    const int i00 = ((i0 - lp0) + ne00) % ne00;
    const int i01 = ((i1 - lp1) + ne01) % ne01;
    const int i02 = ((i2 - lp2) + ne02) % ne02;
    const int i03 = ((i3 - lp3) + ne03) % ne03;
    const int src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;
    dst[dst_idx] = src[src_idx];
}

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

    // Output dimensions
    const int ne0 = (int) dst->ne[0];
    const int ne1 = (int) dst->ne[1];
    const int ne2 = (int) dst->ne[2];
    const int ne3 = (int) dst->ne[3];

    // Launch configuration
    const int num_blocks_x = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;

    sycl::range<3> global_range(ne2 * ne3, ne1, num_blocks_x * SYCL_PAD_BLOCK_SIZE);
    sycl::range<3> local_range(1, 1, SYCL_PAD_BLOCK_SIZE);
    sycl::nd_range<3> nd_range(global_range, local_range);

    if (!circular && ggml_is_contiguous(src0)) {
        // Fast path: contiguous source, int-based dimension-product addressing
        stream->parallel_for(nd_range, [=](sycl::nd_item<3> item) {
            pad_f32(src0_d, dst_d, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, item);
        });
    } else {
        // Stride-based addressing (int) for non-contiguous or circular padding
        const int s00 = (int)(src0->nb[0] / ggml_type_size(src0->type));
        const int s01 = (int)(src0->nb[1] / ggml_type_size(src0->type));
        const int s02 = (int)(src0->nb[2] / ggml_type_size(src0->type));
        const int s03 = (int)(src0->nb[3] / ggml_type_size(src0->type));
        if (circular) {
            stream->parallel_for(nd_range, [=](sycl::nd_item<3> item) {
                pad_f32_circular(src0_d, dst_d, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                                 ne0, ne1, ne2, ne3, s00, s01, s02, s03, item);
            });
        } else {
            stream->parallel_for(nd_range, [=](sycl::nd_item<3> item) {
                pad_f32_strided(src0_d, dst_d, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                                ne0, ne1, ne2, ne3, s00, s01, s02, s03, item);
            });
        }
    }
}
