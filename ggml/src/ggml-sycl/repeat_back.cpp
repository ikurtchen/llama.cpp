#include "repeat_back.hpp"

#include <algorithm>
#include <cstdint>

// ============================================================================
// repeat_back kernel — sums over repeated dimensions
// Grid: (ceil(ne0/WARP_SIZE), ne1, ne2*ne3)
// ============================================================================

template <typename T>
static void k_repeat_back(const T * __restrict__ src, T * __restrict__ dst,
                          const int64_t ne00, const int64_t ne01,
                          const int64_t ne02, const int64_t ne03,
                          const size_t s00, const size_t s01,
                          const size_t s02, const size_t s03,
                          const int64_t ne0, const int64_t ne1,
                          const int64_t ne2, const int64_t ne3,
                          const sycl::nd_item<3> & item) {

    const int64_t tid0  = (int64_t)item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    const int64_t tid1  = (int64_t)item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    const int64_t tid23 = (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    const int64_t tid2  = tid23 % ne2;
    const int64_t tid3  = tid23 / ne2;

    if (tid0 >= ne0) {
        return;
    }

    T sum = 0;
    for (int64_t i3 = tid3; i3 < ne03; i3 += ne3) {
        for (int64_t i2 = tid2; i2 < ne02; i2 += ne2) {
            for (int64_t i1 = tid1; i1 < ne01; i1 += ne1) {
                for (int64_t i0 = tid0; i0 < ne00; i0 += ne0) {
                    sum += src[i3*s03 + i2*s02 + i1*s01 + i0*s00];
                }
            }
        }
    }
    dst[tid3*ne2*ne1*ne0 + tid2*ne1*ne0 + tid1*ne0 + tid0] = sum;
}

template <typename T>
static void repeat_back_sycl(
        const T * src, T * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const size_t s00, const size_t s01, const size_t s02, const size_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        sycl::queue & stream) {

    // block_dims = (WARP_SIZE, 1, 1) → SYCL range<3>(z=1, y=1, x=WARP_SIZE)
    sycl::range<3> block_dims(1, 1, WARP_SIZE);
    unsigned int gx = (ne0 + WARP_SIZE - 1) / WARP_SIZE;
    sycl::range<3> grid_dims(ne2 * ne3, ne1, gx * WARP_SIZE);

    stream.parallel_for(
        sycl::nd_range<3>(grid_dims, block_dims),
        [=](sycl::nd_item<3> item) {
            k_repeat_back(src, dst, ne00, ne01, ne02, ne03,
                          s00, s01, s02, s03, ne0, ne1, ne2, ne3, item);
        });
}

void ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_can_repeat(dst, src0));

    sycl::queue & stream = *(ctx.stream());

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(ne2*ne3 <= (1 << 15));

    const size_t ts  = ggml_type_size(src0->type);
    const size_t s00 = nb00 / ts;
    const size_t s01 = nb01 / ts;
    const size_t s02 = nb02 / ts;
    const size_t s03 = nb03 / ts;

    switch (dst->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            float       * dst_d  = (float       *) dst->data;
            repeat_back_sycl(src0_d, dst_d, ne00, ne01, ne02, ne03,
                             s00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
        } break;
        default: {
            GGML_ASSERT(false);
        } break;
    }
}
