#include "roll.hpp"
#include "presets.hpp"

static inline int64_t wrap_index(const int64_t idx, const int64_t ne) {
    if (idx < 0) {
        return idx + ne;
    }
    if (idx >= ne) {
        return idx - ne;
    }
    return idx;
}

static void roll_f32_sycl(const float * src, float * dst,
                           const int64_t ne00, const int64_t ne01,
                           const int64_t ne02, const int64_t ne03,
                           const int s0, const int s1,
                           const int s2, const int s3,
                           sycl::queue & stream) {
    const int64_t n_elements = ne00 * ne01 * ne02 * ne03;
    const int64_t num_blocks = (n_elements + SYCL_ROLL_BLOCK_SIZE - 1) / SYCL_ROLL_BLOCK_SIZE;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_ROLL_BLOCK_SIZE, SYCL_ROLL_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            const int64_t idx = item.get_global_id(0);

            if (idx >= n_elements) {
                return;
            }

            const int64_t i0 = idx % ne00;
            const int64_t i1 = (idx / ne00) % ne01;
            const int64_t i2 = (idx / (ne00 * ne01)) % ne02;
            const int64_t i3 = (idx / (ne00 * ne01 * ne02)) % ne03;

            const int64_t d0 = wrap_index(i0 - s0, ne00);
            const int64_t d1 = wrap_index(i1 - s1, ne01);
            const int64_t d2 = wrap_index(i2 - s2, ne02);
            const int64_t d3 = wrap_index(i3 - s3, ne03);

            dst[i3 * (ne00 * ne01 * ne02) + i2 * (ne01 * ne00) + i1 * ne00 + i0] =
                src[d3 * (ne00 * ne01 * ne02) + d2 * (ne01 * ne00) + d1 * ne00 + d0];
        }
    );
}

void ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int s0 = dst->op_params[0];
    const int s1 = dst->op_params[1];
    const int s2 = dst->op_params[2];
    const int s3 = dst->op_params[3];

    sycl::queue & stream = *(ctx.stream());

    roll_f32_sycl(src0_d, dst_d, ne00, ne01, ne02, ne03, s0, s1, s2, s3, stream);
}
