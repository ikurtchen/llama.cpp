#include "roll.hpp"

#include <sycl/sycl.hpp>

namespace {
inline int64_t wrap_index(int64_t idx, int64_t ne) {
    if (idx < 0) {
        return idx + ne;
    }
    if (idx >= ne) {
        return idx - ne;
    }
    return idx;
}
}

void ggml_sycl_op_roll(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int s0 = dst->op_params[0];
    const int s1 = dst->op_params[1];
    const int s2 = dst->op_params[2];
    const int s3 = dst->op_params[3];

    const float * src = (const float *) src0->data;
    float * out = (float *) dst->data;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t total = ggml_nelements(src0);

    sycl::queue & q = ctx.stream();
    const int64_t block = 256;
    const int64_t nblk  = (total + block - 1) / block;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t idx = item.get_global_id(0);
        if (idx >= total) {
            return;
        }

        const int64_t i0 = idx % ne00;
        const int64_t i1 = (idx / ne00) % ne01;
        const int64_t i2 = (idx / (ne00 * ne01)) % ne02;
        const int64_t i3 = idx / (ne00 * ne01 * ne02);

        const int64_t d0 = wrap_index(i0 - s0, ne00);
        const int64_t d1 = wrap_index(i1 - s1, ne01);
        const int64_t d2 = wrap_index(i2 - s2, ne02);
        const int64_t d3 = wrap_index(i3 - s3, ne03);

        out[idx] = src[((d3 * ne02 + d2) * ne01 + d1) * ne00 + d0];
    });
}

bool ggml_sycl_supports_roll(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0);
}
