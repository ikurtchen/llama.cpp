#include "pad_reflect_1d.hpp"

#include <sycl/sycl.hpp>

void ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int p0 = opts[0];
    const int p1 = opts[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne0  = dst->ne[0];
    const int64_t total = ggml_nelements(dst);

    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    const char * src = (const char *) src0->data;
    char * dst_data  = (char *) dst->data;
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0];
    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];
    const size_t nb3  = dst->nb[3];

    sycl::queue & q = ctx.stream();
    const int64_t block = 256;
    const int64_t nblk  = (total + block - 1) / block;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t idx = item.get_global_id(0);
        if (idx >= total) {
            return;
        }

        int64_t rem = idx;
        const int64_t i0 = rem % ne0;
        rem /= ne0;
        const int64_t i1 = rem % ne1;
        rem /= ne1;
        const int64_t i2 = rem % ne2;
        const int64_t i3 = rem / ne2;

        const int64_t rel_i0 = i0 - p0;
        int64_t src_idx = rel_i0;
        if (rel_i0 < 0) {
            src_idx = -rel_i0;
        } else if (rel_i0 >= ne00) {
            src_idx = 2 * ne00 - 2 - rel_i0;
        }

        const float value = *(const float *) (src + i3 * nb03 + i2 * nb02 + i1 * nb01 + src_idx * nb00);
        *(float *) (dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0) = value;
    });
}

bool ggml_sycl_supports_pad_reflect_1d(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
}
