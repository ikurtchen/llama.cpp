#include "pad.hpp"

#include <sycl/sycl.hpp>

namespace {
inline int64_t wrap_around(int64_t coord, int64_t size) {
    return (coord + size) % size;
}
}

void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t * params = (const int32_t *) dst->op_params;
    const int32_t lp0 = params[0];
    const int32_t rp0 = params[1];
    const int32_t lp1 = params[2];
    const int32_t rp1 = params[3];
    const int32_t lp2 = params[4];
    const int32_t rp2 = params[5];
    const int32_t lp3 = params[6];
    const int32_t rp3 = params[7];
    const bool circular = params[8] != 0;

    const char * src = (const char *) src0->data;
    char * dst_data  = (char *) dst->data;
    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0];
    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];
    const size_t nb3  = dst->nb[3];
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    const int64_t total = ggml_nelements(dst);

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

        float value = 0.0f;
        if (!circular) {
            if (i0 >= lp0 && i0 < ne0 - rp0 &&
                i1 >= lp1 && i1 < ne1 - rp1 &&
                i2 >= lp2 && i2 < ne2 - rp2 &&
                i3 >= lp3 && i3 < ne3 - rp3) {
                const int64_t s0 = i0 - lp0;
                const int64_t s1 = i1 - lp1;
                const int64_t s2 = i2 - lp2;
                const int64_t s3 = i3 - lp3;
                value = *(const float *) (src + s3 * nb03 + s2 * nb02 + s1 * nb01 + s0 * nb00);
            }
        } else {
            const int64_t ne00 = ne0 - lp0 - rp0;
            const int64_t ne01 = ne1 - lp1 - rp1;
            const int64_t ne02 = ne2 - lp2 - rp2;
            const int64_t ne03 = ne3 - lp3 - rp3;
            const int64_t s0 = wrap_around(i0 - lp0, ne00);
            const int64_t s1 = wrap_around(i1 - lp1, ne01);
            const int64_t s2 = wrap_around(i2 - lp2, ne02);
            const int64_t s3 = wrap_around(i3 - lp3, ne03);
            value = *(const float *) (src + s3 * nb03 + s2 * nb02 + s1 * nb01 + s0 * nb00);
        }

        *(float *) (dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0) = value;
    });
}

bool ggml_sycl_supports_pad(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
}
