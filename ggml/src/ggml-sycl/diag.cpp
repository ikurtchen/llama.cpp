#include "diag.hpp"

#include <sycl/sycl.hpp>

namespace {
template <typename T>
void diag_kernel(sycl::queue & q, T * dst, const T * src, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                 int64_t total) {
    const int64_t block = 256;
    const int64_t nblk  = (total + block - 1) / block;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t idx = item.get_global_id(0);
        if (idx >= total) {
            return;
        }

        const int64_t i0 = idx % ne0;
        const int64_t i1 = (idx / ne0) % ne1;
        const int64_t i2 = (idx / (ne0 * ne1)) % ne2;
        const int64_t i3 = idx / (ne0 * ne1 * ne2);

        if (i0 == i1) {
            const int64_t batch = i3 * ne2 + i2;
            dst[idx] = src[batch * ne0 + i0];
        } else {
            dst[idx] = (T) 0;
        }
    });
}
}

void ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[0] == dst->ne[0]);
    GGML_ASSERT(src0->ne[1] == 1);
    GGML_ASSERT(src0->ne[2] == dst->ne[2]);
    GGML_ASSERT(src0->ne[3] == dst->ne[3]);
    GGML_ASSERT(src0->type == dst->type);

    const int64_t total = ggml_nelements(dst);
    sycl::queue & q = ctx.stream();

    switch (dst->type) {
        case GGML_TYPE_F32:
            diag_kernel(q, (float *) dst->data, (const float *) src0->data,
                        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], total);
            break;
        case GGML_TYPE_F16:
            diag_kernel(q, (sycl::half *) dst->data, (const sycl::half *) src0->data,
                        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], total);
            break;
        default:
            GGML_ABORT("%s: unsupported type %s", __func__, ggml_type_name(dst->type));
    }
}

bool ggml_sycl_supports_diag(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return ggml_is_contiguous(op) && ggml_is_contiguous(src0) && op->type == src0->type &&
           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
}
