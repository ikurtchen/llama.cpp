#include "fill.hpp"

#include <sycl/sycl.hpp>

namespace {
template <typename T>
void fill_kernel(sycl::queue & q, T * dst, int64_t ne, T value) {
    const int64_t block = 256;
    const int64_t nblk  = (ne + block - 1) / block;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= ne) {
            return;
        }
        dst[i] = value;
    });
}
}

void ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    float value;
    memcpy(&value, dst->op_params, sizeof(value));

    const int64_t ne = ggml_nelements(dst);
    sycl::queue & q = ctx.stream();

    switch (dst->type) {
        case GGML_TYPE_F32:
            fill_kernel(q, (float *) dst->data, ne, value);
            break;
        case GGML_TYPE_F16:
            fill_kernel(q, (sycl::half *) dst->data, ne, (sycl::half) value);
            break;
        default:
            GGML_ABORT("%s: unsupported type %s", __func__, ggml_type_name(dst->type));
    }
}

bool ggml_sycl_supports_fill(const ggml_tensor * op) {
    return ggml_is_contiguous(op) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
}
