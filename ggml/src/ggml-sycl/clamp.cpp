// SYCL clamp op: dst = min(max(src0, lo), hi), element-wise, F32 or F16. Mirrors
// ggml-cuda/clamp.cu's algorithm but written directly against SYCL.
#include "clamp.hpp"

#include <sycl/sycl.hpp>

template <typename T>
static void clamp_impl(sycl::queue & q, const T * x, T * dst, float lo, float hi, int64_t k) {
    const int64_t block = 256;
    const int64_t nblk   = (k + block - 1) / block;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= k) {
            return;
        }
        const float v = (float) x[i];
        dst[i] = (T) sycl::fmin(sycl::fmax(v, lo), hi);
    });
}

void ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == src0->type);

    float lo, hi;
    memcpy(&lo, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&hi, (float *) dst->op_params + 1, sizeof(float));

    sycl::queue & q  = ctx.stream();
    const int64_t ne = ggml_nelements(src0);

    if (src0->type == GGML_TYPE_F16) {
        clamp_impl<sycl::half>(q, (const sycl::half *) src0->data, (sycl::half *) dst->data, lo, hi, ne);
    } else {
        clamp_impl<float>(q, (const float *) src0->data, (float *) dst->data, lo, hi, ne);
    }
}

bool ggml_sycl_supports_clamp(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) && op->type == src0->type;
}
