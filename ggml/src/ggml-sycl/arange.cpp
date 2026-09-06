// SYCL arange op: dst[i] = start + step * i, for i in [0, ne0). Mirrors ggml-cuda/arange.cu's
// algorithm but written directly against SYCL.
#include "arange.hpp"

#include <sycl/sycl.hpp>
#include <cmath>

void ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float start, stop, step;
    memcpy(&start, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&stop,  (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&step,  (const float *) dst->op_params + 2, sizeof(float));

    const int64_t steps = (int64_t) ceil((stop - start) / step);
    GGML_ASSERT(ggml_nelements(dst) == steps);

    const int64_t ne0 = dst->ne[0];
    float * dst_d = (float *) dst->data;
    sycl::queue & q = ctx.stream();

    q.parallel_for(sycl::range<1>(ne0), [=](sycl::item<1> item) {
        const int64_t i = item.get_id(0);
        dst_d[i] = start + step * (float) i;
    });
}

bool ggml_sycl_supports_arange(const ggml_tensor * op) {
    return op->type == GGML_TYPE_F32;
}
