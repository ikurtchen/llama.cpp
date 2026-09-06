// SYCL scale op: dst = scale * src0 + bias, element-wise. Mirrors ggml-cuda/scale.cu's algorithm
// (a flat grid-stride loop over all elements) but written directly against SYCL, not translated.
#include "scale.hpp"

#include <sycl/sycl.hpp>

void ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    const float * src0_d = (const float *) src0->data;
    float *       dst_d  = (float *) dst->data;
    const int64_t ne      = ggml_nelements(src0);

    sycl::queue & q = ctx.stream();
    const int64_t block = 256;
    const int64_t nblk   = (ne + block - 1) / block;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= ne) {
            return;
        }
        dst_d[i] = scale * src0_d[i] + bias;
    });
}

bool ggml_sycl_supports_scale(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
}
