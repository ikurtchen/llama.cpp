#include "opt-step-sgd.hpp"

#include <sycl/sycl.hpp>

void ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * grad = dst->src[1];
    const ggml_tensor * params = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(grad->type == GGML_TYPE_F32);
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(params));
    GGML_ASSERT(ggml_are_same_shape(src0, grad));
    GGML_ASSERT(ggml_nelements(params) == 2);

    float * x = (float *) src0->data;
    const float * g = (const float *) grad->data;
    const float * pars = (const float *) params->data;
    const int64_t ne = ggml_nelements(src0);

    sycl::queue & q = ctx.stream();
    const int64_t block = 256;
    const int64_t nblk  = (ne + block - 1) / block;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t i = item.get_global_id(0);
        if (i >= ne) {
            return;
        }
        x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
    });
}

bool ggml_sycl_supports_opt_step_sgd(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * grad = op->src[1];
    const ggml_tensor * params = op->src[2];
    return src0->type == GGML_TYPE_F32 && grad->type == GGML_TYPE_F32 && params->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(grad) && ggml_is_contiguous(params) &&
           ggml_are_same_shape(src0, grad) && ggml_nelements(params) == 2;
}
