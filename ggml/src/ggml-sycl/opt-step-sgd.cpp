#include "opt-step-sgd.hpp"

#include <cstdint>

#define SYCL_OPT_STEP_SGD_BLOCK_SIZE 256

static void opt_step_sgd_f32_sycl(
    float * x, const float * g, const float * pars,
    const int64_t k, queue_ptr stream) {

    const int64_t num_blocks = (k + SYCL_OPT_STEP_SGD_BLOCK_SIZE - 1) / SYCL_OPT_STEP_SGD_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_OPT_STEP_SGD_BLOCK_SIZE, SYCL_OPT_STEP_SGD_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            const int64_t i = item.get_global_id(0);

            if (i >= k) {
                return;
            }

            // pars[0] = learning rate, pars[1] = weight decay
            x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
        });
}

void ggml_sycl_op_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0      = dst->src[0];
    const ggml_tensor * src0_grad = dst->src[1];
    const ggml_tensor * params    = dst->src[2];

    GGML_ASSERT(src0->type      == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
    GGML_ASSERT(params->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(params) == 2);

    float       * src0_d      = (float       *) src0->data;
    const float * src0_grad_d = (const float *) src0_grad->data;
    const float * params_d    = (const float *) params->data;

    queue_ptr stream = ctx.stream();

    const int64_t ne = ggml_nelements(src0);

    opt_step_sgd_f32_sycl(src0_d, src0_grad_d, params_d, ne, stream);
}
