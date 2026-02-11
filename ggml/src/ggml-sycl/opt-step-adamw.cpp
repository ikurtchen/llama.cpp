#include "opt-step-adamw.hpp"

#include "common.hpp"

#define SYCL_OPT_STEP_ADAMW_BLOCK_SIZE 256

static void opt_step_adamw_f32(
    float * __restrict__ x, const float * __restrict__ g, float * __restrict__ g_m, float * __restrict__ g_v,
    const float * __restrict__ pars, const int64_t k,
    const sycl::nd_item<1> & item_ct1) {

    const int64_t i = (int64_t)item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);

    if (i >= k) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[i];
    const float gmi = g_m[i]*beta1 +    gi*(1.0f - beta1);
    const float gvi = g_v[i]*beta2 + gi*gi*(1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    const float mh =            gmi*beta1h;
    const float vh = sycl::sqrt(gvi*beta2h) + eps;

    x[i] = x[i]*(1.0f - alpha*wd) - alpha*mh/vh;
}

void ggml_sycl_op_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0         = dst->src[0];
    const ggml_tensor * src0_grad    = dst->src[1];
    const ggml_tensor * src0_grad_m  = dst->src[2];
    const ggml_tensor * src0_grad_v  = dst->src[3];
    const ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(src0->type         == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type    == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_m->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_v->type  == GGML_TYPE_F32);
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_m));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_v));
    GGML_ASSERT(ggml_is_contiguous(adamw_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    float       * src0_d         = (float       *) src0->data;
    const float * src0_grad_d    = (const float *) src0_grad->data;
    float       * src0_grad_m_d  = (float       *) src0_grad_m->data;
    float       * src0_grad_v_d  = (float       *) src0_grad_v->data;
    const float * adamw_params_d = (const float *) adamw_params->data;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int64_t ne = ggml_nelements(src0);
    const int64_t num_blocks = ceil_div((size_t)ne, (size_t)SYCL_OPT_STEP_ADAMW_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_OPT_STEP_ADAMW_BLOCK_SIZE),
                          sycl::range<1>(SYCL_OPT_STEP_ADAMW_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            opt_step_adamw_f32(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, item_ct1);
        });
}
