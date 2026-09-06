#include "opt-step-adamw.hpp"

#include <sycl/sycl.hpp>

#include <cmath>

void ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0         = dst->src[0];
    const ggml_tensor * src0_grad    = dst->src[1];
    const ggml_tensor * src0_grad_m  = dst->src[2];
    const ggml_tensor * src0_grad_v  = dst->src[3];
    const ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_m->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_v->type == GGML_TYPE_F32);
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

    float * x = (float *) src0->data;
    const float * g = (const float *) src0_grad->data;
    float * g_m = (float *) src0_grad_m->data;
    float * g_v = (float *) src0_grad_v->data;
    const float * pars = (const float *) adamw_params->data;
    const int64_t ne = ggml_nelements(src0);

    sycl::queue & q = ctx.stream();
    const int64_t block = 256;
    const int64_t nblk = (ne + block - 1) / block;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int64_t i = item.get_global_id(0);
                if (i >= ne) {
                    return;
                }

                const float alpha  = pars[0];
                const float beta1  = pars[1];
                const float beta2  = pars[2];
                const float eps    = pars[3];
                const float wd     = pars[4];
                const float beta1h = pars[5];
                const float beta2h = pars[6];

                const float gi  = g[i];
                const float gmi = g_m[i] * beta1 + gi * (1.0f - beta1);
                const float gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

                g_m[i] = gmi;
                g_v[i] = gvi;

                const float mh = gmi * beta1h;
                const float vh = sycl::sqrt(gvi * beta2h) + eps;

                x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
            });
}

bool ggml_sycl_supports_opt_step_adamw(const ggml_tensor * op) {
    const ggml_tensor * src0         = op->src[0];
    const ggml_tensor * src0_grad    = op->src[1];
    const ggml_tensor * src0_grad_m  = op->src[2];
    const ggml_tensor * src0_grad_v  = op->src[3];
    const ggml_tensor * adamw_params = op->src[4];

    return src0->type == GGML_TYPE_F32 &&
           src0_grad->type == GGML_TYPE_F32 &&
           src0_grad_m->type == GGML_TYPE_F32 &&
           src0_grad_v->type == GGML_TYPE_F32 &&
           adamw_params->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src0_grad) &&
           ggml_is_contiguous(src0_grad_m) &&
           ggml_is_contiguous(src0_grad_v) &&
           ggml_is_contiguous(adamw_params) &&
           ggml_are_same_shape(src0, src0_grad) &&
           ggml_are_same_shape(src0, src0_grad_m) &&
           ggml_are_same_shape(src0, src0_grad_v) &&
           ggml_nelements(adamw_params) == 7;
}
