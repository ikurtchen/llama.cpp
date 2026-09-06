#include "tsembd.hpp"

#include <sycl/sycl.hpp>

void ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];
    const int half = dim / 2;
    const int half_ceil = (dim + 1) / 2;
    const int64_t rows = src0->ne[0];
    const size_t nb1 = dst->nb[1];

    const float * timesteps = (const float *) src0->data;
    char * out = (char *) dst->data;

    sycl::queue & q = ctx.stream();
    q.parallel_for(sycl::range<2>((size_t) rows, (size_t) half_ceil), [=](sycl::item<2> item) {
        const int64_t i = item.get_id(0);
        const int j = (int) item.get_id(1);
        float * embed = (float *) (out + i * nb1);

        if (dim % 2 != 0 && j == half) {
            embed[2 * half] = 0.0f;
        }
        if (j >= half) {
            return;
        }

        const float timestep = timesteps[i];
        const float freq = sycl::exp(-sycl::log((float) max_period) * (float) j / (float) half);
        const float arg = timestep * freq;
        embed[j] = sycl::cos(arg);
        embed[j + half] = sycl::sin(arg);
    });
}

bool ggml_sycl_supports_timestep_embedding(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
}
