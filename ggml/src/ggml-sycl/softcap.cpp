#include "softcap.hpp"
#include "common.hpp"

#define SYCL_SOFTCAP_BLOCK_SIZE 256

static void softcap_kernel(const float * x, float * dst, const float scale, const float softcap, const int k,
                           const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t)item.get_global_id(0);
    if (i >= k) {
        return;
    }
    dst[i] = sycl::tanh(scale * x[i]) * softcap;
}

void ggml_sycl_op_softcap(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);

    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    const int64_t k = ggml_nelements(src0);
    const int64_t num_blocks = (k + SYCL_SOFTCAP_BLOCK_SIZE - 1) / SYCL_SOFTCAP_BLOCK_SIZE;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    stream->parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_SOFTCAP_BLOCK_SIZE, SYCL_SOFTCAP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            softcap_kernel(src0_d, dst_d, scale, softcap, k, item);
        });
}