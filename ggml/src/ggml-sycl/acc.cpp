// SYCL acc op: dst = copy of src0, with src1 added in-place at a byte offset (used to write a
// smaller tensor into a slice of a larger one, e.g. updating part of a KV cache/view). Mirrors
// ggml-cuda/acc.cu's algorithm but written directly against SYCL.
#include "acc.hpp"

#include <sycl/sycl.hpp>

void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));

    const int64_t s1     = dst->op_params[0] / sizeof(float);
    const int64_t s2     = dst->op_params[1] / sizeof(float);
    const int64_t s3     = dst->op_params[2] / sizeof(float);
    const int64_t offset = dst->op_params[3] / sizeof(float);

    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const int64_t ne  = ggml_nelements(dst);

    const float * x = (const float *) src0->data;
    const float * y = (const float *) src1->data;
    float       * d = (float *) dst->data;

    sycl::queue & q = ctx.stream();
    q.parallel_for(sycl::range<1>(ne), [=](sycl::item<1> item) {
        const int64_t i = item.get_id(0);
        const int64_t src1_idx = i - offset;

        int64_t tmp  = src1_idx;
        const int64_t i13 = tmp / s3; tmp -= i13 * s3;
        const int64_t i12 = tmp / s2; tmp -= i12 * s2;
        const int64_t i11 = tmp / s1; tmp -= i11 * s1;
        const int64_t i10 = tmp;

        float val = x[i];
        if (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
            val += y[((i13 * ne12 + i12) * ne11 + i11) * ne10 + i10];
        }
        d[i] = val;
    });
}

bool ggml_sycl_supports_acc(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(src1);
}
