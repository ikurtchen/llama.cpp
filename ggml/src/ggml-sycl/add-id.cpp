#include "add-id.hpp"

#include <sycl/sycl.hpp>

void ggml_sycl_op_add_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);
    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb20 == sizeof(int32_t));

    const float * a = (const float *) src0->data;
    const float * b = (const float *) src1->data;
    const int32_t * ids = (const int32_t *) src2->data;
    float * out = (float *) dst->data;

    sycl::queue & q = ctx.stream();
    const int64_t block = std::min<int64_t>(ne0, 256);
    const int64_t groups = ne01 * ne02;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * block), sycl::range<1>(block)),
        [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group(0);
        const int64_t i1 = group % ne01;
        const int64_t i2 = group / ne01;
        const int64_t expert = *(const int32_t *) ((const char *) ids + i1 * sizeof(int32_t) + i2 * nb21);

        float * dst_row = out + (i2 * ne01 + i1) * ne0;
        const float * src0_row = (const float *) ((const char *) a + i1 * nb01 + i2 * nb02);
        const float * src1_row = (const float *) ((const char *) b + expert * nb11);

        for (int64_t i0 = item.get_local_id(0); i0 < ne0; i0 += block) {
            dst_row[i0] = src0_row[i0] + src1_row[i0];
        }
    });
}

bool ggml_sycl_supports_add_id(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
           src2->type == GGML_TYPE_I32 && op->ne[3] == 1 && src0->ne[3] == 1 && ggml_is_contiguous(src0);
}
