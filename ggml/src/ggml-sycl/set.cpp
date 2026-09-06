#include "set.hpp"
#include "cpy.hpp"

void ggml_sycl_op_set(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32));
    GGML_ASSERT(src1->type == src0->type);
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * params = (const int32_t *) dst->op_params;
    const size_t nb1 = params[0];
    const size_t nb2 = params[1];
    const size_t nb3 = params[2];
    const size_t offset = params[3];
    const bool inplace = params[4] != 0;

    if (!inplace) {
        ggml_sycl_cpy(ctx, src0, dst);
    }

    ggml_tensor dst_view = *dst;
    dst_view.data  = (void *) ((char *) dst->data + offset);
    dst_view.ne[0] = src1->ne[0];
    dst_view.ne[1] = src1->ne[1];
    dst_view.ne[2] = src1->ne[2];
    dst_view.ne[3] = src1->ne[3];
    dst_view.nb[0] = ggml_element_size(dst);
    dst_view.nb[1] = nb1;
    dst_view.nb[2] = nb2;
    dst_view.nb[3] = nb3;

    ggml_sycl_cpy(ctx, src1, &dst_view);
}

bool ggml_sycl_supports_set(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_type t = op->type;

    return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
           src0->type == t && src1->type == t &&
           ggml_is_contiguous(op) && ggml_is_contiguous(src0) && ggml_is_contiguous(src1);
}
