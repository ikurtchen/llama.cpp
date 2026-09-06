#include "softcap.hpp"

#include <sycl/sycl.hpp>

bool ggml_sycl_can_fuse_softcap(const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_can_fuse_subgraph(cgraph, node_idx,
            { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { node_idx + 2 })) {
        return false;
    }

    const ggml_tensor * scale0 = cgraph->nodes[node_idx + 0];
    const ggml_tensor * unary  = cgraph->nodes[node_idx + 1];
    const ggml_tensor * scale1 = cgraph->nodes[node_idx + 2];
    const ggml_tensor * src0   = scale0->src[0];

    return unary->src[0] == scale0 && scale1->src[0] == unary &&
           ggml_get_unary_op(unary) == GGML_UNARY_OP_TANH &&
           src0->type == GGML_TYPE_F32 && scale0->type == GGML_TYPE_F32 &&
           unary->type == GGML_TYPE_F32 && scale1->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(scale1);
}

void ggml_sycl_op_softcap(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale, (float *) src->op_params + 0, sizeof(scale));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(softcap));

    const float * in = (const float *) src0->data;
    float * out = (float *) dst->data;
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
        out[i] = sycl::tanh(scale * in[i]) * softcap;
    });
}
