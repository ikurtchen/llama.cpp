// SYCL diag_mask_inf op: dst[row,col] = -FLT_MAX where col > n_past + row % rows_per_channel,
// else copy src0. Mirrors ggml-cuda/diagmask.cu's algorithm but written directly against SYCL.
#include "diagmask.hpp"

#include <sycl/sycl.hpp>
#include <cfloat>

static void diag_mask_inf_f32(sycl::queue & q, const float * x, float * dst, int ncols, int rows_per_channel,
                               int n_past, int nrows) {
    const int block = 256;
    const int nblk   = (ncols + block - 1) / block;
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>(nrows, nblk * block), sycl::range<2>(1, block)),
        [=](sycl::nd_item<2> item) {
        const int row = item.get_global_id(0);
        const int col = item.get_global_id(1);
        if (col >= ncols) {
            return;
        }
        const int i = row * ncols + col;
        dst[i] = (col > n_past + row % rows_per_channel) ? -FLT_MAX : x[i];
    });
}

void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00  = src0->ne[0];
    const int64_t ne01  = src0->ne[1];
    const int nrows0    = ggml_nrows(src0);
    const int n_past    = ((const int32_t *) dst->op_params)[0];

    sycl::queue & q = ctx.stream();
    diag_mask_inf_f32(q, (const float *) src0->data, (float *) dst->data, (int) ne00, (int) ne01, n_past, nrows0);
}

bool ggml_sycl_supports_diag_mask_inf(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
}
