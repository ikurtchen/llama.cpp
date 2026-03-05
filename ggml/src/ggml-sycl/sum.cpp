#include "sum.hpp"
#include "sumrows.hpp"

// ---------------------------------------------------------------------------
// SUM: reduce ALL elements of src0 to a single scalar.
// Treats the entire tensor as a single row of nelements columns.
// ---------------------------------------------------------------------------
void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const float * src0_d = (const float *) src0->data;
    float *       dst_d  = (float *) dst->data;

    const int ncols = (int) ggml_nelements(src0);

    sycl::queue & stream = *(ctx.stream());

    reduce_rows_f32_sycl(src0_d, dst_d, ncols, /*nrows=*/1, /*norm=*/false, stream);
}
