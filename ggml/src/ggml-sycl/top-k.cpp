#include "top-k.hpp"

#include "common.hpp"

// Declared in ggml-sycl.cpp — bitonic argsort for F32->I32
void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                           const int nrows, ggml_sort_order order,
                           queue_ptr stream, int device);

void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const float * src0_dd = static_cast<const float *>(src0->data);
    int32_t *     dst_dd  = static_cast<int32_t *>(dst->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
    const int64_t k     = dst->ne[0];

    // Allocate temp buffer for full argsort result (ncols per row)
    ggml_sycl_pool_alloc<int> tmp_alloc(ctx.pool(), ncols * nrows);
    int * tmp_dst = tmp_alloc.get();

    // Argsort in descending order — first k indices will be the top-k
    argsort_f32_i32_sycl(src0_dd, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC,
                         main_stream, ctx.device);

    // Copy first k indices per row from tmp_dst to dst_dd
    const int k_int     = static_cast<int>(k);
    const int ncols_int = static_cast<int>(ncols);

    main_stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nrows * 256), sycl::range<1>(256)),
            [=](sycl::nd_item<1> item) {
                const int row = item.get_group(0);
                for (int j = item.get_local_id(0); j < k_int; j += item.get_local_range(0)) {
                    dst_dd[row * k_int + j] = tmp_dst[row * ncols_int + j];
                }
            });
    });
}
