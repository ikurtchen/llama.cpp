#include "sumrows.hpp"

// ---------------------------------------------------------------------------
// reduce_rows_f32 — SYCL port of ggml-cuda/reduce_rows.cuh
//
// Adaptive block sizing:
//   - ncols < 256 : block_size = WARP_SIZE, simple scalar loop + warp_reduce_sum
//                    (no SLM needed — avoids wasting work-items for small rows)
//   - ncols >= 256: block_size = 256, 8x unroll + block_reduce<SUM>
//
// When norm==true the row sum is divided by ncols (used by MEAN).
// ---------------------------------------------------------------------------

static constexpr int SUMROWS_BLOCK_SIZE = 256;

// Small-row path: one warp (WARP_SIZE work-items) per row, warp_reduce_sum.
static void reduce_rows_f32_small(const float * x, float * dst, int ncols,
                                  int nrows, bool norm, sycl::queue & stream) {
    const sycl::range<1> global_range(nrows * WARP_SIZE);
    const sycl::range<1> local_range(WARP_SIZE);

    stream.submit([&](sycl::handler & h) {
        if (norm) {
            h.parallel_for(
                sycl::nd_range<1>(global_range, local_range),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    const int row = item.get_group(0);
                    const int col = item.get_local_id(0);

                    float sum = 0.0f;
                    for (int i = col; i < ncols; i += WARP_SIZE) {
                        sum += x[row * ncols + i];
                    }
                    sum = warp_reduce_sum(sum);

                    if (col == 0) {
                        dst[row] = sum / ncols;
                    }
                });
        } else {
            h.parallel_for(
                sycl::nd_range<1>(global_range, local_range),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    const int row = item.get_group(0);
                    const int col = item.get_local_id(0);

                    float sum = 0.0f;
                    for (int i = col; i < ncols; i += WARP_SIZE) {
                        sum += x[row * ncols + i];
                    }
                    sum = warp_reduce_sum(sum);

                    if (col == 0) {
                        dst[row] = sum;
                    }
                });
        }
    });
}

// Large-row path: 256 work-items per row, 8x unroll + block_reduce<SUM>.
static void reduce_rows_f32_large(const float * x, float * dst, int ncols,
                                  int nrows, bool norm, sycl::queue & stream) {
    const sycl::range<1> global_range(nrows * SUMROWS_BLOCK_SIZE);
    const sycl::range<1> local_range(SUMROWS_BLOCK_SIZE);
    constexpr int num_warps = SUMROWS_BLOCK_SIZE / WARP_SIZE;

    stream.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> shared_acc(num_warps, h);

        if (norm) {
            h.parallel_for(
                sycl::nd_range<1>(global_range, local_range),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    const int row = item.get_group(0);
                    const int col = item.get_local_id(0);

                    constexpr int num_unroll = 8;
                    float sum_temp[num_unroll] = { 0.0f };

                    for (int i = col; i < ncols;) {
                        float temp[num_unroll];
                        for (int j = 0; j < num_unroll; ++j) {
                            temp[j] = (i < ncols) ? x[row * ncols + i] : 0.0f;
                            i += SUMROWS_BLOCK_SIZE;
                        }
                        for (int j = 0; j < num_unroll; ++j) {
                            sum_temp[j] += temp[j];
                        }
                    }

                    float sum = 0.0f;
                    for (int j = 0; j < num_unroll; ++j) {
                        sum += sum_temp[j];
                    }

                    float * shared_vals = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
                    sum = block_reduce<block_reduce_method::SUM, SUMROWS_BLOCK_SIZE>(sum, shared_vals, item);

                    if (col == 0) {
                        dst[row] = sum / ncols;
                    }
                });
        } else {
            h.parallel_for(
                sycl::nd_range<1>(global_range, local_range),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    const int row = item.get_group(0);
                    const int col = item.get_local_id(0);

                    constexpr int num_unroll = 8;
                    float sum_temp[num_unroll] = { 0.0f };

                    for (int i = col; i < ncols;) {
                        float temp[num_unroll];
                        for (int j = 0; j < num_unroll; ++j) {
                            temp[j] = (i < ncols) ? x[row * ncols + i] : 0.0f;
                            i += SUMROWS_BLOCK_SIZE;
                        }
                        for (int j = 0; j < num_unroll; ++j) {
                            sum_temp[j] += temp[j];
                        }
                    }

                    float sum = 0.0f;
                    for (int j = 0; j < num_unroll; ++j) {
                        sum += sum_temp[j];
                    }

                    float * shared_vals = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get_raw();
                    sum = block_reduce<block_reduce_method::SUM, SUMROWS_BLOCK_SIZE>(sum, shared_vals, item);

                    if (col == 0) {
                        dst[row] = sum;
                    }
                });
        }
    });
}

void reduce_rows_f32_sycl(const float * x, float * dst, int ncols, int nrows,
                           bool norm, sycl::queue & stream) {
    if (ncols < SUMROWS_BLOCK_SIZE) {
        reduce_rows_f32_small(x, dst, ncols, nrows, norm, stream);
    } else {
        reduce_rows_f32_large(x, dst, ncols, nrows, norm, stream);
    }
}

// ---------------------------------------------------------------------------
// SUM_ROWS: reduce each row of src0 to a single scalar in dst.
// ---------------------------------------------------------------------------
void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const float * src0_d = (const float *) src0->data;
    float *       dst_d  = (float *) dst->data;

    const int ncols = (int) src0->ne[0];
    const int nrows = (int) ggml_nrows(src0);

    sycl::queue & stream = *(ctx.stream());

    reduce_rows_f32_sycl(src0_d, dst_d, ncols, nrows, /*norm=*/false, stream);
}

// ---------------------------------------------------------------------------
// MEAN: reduce each row of src0 and divide by ncols.
// ---------------------------------------------------------------------------
void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const float * src0_d = (const float *) src0->data;
    float *       dst_d  = (float *) dst->data;

    const int ncols = (int) src0->ne[0];
    const int nrows = (int) ggml_nrows(src0);

    sycl::queue & stream = *(ctx.stream());

    reduce_rows_f32_sycl(src0_d, dst_d, ncols, nrows, /*norm=*/true, stream);
}
