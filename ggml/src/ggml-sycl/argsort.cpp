// SYCL argsort op: per-row bitonic sort producing the permutation of indices that sorts each
// row ascending or descending. Mirrors ggml-cuda/argsort.cu's bitonic-sort kernel (the portable
// path; CUDA's CUB-based radix/segmented sort fast path for large ncols is not ported here since
// correctness, not peak throughput, is the goal of this migration step).
#include "argsort.hpp"

#include <sycl/sycl.hpp>

int ggml_sycl_next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void ggml_sycl_argsort_bitonic_raw(sycl::queue & q, const float * x, int * dst, int ncols, int ncols_pad,
                                    int nrows, bool ascending) {
    sycl::range<1> local(ncols_pad);
    sycl::range<1> global((size_t) nrows * ncols_pad);

    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int, 1> row_idx(sycl::range<1>(ncols_pad), cgh);
        cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
            const int col = item.get_local_id(0);
            const int row = item.get_group(0);
            const float * x_row = x + (size_t) row * ncols;

            row_idx[col] = col;
            item.barrier(sycl::access::fence_space::local_space);

            for (int k = 2; k <= ncols_pad; k *= 2) {
                for (int j = k / 2; j > 0; j /= 2) {
                    const int ixj = col ^ j;
                    if (ixj > col) {
                        if ((col & k) == 0) {
                            const bool swap_needed = row_idx[col] >= ncols ||
                                (row_idx[ixj] < ncols && (ascending ?
                                    x_row[row_idx[col]] > x_row[row_idx[ixj]] :
                                    x_row[row_idx[col]] < x_row[row_idx[ixj]]));
                            if (swap_needed) {
                                const int tmp = row_idx[col];
                                row_idx[col]  = row_idx[ixj];
                                row_idx[ixj]  = tmp;
                            }
                        } else {
                            const bool swap_needed = row_idx[ixj] >= ncols ||
                                (row_idx[col] < ncols && (ascending ?
                                    x_row[row_idx[col]] < x_row[row_idx[ixj]] :
                                    x_row[row_idx[col]] > x_row[row_idx[ixj]]));
                            if (swap_needed) {
                                const int tmp = row_idx[col];
                                row_idx[col]  = row_idx[ixj];
                                row_idx[ixj]  = tmp;
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
            }

            if (col < ncols) {
                dst[(size_t) row * ncols + col] = row_idx[col];
            }
        });
    });
}

void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
    const int ncols_pad = ggml_sycl_next_power_of_2((int) ncols);

    const enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];
    GGML_ASSERT(order == GGML_SORT_ORDER_ASC || order == GGML_SORT_ORDER_DESC);

    sycl::queue & q = ctx.stream();
    ggml_sycl_argsort_bitonic_raw(q, (const float *) src0->data, (int *) dst->data,
                            (int) ncols, ncols_pad, (int) nrows, order == GGML_SORT_ORDER_ASC);
}

bool ggml_sycl_supports_argsort(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    // bitonic sort needs one work-group per row sized to the next power-of-2 column count,
    // limited by device local-memory / work-group-size capacity (matches CUDA's ncols<=1024 cap)
    const int ncols_pad = ggml_sycl_next_power_of_2((int) src0->ne[0]);
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 &&
           ggml_is_contiguous(src0) && ncols_pad <= 1024;
}
