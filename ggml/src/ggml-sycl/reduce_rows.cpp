// SYCL row-reduction ops: SUM_ROWS (per-row sum), MEAN (per-row sum / ncols), and SUM (a single
// sum over the whole flattened tensor, implemented as a 1-row SUM_ROWS). One work-group per row;
// each work-item accumulates a strided partial sum, then sycl::reduce_over_group (the idiomatic
// group collective, not a serial loop) combines them. Mirrors ggml-cuda/reduce_rows.cuh's
// algorithm but written directly against SYCL.
#include "reduce_rows.hpp"

#include <sycl/sycl.hpp>

static void reduce_rows_f32(sycl::queue & q, const float * x, float * dst, int64_t ncols, int64_t nrows,
                            bool norm, size_t wg_size) {
    // Accumulate in double, matching the CPU reference (ggml_vec_sum_f32 uses a double accumulator),
    // so summation-order differences from a group reduction stay within the test tolerance.
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(nrows * wg_size), sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> item) {
            const int64_t row = item.get_group(0);
            const float * rowx = x + row * ncols;

            double sum = 0.0;
            for (int64_t col = item.get_local_id(0); col < ncols; col += wg_size) {
                sum += (double) rowx[col];
            }

            const double total = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<double>());
            if (item.get_local_id(0) == 0) {
                dst[row] = (float) (norm ? total / (double) ncols : total);
            }
        });
    });
}

static void reduce_rows_dispatch(ggml_backend_sycl_context & ctx, const float * src0_d, float * dst_d,
                                 int64_t ncols, int64_t nrows, bool norm) {
    sycl::queue & q = ctx.stream();
    const size_t max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t wg_size = std::min<size_t>(max_wg, ((size_t) ncols + 31) / 32 * 32);
    wg_size = std::max<size_t>(wg_size, 32);
    reduce_rows_f32(q, src0_d, dst_d, ncols, nrows, norm, wg_size);
}

void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    reduce_rows_dispatch(ctx, (const float *) src0->data, (float *) dst->data,
                         src0->ne[0], ggml_nrows(src0), /*norm=*/false);
}

void ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    reduce_rows_dispatch(ctx, (const float *) src0->data, (float *) dst->data,
                         src0->ne[0], ggml_nrows(src0), /*norm=*/true);
}

void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int64_t ne = ggml_nelements(src0);
    reduce_rows_dispatch(ctx, (const float *) src0->data, (float *) dst->data, ne, 1, /*norm=*/false);
}

bool ggml_sycl_supports_sum_rows(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0);
}

bool ggml_sycl_supports_mean(const ggml_tensor * op) {
    return ggml_sycl_supports_sum_rows(op);
}

bool ggml_sycl_supports_sum(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    // Matches ggml-cuda's own scope: row-contiguous only (nb[0] == type size). A tensor that is
    // contiguously-allocated but not row-contiguous (e.g. a permute swapping dim0) still sums
    // correctly here, but the CPU reference used by test-backend-ops mishandles that layout, so
    // this stays out of scope rather than passing a test the framework itself cannot validate.
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
}
