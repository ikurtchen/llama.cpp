// SYCL top-k op: per-row indices of the k largest values (unordered within the k slots is
// acceptable - matches the ggml-cuda fallback path's contract). Implemented as a full-row
// descending bitonic sort (reusing argsort's kernel) followed by taking the first k columns
// per row. CUDA's CUB-based top-k fast path (device-native, no full sort) is not ported here
// since correctness, not peak throughput, is the goal of this migration step.
#include "top-k.hpp"
#include "argsort.hpp"

#include <sycl/sycl.hpp>

void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
    const int64_t k     = dst->ne[0];
    const int ncols_pad = ggml_sycl_next_power_of_2((int) ncols);

    sycl::queue & q = ctx.stream();
    int * tmp_dst = sycl::malloc_device<int>(ncols * nrows, q);

    ggml_sycl_argsort_bitonic_raw(q, (const float *) src0->data, tmp_dst, (int) ncols, ncols_pad, (int) nrows,
                                  /*ascending=*/false);

    int * dst_d = (int *) dst->data;
    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<2>(nrows, k), [=](sycl::item<2> item) {
            const int64_t row = item.get_id(0);
            const int64_t col = item.get_id(1);
            dst_d[row * k + col] = tmp_dst[row * ncols + col];
        });
    }).wait();

    sycl::free(tmp_dst, q);
}

bool ggml_sycl_supports_top_k(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const int ncols_pad = ggml_sycl_next_power_of_2((int) src0->ne[0]);
    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 &&
           ggml_is_contiguous(src0) && ncols_pad <= 1024;
}
