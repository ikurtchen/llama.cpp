#include "fwht.hpp"

#include <sycl/sycl.hpp>

#include <cmath>

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));

    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n = src->ne[0];
    if (n < 64 || n > 512 || (n & (n - 1)) != 0) {
        return false;
    }

    sycl::queue & q = ctx.stream();
    const size_t max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    if ((size_t) n > max_wg) {
        return false;
    }

    const int64_t nrows = ggml_nrows(src);
    const float * src_d = (const float *) src->data;
    float * dst_d = (float *) dst->data;
    const float scale = 1.0f / std::sqrt((float) n);

    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> row(sycl::range<1>(n), cgh);

        cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(nrows, n), sycl::range<2>(1, n)),
                [=](sycl::nd_item<2> item) {
                    const int64_t r = item.get_group(0);
                    const int64_t lane = item.get_local_id(1);

                    row[lane] = src_d[r * n + lane] * scale;
                    sycl::group_barrier(item.get_group());

                    for (int64_t len = 1; len < n; len <<= 1) {
                        if ((lane & len) == 0) {
                            const float x = row[lane];
                            const float y = row[lane + len];
                            row[lane] = x + y;
                            row[lane + len] = x - y;
                        }
                        sycl::group_barrier(item.get_group());
                    }

                    dst_d[r * n + lane] = row[lane];
                });
    });

    return true;
}
