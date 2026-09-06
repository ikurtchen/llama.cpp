#include "cumsum.hpp"

#include <sycl/sycl.hpp>

namespace {

static size_t ggml_sycl_cumsum_wg_size(sycl::queue & q, int64_t ne00) {
    const size_t max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t wg = 32;
    while (wg < (size_t) ne00 && wg < max_wg && wg < 256) {
        wg *= 2;
    }
    return std::min(wg, max_wg);
}

}

void ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * src = (const float *) src0->data;
    float * out = (float *) dst->data;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t nrows = ne01 * ne02 * ne03;

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t nb0 = dst->nb[0];
    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    sycl::queue & q = ctx.stream();
    const size_t wg = ggml_sycl_cumsum_wg_size(q, ne00);

    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> carry(sycl::range<1>(1), cgh);

        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * (int64_t) wg), sycl::range<1>(wg)),
                [=](sycl::nd_item<1> item) {
                    const int64_t row = item.get_group(0);
                    const int64_t lid = item.get_local_id(0);

                    int64_t tmp = row;
                    const int64_t i1 = tmp % ne01;
                    tmp /= ne01;
                    const int64_t i2 = tmp % ne02;
                    const int64_t i3 = tmp / ne02;

                    const char * src_row = (const char *) src + i1 * nb01 + i2 * nb02 + i3 * nb03;
                    char * dst_row = (char *) out + i1 * nb1 + i2 * nb2 + i3 * nb3;

                    if (lid == 0) {
                        carry[0] = 0.0f;
                    }
                    sycl::group_barrier(item.get_group());

                    for (int64_t base = 0; base < ne00; base += (int64_t) wg) {
                        const int64_t idx = base + lid;
                        const float val = idx < ne00 ? *(const float *) (src_row + idx * nb00) : 0.0f;

                        const float scan = sycl::inclusive_scan_over_group(item.get_group(), val, sycl::plus<float>());
                        const float total = sycl::reduce_over_group(item.get_group(), val, sycl::plus<float>());

                        if (idx < ne00) {
                            *(float *) (dst_row + idx * nb0) = carry[0] + scan;
                        }

                        sycl::group_barrier(item.get_group());
                        if (lid == (int64_t) item.get_local_range(0) - 1) {
                            carry[0] += total;
                        }
                        sycl::group_barrier(item.get_group());
                    }
                });
    });
}

bool ggml_sycl_supports_cumsum(const ggml_tensor * op) {
    return op->type == GGML_TYPE_F32 &&
           op->src[0]->type == GGML_TYPE_F32;
}
