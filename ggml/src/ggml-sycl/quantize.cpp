#include "quantize.hpp"

#include <sycl/sycl.hpp>

namespace {

inline void store_q8_stats(block_q8_0 & y, float d, float sum) {
    GGML_UNUSED(sum);
    y.d = (ggml_half) d;
}

inline void store_q8_stats(block_q8_1 & y, float d, float sum) {
    y.data.d = (ggml_half) d;
    y.data.s = (ggml_half) sum;
}

template <typename block_t, bool with_sum>
void quantize_row_q8_x(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, sycl::queue & q) {
    GGML_UNUSED(ids);
    GGML_UNUSED(type_src0);
    GGML_ASSERT(ids == nullptr);
    GGML_ASSERT(ne0 % QK8_0 == 0);

    const int64_t blocks_per_row = ne0 / QK8_0;
    const int64_t rows = ne1 * ne2 * ne3;
    const int64_t total_blocks = rows * blocks_per_row;
    block_t * y = (block_t *) vy;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_blocks * QK8_0), sycl::range<1>(QK8_0)),
        [=](sycl::nd_item<1> item) {
        const int64_t ib = item.get_group(0);
        const int      lid = item.get_local_id(0);
        const int64_t  row = ib / blocks_per_row;
        const int64_t  block = ib % blocks_per_row;

        const int64_t i1 = row % ne1;
        const int64_t t = row / ne1;
        const int64_t i2 = t % ne2;
        const int64_t i3 = t / ne2;

        const int64_t i0 = block * QK8_0 + lid;
        const float xi = i0 < ne00 ? x[i3 * s03 + i2 * s02 + i1 * s01 + i0] : 0.0f;

        const float amax = sycl::reduce_over_group(item.get_group(), sycl::fabs(xi), sycl::maximum<float>());
        const float sum = with_sum ? sycl::reduce_over_group(item.get_group(), xi, sycl::plus<float>()) : 0.0f;
        const float d = amax / 127.0f;
        const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        const int8_t qv = amax > 0.0f ? (int8_t) sycl::round(xi * d_inv) : 0;

        y[ib].qs[lid] = qv;

        if (lid == 0) {
            store_q8_stats(y[ib], d, sum);
        }
    });
}

} // namespace

void ggml_sycl_quantize_row_q8_0(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, sycl::queue & q) {
    quantize_row_q8_x<block_q8_0, false>(x, ids, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, q);
}

void ggml_sycl_quantize_row_q8_1(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, sycl::queue & q) {
    quantize_row_q8_x<block_q8_1, true>(x, ids, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, q);
}
