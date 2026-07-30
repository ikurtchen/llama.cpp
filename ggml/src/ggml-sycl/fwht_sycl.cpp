#include "common.hpp"
#include <sycl/sycl.hpp>

template <int N>
static void fwht_kernel(const float scale, const float * src, float * dst, int64_t n_rows,
                        sycl::nd_item<1> item_ct1) {
    static constexpr int warp_size = WARP_SIZE;
    static constexpr int el_w = N / warp_size;

    const int64_t row = item_ct1.get_group(0);
    const int lane = item_ct1.get_local_id(0);

    if (row >= n_rows) {
        return;
    }

    src += row * N;
    dst += row * N;

    float reg[el_w];

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), val, h);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }
    const int     N    = (int) src->ne[0];
    const int64_t rows = ggml_nrows(src);

    if ((N & (N - 1)) != 0) {
        return false;
    }

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *) dst->data;

    const float scale = 1.0f / sycl::sqrt((float) N);

    dpct::queue_ptr stream = ctx.stream();

    // 1D nd_range: one work-group per row, WARP_SIZE work-items per work-group
    // Each work-group = one sub-group, processing one row's FWHT entirely in registers
    switch (N) {
        case 64:
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(rows * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    fwht_kernel<64>(scale, src_d, dst_d, rows, item_ct1);
                });
            return true;
        case 128:
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(rows * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    fwht_kernel<128>(scale, src_d, dst_d, rows, item_ct1);
                });
            return true;
        case 256:
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(rows * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    fwht_kernel<256>(scale, src_d, dst_d, rows, item_ct1);
                });
            return true;
        case 512:
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(rows * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    fwht_kernel<512>(scale, src_d, dst_d, rows, item_ct1);
                });
            return true;
        default:
            return false;
    }
}
