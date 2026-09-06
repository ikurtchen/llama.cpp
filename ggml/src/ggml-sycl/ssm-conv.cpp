#include "ssm-conv.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace {

template <bool apply_silu>
static void ssm_conv_f32(
        sycl::queue & q,
        const float * src0,
        const float * src1,
        const float * bias,
        int src0_nb1,
        int src0_nb2,
        int src1_nb1,
        float * dst,
        int dst_nb0,
        int dst_nb1,
        int dst_nb2,
        int64_t nc,
        int64_t nr,
        int64_t n_t,
        int64_t n_s) {
    constexpr int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    const size_t blocks_y = (size_t) ((nr + threads - 1) / threads);
    q.parallel_for(
            sycl::nd_range<3>(sycl::range<3>((size_t) n_s, blocks_y, (size_t) threads), sycl::range<3>(1, 1, threads)),
            [=](sycl::nd_item<3> item) {
                const int seq = (int) item.get_group(0);
                const int row = (int) item.get_group(1) * threads + (int) item.get_local_id(2);
                if (row >= nr) {
                    return;
                }

                const float * x_block = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src0) + seq * src0_nb2);
                const float * w_row = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src1) + (int64_t) row * src1_nb1);
                float * y_block = reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + seq * dst_nb2 + (int64_t) row * dst_nb0);

                const int stride_x = src0_nb1 / (int) sizeof(float);
                const int stride_y = dst_nb1 / (int) sizeof(float);
                const float bias_val = bias ? bias[row] : 0.0f;

                for (int64_t token = 0; token < n_t; ++token) {
                    float sum = bias_val;
                    const float * x_row = x_block + token + (int64_t) row * stride_x;
                    for (int64_t j = 0; j < nc; ++j) {
                        sum += x_row[j] * w_row[j];
                    }
                    if constexpr (apply_silu) {
                        sum = sum / (1.0f + sycl::exp(-sum));
                    }
                    y_block[token * stride_y] = sum;
                }
            });
}

} // namespace

void ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * (int64_t) sizeof(float));

    const int64_t nc  = src1->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];

    sycl::queue & q = ctx.stream();
    ssm_conv_f32<false>(q,
            static_cast<const float *>(src0->data),
            static_cast<const float *>(src1->data),
            nullptr,
            (int) src0->nb[1],
            (int) src0->nb[2],
            (int) src1->nb[1],
            static_cast<float *>(dst->data),
            (int) dst->nb[0],
            (int) dst->nb[1],
            (int) dst->nb[2],
            nc, nr, n_t, n_s);
}

bool ggml_sycl_supports_ssm_conv(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1) {
        return false;
    }
    return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           src0->ne[1] % 128 == 0 && src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) &&
           src0->nb[1] == src0->ne[0] * (int64_t) sizeof(float);
}
