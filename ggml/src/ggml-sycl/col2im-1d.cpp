#include "col2im-1d.hpp"

#include <sycl/sycl.hpp>
#include <type_traits>

namespace {
template <typename T>
inline float to_float(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, sycl::half>) {
        return (float) value;
    } else {
        return GGML_BF16_TO_FP32(value);
    }
}

template <typename T>
inline T from_float(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, sycl::half>) {
        return (sycl::half) value;
    } else {
        return GGML_FP32_TO_BF16(value);
    }
}

template <typename T>
void col2im_1d_dispatch(sycl::queue & q, const T * col, T * out, int t_in, int t_out, int oc, int k, int k_oc, int s0, int p0) {
    const int total = t_out * oc;
    const int block = 256;
    const int nblk = (total + block - 1) / block;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int idx = (int) item.get_global_id(0);
                if (idx >= total) {
                    return;
                }

                const int cur_oc = idx / t_out;
                const int cur_t_out = idx - cur_oc * t_out;
                const int t_abs = cur_t_out + p0;

                int t_in_min = (t_abs - k + s0) / s0;
                if (t_in_min < 0) {
                    t_in_min = 0;
                }
                int t_in_max = t_abs / s0;
                if (t_in_max >= t_in) {
                    t_in_max = t_in - 1;
                }

                float sum = 0.0f;
                for (int cur_t_in = t_in_min; cur_t_in <= t_in_max; ++cur_t_in) {
                    const int cur_k = t_abs - cur_t_in * s0;
                    sum += to_float(col[(cur_oc * k + cur_k) + cur_t_in * k_oc]);
                }

                out[idx] = from_float<T>(sum);
            });
}
}

void ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int s0 = ggml_get_op_params_i32(dst, 0);
    const int oc = ggml_get_op_params_i32(dst, 1);
    const int p0 = ggml_get_op_params_i32(dst, 2);

    const int k_oc = (int) src0->ne[0];
    const int t_in = (int) src0->ne[1];
    const int k = k_oc / oc;
    const int t_out = (int) dst->ne[0];

    sycl::queue & q = ctx.stream();
    switch (src0->type) {
        case GGML_TYPE_F32:
            col2im_1d_dispatch(q, (const float *) src0->data, (float *) dst->data, t_in, t_out, oc, k, k_oc, s0, p0);
            break;
        case GGML_TYPE_F16:
            col2im_1d_dispatch(q, (const sycl::half *) src0->data, (sycl::half *) dst->data, t_in, t_out, oc, k, k_oc, s0, p0);
            break;
        case GGML_TYPE_BF16:
            col2im_1d_dispatch(q, (const ggml_bf16_t *) src0->data, (ggml_bf16_t *) dst->data, t_in, t_out, oc, k, k_oc, s0, p0);
            break;
        default:
            GGML_ABORT("col2im_1d: unsupported type");
    }
}

bool ggml_sycl_supports_col2im_1d(const ggml_tensor * op) {
    const ggml_type src0_type = op->src[0]->type;
    return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
           op->type == src0_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op);
}
