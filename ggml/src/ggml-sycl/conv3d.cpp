#include "conv3d.hpp"

#include <sycl/sycl.hpp>
#include <type_traits>

namespace {
template <typename T>
inline float to_float(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else {
        return (float) value;
    }
}

template <typename T>
void conv3d_dispatch(
        sycl::queue & q,
        const T * kernel,
        const float * src,
        float * dst,
        int64_t n,
        int64_t ic,
        int64_t id,
        int64_t ih,
        int64_t iw,
        int64_t oc,
        int64_t kd,
        int64_t kh,
        int64_t kw,
        int64_t od,
        int64_t oh,
        int64_t ow,
        int s0,
        int s1,
        int s2,
        int p0,
        int p1,
        int p2,
        int d0,
        int d1,
        int d2) {
    const int64_t total = n * oc * od * oh * ow;
    const int64_t block = 256;
    const int64_t nblk = (total + block - 1) / block;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int64_t idx = item.get_global_id(0);
                if (idx >= total) {
                    return;
                }

                int64_t rem = idx;
                const int64_t dst_x = rem % ow;
                rem /= ow;
                const int64_t dst_y = rem % oh;
                rem /= oh;
                const int64_t dst_z = rem % od;
                rem /= od;
                const int64_t cur_oc = rem % oc;
                const int64_t batch_idx = rem / oc;

                float acc = 0.0f;
                for (int64_t cur_ic = 0; cur_ic < ic; ++cur_ic) {
                    for (int64_t kz = 0; kz < kd; ++kz) {
                        const int64_t sz = dst_z * s2 + kz * d2 - p2;
                        if (sz < 0 || sz >= id) {
                            continue;
                        }
                        for (int64_t ky = 0; ky < kh; ++ky) {
                            const int64_t sy = dst_y * s1 + ky * d1 - p1;
                            if (sy < 0 || sy >= ih) {
                                continue;
                            }
                            for (int64_t kx = 0; kx < kw; ++kx) {
                                const int64_t sx = dst_x * s0 + kx * d0 - p0;
                                if (sx < 0 || sx >= iw) {
                                    continue;
                                }

                                const int64_t src_index = sx + iw * (sy + ih * (sz + id * (batch_idx * ic + cur_ic)));
                                const int64_t kernel_index = kx + kw * (ky + kh * (kz + kd * (cur_ic + ic * cur_oc)));
                                acc += src[src_index] * to_float(kernel[kernel_index]);
                            }
                        }
                    }
                }

                dst[idx] = acc;
            });
}
}

void ggml_sycl_op_conv3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * src = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int s0 = ggml_get_op_params_i32(dst, 0);
    const int s1 = ggml_get_op_params_i32(dst, 1);
    const int s2 = ggml_get_op_params_i32(dst, 2);
    const int p0 = ggml_get_op_params_i32(dst, 3);
    const int p1 = ggml_get_op_params_i32(dst, 4);
    const int p2 = ggml_get_op_params_i32(dst, 5);
    const int d0 = ggml_get_op_params_i32(dst, 6);
    const int d1 = ggml_get_op_params_i32(dst, 7);
    const int d2 = ggml_get_op_params_i32(dst, 8);
    const int ic = ggml_get_op_params_i32(dst, 9);
    const int n = ggml_get_op_params_i32(dst, 10);
    const int oc = ggml_get_op_params_i32(dst, 11);

    sycl::queue & q = ctx.stream();
    if (kernel->type == GGML_TYPE_F32) {
        conv3d_dispatch(q, (const float *) kernel->data, (const float *) src->data, (float *) dst->data,
                n, ic, src->ne[2], src->ne[1], src->ne[0], oc, kernel->ne[2], kernel->ne[1], kernel->ne[0],
                dst->ne[2], dst->ne[1], dst->ne[0], s0, s1, s2, p0, p1, p2, d0, d1, d2);
    } else {
        conv3d_dispatch(q, (const sycl::half *) kernel->data, (const float *) src->data, (float *) dst->data,
                n, ic, src->ne[2], src->ne[1], src->ne[0], oc, kernel->ne[2], kernel->ne[1], kernel->ne[0],
                dst->ne[2], dst->ne[1], dst->ne[0], s0, s1, s2, p0, p1, p2, d0, d1, d2);
    }
}

bool ggml_sycl_supports_conv3d(const ggml_tensor * op) {
    return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
           op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
}
