#include "conv2d-transpose.hpp"

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
void conv2d_transpose_dispatch(
        sycl::queue & q,
        const float * input,
        const T * kernel,
        float * output,
        int in_w,
        int in_h,
        int out_w,
        int out_h,
        int kernel_w,
        int kernel_h,
        int stride,
        int c_in,
        int c_out,
        int batches) {
    const int64_t total = (int64_t) out_w * out_h * c_out * batches;
    const int64_t block = 256;
    const int64_t nblk = (total + block - 1) / block;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int64_t global_idx = item.get_global_id(0);
                if (global_idx >= total) {
                    return;
                }

                const int out_x_idx = global_idx % out_w;
                const int out_y_idx = (global_idx / out_w) % out_h;
                const int c_idx = (global_idx / (out_w * out_h)) % c_out;
                const int n_idx = global_idx / (out_w * out_h * c_out);

                float accumulator = 0.0f;
                for (int c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        int in_y = out_y_idx - kh;
                        if (in_y < 0 || in_y % stride != 0) {
                            continue;
                        }
                        in_y /= stride;
                        if (in_y >= in_h) {
                            continue;
                        }

                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int in_x = out_x_idx - kw;
                            if (in_x < 0 || in_x % stride != 0) {
                                continue;
                            }
                            in_x /= stride;
                            if (in_x >= in_w) {
                                continue;
                            }

                            const int input_idx = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                            const int kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx + kernel_w * kh + kw;
                            accumulator += input[input_idx] * to_float(kernel[kernel_idx]);
                        }
                    }
                }

                output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y_idx + out_x_idx] = accumulator;
            });
}
}

void ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int input_w = (int) input->ne[0];
    const int input_h = (int) input->ne[1];
    const int output_w = (int) dst->ne[0];
    const int output_h = (int) dst->ne[1];
    const int channels_in = (int) input->ne[2];
    const int channels_out = (int) kernel->ne[2];
    const int kernel_w = (int) kernel->ne[0];
    const int kernel_h = (int) kernel->ne[1];
    const int stride = ggml_get_op_params_i32(dst, 0);
    const int batches = (int) input->ne[3];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    sycl::queue & q = ctx.stream();
    if (kernel->type == GGML_TYPE_F32) {
        conv2d_transpose_dispatch(q, (const float *) input->data, (const float *) kernel->data, (float *) dst->data,
                input_w, input_h, output_w, output_h, kernel_w, kernel_h, stride, channels_in, channels_out, batches);
    } else {
        conv2d_transpose_dispatch(q, (const float *) input->data, (const sycl::half *) kernel->data, (float *) dst->data,
                input_w, input_h, output_w, output_h, kernel_w, kernel_h, stride, channels_in, channels_out, batches);
    }
}

bool ggml_sycl_supports_conv2d_transpose(const ggml_tensor * op) {
    const int64_t cout = op->src[0]->ne[2];
    const int64_t cin  = op->src[0]->ne[3];

    return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
           op->src[1]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           cout == op->ne[2] &&
           cin == op->src[1]->ne[2] &&
           ggml_is_contiguous(op->src[0]) &&
           ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op);
}
