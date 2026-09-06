#include "conv2d-dw.hpp"

#include <sycl/sycl.hpp>

namespace {
static inline int max_i32(int a, int b) { return a > b ? a : b; }
static inline int min_i32(int a, int b) { return a < b ? a : b; }

struct conv_params {
    int in_w;
    int in_h;
    int out_w;
    int out_h;
    int kernel_w;
    int kernel_h;
    int stride_x;
    int stride_y;
    int padding_x;
    int padding_y;
    int dilation_x;
    int dilation_y;
    int channels;
    int batches;
};

struct kernel_bounds {
    int y_min;
    int y_max;
    int x_min;
    int x_max;
};

static inline kernel_bounds calculate_kernel_bounds(int out_x, int out_y, const conv_params & params) {
    kernel_bounds bounds;
    bounds.y_min = max_i32(0, (params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
    bounds.y_max = min_i32(params.kernel_h, (params.in_h + params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
    bounds.x_min = max_i32(0, (params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
    bounds.x_max = min_i32(params.kernel_w, (params.in_w + params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
    return bounds;
}

static inline int calculate_input_coord(int out_coord, int kern_coord, int stride, int dilation, int padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    static inline int input_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.in_w * params.in_h) + c * params.in_w * params.in_h + y * params.in_w + x;
    }

    static inline int kernel_index(int c, int ky, int kx, const conv_params & params) {
        return c * params.kernel_h * params.kernel_w + ky * params.kernel_w + kx;
    }

    static inline int output_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.out_w * params.out_h) + c * params.out_w * params.out_h + y * params.out_w + x;
    }

    static inline void unpack_indices(int global_idx, const conv_params & params, int & n, int & c, int & out_y, int & out_x) {
        out_x = global_idx % params.out_w;
        out_y = (global_idx / params.out_w) % params.out_h;
        c = (global_idx / (params.out_w * params.out_h)) % params.channels;
        n = global_idx / (params.out_w * params.out_h * params.channels);
    }
};

struct cwhn_layout {
    static inline int input_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.in_w * params.in_h) + (y * params.in_w + x) * params.channels + c;
    }

    static inline int kernel_index(int c, int ky, int kx, const conv_params & params) {
        return (ky * params.kernel_w + kx) * params.channels + c;
    }

    static inline int output_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.out_w * params.out_h) + y * (params.out_w * params.channels) + x * params.channels + c;
    }

    static inline void unpack_indices(int global_idx, const conv_params & params, int & n, int & c, int & out_y, int & out_x) {
        c = global_idx % params.channels;
        out_x = (global_idx / params.channels) % params.out_w;
        out_y = (global_idx / (params.channels * params.out_w)) % params.out_h;
        n = global_idx / (params.channels * params.out_w * params.out_h);
    }
};

template <typename Layout>
void conv2d_dw_dispatch(sycl::queue & q, const float * input, const float * kernel, float * output, const conv_params & params) {
    const int total = params.batches * params.channels * params.out_h * params.out_w;
    const int block = 256;
    const int nblk = (total + block - 1) / block;

    q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblk * block), sycl::range<1>(block)),
            [=](sycl::nd_item<1> item) {
                const int global_idx = (int) item.get_global_id(0);
                if (global_idx >= total) {
                    return;
                }

                int batch_idx;
                int channel_idx;
                int out_y_idx;
                int out_x_idx;
                Layout::unpack_indices(global_idx, params, batch_idx, channel_idx, out_y_idx, out_x_idx);

                float accumulator = 0.0f;
                const kernel_bounds bounds = calculate_kernel_bounds(out_x_idx, out_y_idx, params);

                for (int kern_y = bounds.y_min; kern_y < bounds.y_max; ++kern_y) {
                    const int in_y_idx = calculate_input_coord(out_y_idx, kern_y, params.stride_y, params.dilation_y, params.padding_y);
                    for (int kern_x = bounds.x_min; kern_x < bounds.x_max; ++kern_x) {
                        const int in_x_idx = calculate_input_coord(out_x_idx, kern_x, params.stride_x, params.dilation_x, params.padding_x);
                        accumulator += input[Layout::input_index(batch_idx, channel_idx, in_y_idx, in_x_idx, params)] *
                                       kernel[Layout::kernel_index(channel_idx, kern_y, kern_x, params)];
                    }
                }

                output[Layout::output_index(batch_idx, channel_idx, out_y_idx, out_x_idx, params)] = accumulator;
            });
}
}

void ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const conv_params params = {
        (int) input->ne[0],
        (int) input->ne[1],
        (int) dst->ne[0],
        (int) dst->ne[1],
        (int) kernel->ne[0],
        (int) kernel->ne[1],
        ggml_get_op_params_i32(dst, 0),
        ggml_get_op_params_i32(dst, 1),
        ggml_get_op_params_i32(dst, 2),
        ggml_get_op_params_i32(dst, 3),
        ggml_get_op_params_i32(dst, 4),
        ggml_get_op_params_i32(dst, 5),
        (int) dst->ne[2],
        (int) dst->ne[3],
    };

    sycl::queue & q = ctx.stream();
    if (ggml_is_contiguous(input)) {
        conv2d_dw_dispatch<whcn_layout>(q, (const float *) input->data, (const float *) kernel->data, (float *) dst->data, params);
    } else if (ggml_is_contiguous_channels(input)) {
        conv2d_dw_dispatch<cwhn_layout>(q, (const float *) input->data, (const float *) kernel->data, (float *) dst->data, params);
    } else {
        GGML_ABORT("Unsupported memory layout for conv_2d_dw");
    }
}

bool ggml_sycl_supports_conv2d_dw(const ggml_tensor * op) {
    return op->src[0]->type == GGML_TYPE_F32 &&
           op->src[1]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           (ggml_is_contiguous(op->src[1]) || ggml_is_contiguous_channels(op->src[1]));
}
