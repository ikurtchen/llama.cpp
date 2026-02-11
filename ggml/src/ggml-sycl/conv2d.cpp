#include "conv2d.hpp"

#include <sycl/sycl.hpp>

#include "ggml.h"

#define SYCL_CONV2D_BLOCK_SIZE 256

struct conv_params {
    int64_t IW, IH;
    int64_t OW, OH;
    int64_t KW, KH;
    int64_t ST_X, ST_Y;
    int64_t PD_X, PD_Y;
    int64_t DL_X, DL_Y;
    int64_t IC, OC;
    int64_t B;
    int64_t TOTAL;
};

struct kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

static inline kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, const conv_params & P) {
    kernel_bounds bounds;
    const auto max64 = [](int64_t a, int64_t b) { return (a > b) ? a : b; };
    const auto min64 = [](int64_t a, int64_t b) { return (a < b) ? a : b; };
    bounds.y_min = max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

static inline int64_t calculate_input_coord(int64_t out_coord, int64_t kern_coord,
                                             int64_t stride, int64_t dilation, int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

// WHCN layout indexing helpers
static inline int64_t whcn_input_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
    return n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

static inline int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, const conv_params & P) {
    return c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

static inline int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv_params & P) {
    return n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

static inline void whcn_unpack_indices(int64_t global_idx, const conv_params & P,
                                        int64_t & n, int64_t & c, int64_t & out_y, int64_t & out_x) {
    out_x = global_idx % P.OW;
    out_y = (global_idx / P.OW) % P.OH;
    c     = (global_idx / (P.OW * P.OH)) % P.OC;
    n     = global_idx / (P.OW * P.OH * P.OC);
}

template <typename T>
static void conv2d_kernel(const float * __restrict__ input,
                          const T * __restrict__ kernel_data,
                          float * __restrict__ output,
                          const conv_params P,
                          const sycl::nd_item<1> & item) {
    const int64_t global_idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

    if (global_idx >= P.TOTAL) {
        return;
    }

    int64_t n, c_out, out_y, out_x;
    whcn_unpack_indices(global_idx, P, n, c_out, out_y, out_x);

    float acc = 0.0f;

    for (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

        for (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            const int64_t in_y = calculate_input_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);

            for (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                const int64_t in_x = calculate_input_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);

                const float input_val  = input[whcn_input_index(n, c_in, in_y, in_x, P)];
                const T     kernel_val = kernel_data[whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += input_val * static_cast<float>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T>
static void conv2d_sycl_internal(const float * input, const T * kernel_data, float * output,
                                  const conv_params P, queue_ptr stream) {
    const int64_t num_blocks = ceil_div((size_t) P.TOTAL, (size_t) SYCL_CONV2D_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE),
                          sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) {
            conv2d_kernel<T>(input, kernel_data, output, P, item);
        });
}

static void conv2d_sycl_f16(const float * input, const sycl::half * kernel_data, float * output,
                             const conv_params P, queue_ptr stream) {
    conv2d_sycl_internal<sycl::half>(input, kernel_data, output, P, stream);
}

static void conv2d_sycl_f32(const float * input, const float * kernel_data, float * output,
                             const conv_params P, queue_ptr stream) {
    conv2d_sycl_internal<float>(input, kernel_data, output, P, stream);
}

void ggml_sycl_op_conv_2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    float *             K_D    = (float *) kernel->data;
    const float *       X_D    = (const float *) input->data;
    float *             Y_D    = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int       ST_X = p[0];  // stride_x
    const int       ST_Y = p[1];  // stride_y
    const int       PD_X = p[2];  // padding_x
    const int       PD_Y = p[3];  // padding_y
    const int       DL_X = p[4];  // dilation_x
    const int       DL_Y = p[5];  // dilation_y

    // No cwhn
    GGML_ASSERT(p[6] == false);

    const int IW = input->ne[0];   // input_w
    const int IH = input->ne[1];   // input_h
    const int OW = dst->ne[0];     // output_w
    const int OH = dst->ne[1];     // output_h
    const int KW = kernel->ne[0];  // kernel_w
    const int KH = kernel->ne[1];  // kernel_h
    const int IC = input->ne[2];   // input_channels
    const int OC = kernel->ne[3];  // output_channels
    const int B  = input->ne[3];   // n_batches

    const int64_t   total  = B * OC * OH * OW;
    const conv_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_sycl_f16(X_D, (sycl::half *) K_D, Y_D, params, stream);
    } else {
        conv2d_sycl_f32(X_D, K_D, Y_D, params, stream);
    }
}
