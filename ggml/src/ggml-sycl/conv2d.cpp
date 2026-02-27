#include "conv2d.hpp"

#include <sycl/sycl.hpp>

#include "ggml.h"

#define SYCL_CONV2D_BLOCK_SIZE 256

// opt_024_003: Use int32_t for index arithmetic (faster integer division on Xe2)
// Runtime assertion added at conv_params construction to prevent overflow
struct conv_params {
    int32_t IW, IH;
    int32_t OW, OH;
    int32_t KW, KH;
    int32_t ST_X, ST_Y;
    int32_t PD_X, PD_Y;
    int32_t DL_X, DL_Y;
    int32_t IC, OC;
    int32_t B;
    int32_t TOTAL;
};

struct kernel_bounds {
    int32_t y_min, y_max;
    int32_t x_min, x_max;
};

// opt_024_002: Add constexpr bounds for compile-time known kernel sizes
static inline kernel_bounds calculate_kernel_bounds(int32_t out_x, int32_t out_y, const conv_params & P) {
    kernel_bounds bounds;
    const auto max32 = [](int32_t a, int32_t b) { return (a > b) ? a : b; };
    const auto min32 = [](int32_t a, int32_t b) { return (a < b) ? a : b; };
    bounds.y_min = max32(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min32(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max32(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min32(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

// opt_024_002: Specialization for 1x1 kernels (no bounds checking needed, fastest path)
static inline kernel_bounds calculate_kernel_bounds_1x1(int32_t out_x, int32_t out_y, const conv_params & P) {
    (void)out_x; (void)out_y; (void)P;
    kernel_bounds bounds = {0, 1, 0, 1};
    return bounds;
}

// opt_024_002: Specialization for 3x3 kernels
static inline kernel_bounds calculate_kernel_bounds_3x3(int32_t out_x, int32_t out_y, const conv_params & P) {
    kernel_bounds bounds;
    const auto max32 = [](int32_t a, int32_t b) { return (a > b) ? a : b; };
    const auto min32 = [](int32_t a, int32_t b) { return (a < b) ? a : b; };
    bounds.y_min = max32(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min32(3, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max32(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min32(3, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

// opt_024_002: Specialization for 5x5 kernels
static inline kernel_bounds calculate_kernel_bounds_5x5(int32_t out_x, int32_t out_y, const conv_params & P) {
    kernel_bounds bounds;
    const auto max32 = [](int32_t a, int32_t b) { return (a > b) ? a : b; };
    const auto min32 = [](int32_t a, int32_t b) { return (a < b) ? a : b; };
    bounds.y_min = max32(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min32(5, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max32(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min32(5, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

// opt_024_002: Specialization for 7x7 kernels
static inline kernel_bounds calculate_kernel_bounds_7x7(int32_t out_x, int32_t out_y, const conv_params & P) {
    kernel_bounds bounds;
    const auto max32 = [](int32_t a, int32_t b) { return (a > b) ? a : b; };
    const auto min32 = [](int32_t a, int32_t b) { return (a < b) ? a : b; };
    bounds.y_min = max32(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min32(7, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max32(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min32(7, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

static inline int32_t calculate_input_coord(int32_t out_coord, int32_t kern_coord,
                                             int32_t stride, int32_t dilation, int32_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

// WHCN layout indexing helpers - using int64_t for the final index (size_t equivalent)
static inline int64_t whcn_input_index(int32_t n, int32_t c, int32_t y, int32_t x, const conv_params & P) {
    return (int64_t)n * (P.IC * P.IW * P.IH) + (int64_t)c * P.IW * P.IH + (int64_t)y * P.IW + x;
}

static inline int64_t whcn_kernel_index(int32_t c_out, int32_t c_in, int32_t ky, int32_t kx, const conv_params & P) {
    return (int64_t)c_out * (P.IC * P.KH * P.KW) + (int64_t)c_in * (P.KH * P.KW) + (int64_t)ky * P.KW + kx;
}

static inline int64_t whcn_output_index(int32_t n, int32_t c, int32_t y, int32_t x, const conv_params & P) {
    return (int64_t)n * (P.OC * P.OW * P.OH) + (int64_t)c * P.OW * P.OH + (int64_t)y * P.OW + x;
}

static inline void whcn_unpack_indices(int64_t global_idx, const conv_params & P,
                                        int32_t & n, int32_t & c, int32_t & out_y, int32_t & out_x) {
    out_x = global_idx % P.OW;
    out_y = (global_idx / P.OW) % P.OH;
    c     = (global_idx / (P.OW * P.OH)) % P.OC;
    n     = global_idx / (P.OW * P.OH * P.OC);
}

// opt_024_002: Template kernel with compile-time kernel size specialization
// opt_024_004: Added [[sycl::reqd_sub_group_size(16)]] for SIMD16 on Xe2
// opt_024_005: calculate_kernel_bounds hoisted outside c_in loop
// opt_024_006: Using sycl::fma for fused multiply-add
template <typename T, int32_t KW, int32_t KH>
[[sycl::reqd_sub_group_size(16)]]
static void conv2d_kernel(const float * __restrict__ input,
                          const T * __restrict__ kernel_data,
                          float * __restrict__ output,
                          const conv_params P,
                          const sycl::nd_item<1> & item) {
    const int64_t global_idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

    if (global_idx >= P.TOTAL) {
        return;
    }

    int32_t n, c_out, out_y, out_x;
    whcn_unpack_indices(global_idx, P, n, c_out, out_y, out_x);

    float acc = 0.0f;

    // opt_024_005: Hoist kernel bounds calculation outside c_in loop
    // opt_024_002: Use constexpr bounds for known kernel sizes
    kernel_bounds bounds;
    if constexpr (KW == 1 && KH == 1) {
        bounds = {0, 1, 0, 1};
    } else if constexpr (KW == 3 && KH == 3) {
        bounds = calculate_kernel_bounds_3x3(out_x, out_y, P);
    } else if constexpr (KW == 5 && KH == 5) {
        bounds = calculate_kernel_bounds_5x5(out_x, out_y, P);
    } else if constexpr (KW == 7 && KH == 7) {
        bounds = calculate_kernel_bounds_7x7(out_x, out_y, P);
    } else {
        bounds = calculate_kernel_bounds(out_x, out_y, P);
    }

    for (int32_t c_in = 0; c_in < P.IC; ++c_in) {
        for (int32_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            const int32_t in_y = calculate_input_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);

            for (int32_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                const int32_t in_x = calculate_input_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);

                const float input_val  = input[whcn_input_index(n, c_in, in_y, in_x, P)];
                const T     kernel_val = kernel_data[whcn_kernel_index(c_out, c_in, ky, kx, P)];
                // opt_024_006: Use explicit sycl::fma for fused multiply-add
                acc = sycl::fma(input_val, static_cast<float>(kernel_val), acc);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

// opt_024_002: Dispatcher for specialized kernels based on kernel size
template <typename T>
static void conv2d_sycl_dispatch(const float * input, const T * kernel_data, float * output,
                                  const conv_params P, queue_ptr stream) {
    const int64_t num_blocks = ceil_div((size_t) P.TOTAL, (size_t) SYCL_CONV2D_BLOCK_SIZE);

    // opt_024_002: Switch on kernel size to dispatch specialized kernels
    // opt_024_004: reqd_sub_group_size(16) attribute applied in kernel lambda
    if (P.KW == 1 && P.KH == 1) {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE),
                              sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                conv2d_kernel<T, 1, 1>(input, kernel_data, output, P, item);
            });
    } else if (P.KW == 3 && P.KH == 3) {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE),
                              sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                conv2d_kernel<T, 3, 3>(input, kernel_data, output, P, item);
            });
    } else if (P.KW == 5 && P.KH == 5) {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE),
                              sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                conv2d_kernel<T, 5, 5>(input, kernel_data, output, P, item);
            });
    } else if (P.KW == 7 && P.KH == 7) {
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE),
                              sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                conv2d_kernel<T, 7, 7>(input, kernel_data, output, P, item);
            });
    } else {
        // Fallback: generic kernel with dynamic bounds
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE),
                              sycl::range<1>(SYCL_CONV2D_BLOCK_SIZE)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                conv2d_kernel<T, 0, 0>(input, kernel_data, output, P, item);
            });
    }
}

static void conv2d_sycl_f16(const float * input, const sycl::half * kernel_data, float * output,
                             const conv_params P, queue_ptr stream) {
    conv2d_sycl_dispatch<sycl::half>(input, kernel_data, output, P, stream);
}

static void conv2d_sycl_f32(const float * input, const float * kernel_data, float * output,
                             const conv_params P, queue_ptr stream) {
    conv2d_sycl_dispatch<float>(input, kernel_data, output, P, stream);
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

    const int64_t total = (int64_t)B * OC * OH * OW;

    // opt_024_003: Runtime overflow check for int32_t conversion
    GGML_ASSERT((int64_t)IW * IH * IC <= INT32_MAX && "Input tensor too large for int32_t");
    GGML_ASSERT((int64_t)KW * KH * IC * OC <= INT32_MAX && "Kernel tensor too large for int32_t");
    GGML_ASSERT((int64_t)OW * OH * OC * B <= INT32_MAX && "Output tensor too large for int32_t");

    const conv_params params = {
        (int32_t)IW, (int32_t)IH, (int32_t)OW, (int32_t)OH,
        (int32_t)KW, (int32_t)KH,
        (int32_t)ST_X, (int32_t)ST_Y,
        (int32_t)PD_X, (int32_t)PD_Y,
        (int32_t)DL_X, (int32_t)DL_Y,
        (int32_t)IC, (int32_t)OC,
        (int32_t)B, (int32_t)total
    };

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_sycl_f16(X_D, (sycl::half *) K_D, Y_D, params, stream);
    } else {
        conv2d_sycl_f32(X_D, K_D, Y_D, params, stream);
    }
}
