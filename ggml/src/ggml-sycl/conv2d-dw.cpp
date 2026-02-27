#include "conv2d-dw.hpp"

#include <sycl/sycl.hpp>

#include "ggml.h"

#define SYCL_CONV2D_DW_BLOCK_SIZE 256

struct conv2d_dw_params {
    int64_t IW, IH;
    int64_t OW, OH;
    int64_t KW, KH;
    int64_t ST_X, ST_Y;
    int64_t PD_X, PD_Y;
    int64_t DL_X, DL_Y;
    int64_t C;
    int64_t B;
    int64_t TOTAL;
};

struct dw_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

static inline dw_kernel_bounds calculate_dw_kernel_bounds(int64_t out_x, int64_t out_y, const conv2d_dw_params & P) {
    dw_kernel_bounds bounds;
    const auto max64 = [](int64_t a, int64_t b) { return (a > b) ? a : b; };
    const auto min64 = [](int64_t a, int64_t b) { return (a < b) ? a : b; };
    bounds.y_min = max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    return bounds;
}

static inline int64_t calculate_dw_input_coord(int64_t out_coord, int64_t kern_coord,
                                                int64_t stride, int64_t dilation, int64_t padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

// WHCN layout indexing helpers (width-height-channel-batch, i.e. contiguous)
static inline int64_t whcn_input_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv2d_dw_params & P) {
    return n * (P.C * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

static inline int64_t whcn_kernel_index(int64_t c, int64_t ky, int64_t kx, const conv2d_dw_params & P) {
    return c * P.KH * P.KW + ky * P.KW + kx;
}

static inline int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv2d_dw_params & P) {
    return n * (P.C * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

static inline void whcn_unpack_indices(int64_t global_idx, const conv2d_dw_params & P,
                                       int64_t & n, int64_t & c, int64_t & out_y, int64_t & out_x) {
    out_x = global_idx % P.OW;
    out_y = (global_idx / P.OW) % P.OH;
    c     = (global_idx / (P.OW * P.OH)) % P.C;
    n     = global_idx / (P.OW * P.OH * P.C);
}

// CWHN layout indexing helpers (channel-width-height-batch, i.e. contiguous channels)
static inline int64_t cwhn_input_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv2d_dw_params & P) {
    return n * (P.C * P.IW * P.IH) + (y * P.IW + x) * P.C + c;
}

static inline int64_t cwhn_kernel_index(int64_t c, int64_t ky, int64_t kx, const conv2d_dw_params & P) {
    return (ky * P.KW + kx) * P.C + c;
}

static inline int64_t cwhn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, const conv2d_dw_params & P) {
    return n * (P.C * P.OW * P.OH) + y * (P.OW * P.C) + x * P.C + c;
}

static inline void cwhn_unpack_indices(int64_t global_idx, const conv2d_dw_params & P,
                                       int64_t & n, int64_t & c, int64_t & out_y, int64_t & out_x) {
    c     = global_idx % P.C;
    out_x = (global_idx / P.C) % P.OW;
    out_y = (global_idx / (P.C * P.OW)) % P.OH;
    n     = global_idx / (P.C * P.OW * P.OH);
}

template <bool is_cwhn>
static void conv2d_dw_kernel(const float * __restrict__ input,
                             const float * __restrict__ kernel_data,
                             float * __restrict__ output,
                             const conv2d_dw_params P,
                             const sycl::nd_item<1> & item) {
    const int64_t global_idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

    if (global_idx >= P.TOTAL) {
        return;
    }

    int64_t n, c, out_y, out_x;
    if constexpr (is_cwhn) {
        cwhn_unpack_indices(global_idx, P, n, c, out_y, out_x);
    } else {
        whcn_unpack_indices(global_idx, P, n, c, out_y, out_x);
    }

    float acc = 0.0f;
    dw_kernel_bounds bounds = calculate_dw_kernel_bounds(out_x, out_y, P);

    // opt_task_015: Enable unroll for kernel height loop
    // See: hw_spec_b60.md, optimization_guide
    #pragma unroll
    for (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        const int64_t in_y = calculate_dw_input_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);

        // opt_task_015: Enable unroll for kernel width loop
        // See: hw_spec_b60.md, optimization_guide
        #pragma unroll
        for (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            const int64_t in_x = calculate_dw_input_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);

            float input_val;
            float kernel_val;
            if constexpr (is_cwhn) {
                input_val  = input[cwhn_input_index(n, c, in_y, in_x, P)];
                kernel_val = kernel_data[cwhn_kernel_index(c, ky, kx, P)];
            } else {
                input_val  = input[whcn_input_index(n, c, in_y, in_x, P)];
                kernel_val = kernel_data[whcn_kernel_index(c, ky, kx, P)];
            }

            acc += input_val * kernel_val;
        }
    }

    if constexpr (is_cwhn) {
        output[cwhn_output_index(n, c, out_y, out_x, P)] = acc;
    } else {
        output[whcn_output_index(n, c, out_y, out_x, P)] = acc;
    }
}

template <bool is_cwhn>
static void conv2d_dw_sycl(const float * input, const float * kernel_data, float * output,
                            const conv2d_dw_params P, queue_ptr stream) {
    const int64_t num_blocks = ceil_div((size_t) P.TOTAL, (size_t) SYCL_CONV2D_DW_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CONV2D_DW_BLOCK_SIZE),
                          sycl::range<1>(SYCL_CONV2D_DW_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) {
            conv2d_dw_kernel<is_cwhn>(input, kernel_data, output, P, item);
        });
}

void ggml_sycl_op_conv_2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const float * w_d = (const float *) kernel->data;
    const float * x_d = (const float *) input->data;
    float *       y_d = (float *) dst->data;

    queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int32_t * p    = (const int32_t *) dst->op_params;
    const int ST_X = p[0];
    const int ST_Y = p[1];
    const int PD_X = p[2];
    const int PD_Y = p[3];
    const int DL_X = p[4];
    const int DL_Y = p[5];

    const int IW = input->ne[0];
    const int IH = input->ne[1];
    const int OW = dst->ne[0];
    const int OH = dst->ne[1];
    const int KW = kernel->ne[0];
    const int KH = kernel->ne[1];
    const int C  = dst->ne[2];
    const int B  = dst->ne[3];

    const int64_t total = (int64_t) B * C * OH * OW;
    const conv2d_dw_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, C, B, total };

    if (ggml_is_contiguous(input)) {
        conv2d_dw_sycl<false>(x_d, w_d, y_d, params, stream);
    } else if (ggml_is_contiguous_channels(input)) {
        conv2d_dw_sycl<true>(x_d, w_d, y_d, params, stream);
    } else {
        GGML_ABORT("Unsupported memory layout for conv_2d_dw");
    }
}
