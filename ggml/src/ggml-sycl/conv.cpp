#include "conv.hpp"

// ============================================================================
// CONV_TRANSPOSE_1D (existing)
// ============================================================================

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // weight
    const ggml_tensor * src1 = dst->src[1]; // input

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float       * dst_d  = (float *)dst->data;

    sycl::queue & stream = *ctx.stream();

    const int32_t * opts = (const int32_t *)dst->op_params;
    const int s0 = opts[0];

    // Extract tensor dimensions to local variables (never capture ggml_tensor* into device code)
    const int src0_ne0 = src0->ne[0];
    const int src0_ne1 = src0->ne[1];
    const int src0_ne2 = src0->ne[2];
    const int src1_ne0 = src1->ne[0];
    const int dst_ne0  = dst->ne[0];

    const int64_t output_size = ggml_nelements(dst);

    const int block_size = SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    const int num_blocks = (output_size + block_size - 1) / block_size;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * block_size, block_size),
        [=](sycl::nd_item<1> item) {
            const int global_index = item.get_global_id(0);
            if (global_index >= output_size) {
                return;
            }

            const int out_index = global_index / dst_ne0;

            float accumulator = 0;

            #pragma unroll 4
            for (int c = 0; c < src0_ne2; c++) {
                const int idx = global_index % dst_ne0;

                const int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
                const int input_offset = src1_ne0 * c;

                #pragma unroll 4
                for (int i = 0; i < src1_ne0; i++) {
                    if (!(idx >= i * s0 && idx < i * s0 + src0_ne0)) {
                        continue;
                    }
                    const int weight_idx = idx - i * s0;

                    const float kernel_weight = src0_d[kernel_offset + weight_idx];
                    const float input_value   = src1_d[input_offset + i];

                    accumulator += kernel_weight * input_value;
                }
            }
            dst_d[global_index] = accumulator;
        }
    );
}

// ============================================================================
// CONV_2D
// ============================================================================

// Helper structs for conv2d (matching CUDA conv2d.cu)
struct conv2d_params {
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

static inline int64_t sycl_max64(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}

static inline int64_t sycl_min64(int64_t a, int64_t b) {
    return (a < b) ? a : b;
}

template <typename T>
static void conv2d_sycl(const float * X_D, const T * K_D, float * Y_D,
                         const conv2d_params P, sycl::queue & stream) {
    const int block_size = SYCL_CONV2D_BLOCK_SIZE;
    const int num_blocks = (P.TOTAL + block_size - 1) / block_size;

    // Extract all params into local variables for lambda capture
    const int64_t IW = P.IW, IH = P.IH;
    const int64_t OW = P.OW, OH = P.OH;
    const int64_t KW = P.KW, KH = P.KH;
    const int64_t ST_X = P.ST_X, ST_Y = P.ST_Y;
    const int64_t PD_X = P.PD_X, PD_Y = P.PD_Y;
    const int64_t DL_X = P.DL_X, DL_Y = P.DL_Y;
    const int64_t IC = P.IC, OC = P.OC;
    const int64_t B = P.B;
    const int64_t TOTAL = P.TOTAL;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * block_size, block_size),
        [=](sycl::nd_item<1> item) {
            const int64_t global_idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

            if (global_idx >= TOTAL) {
                return;
            }

            // unpack_indices (whcn_layout)
            const int64_t out_x = global_idx % OW;
            const int64_t out_y = (global_idx / OW) % OH;
            const int64_t c_out = (global_idx / (OW * OH)) % OC;
            const int64_t n     = global_idx / (OW * OH * OC);

            float acc = 0.0f;

            #pragma unroll 4
            for (int64_t c_in = 0; c_in < IC; ++c_in) {
                // calculate_kernel_bounds
                const int64_t y_min = sycl_max64(0, (PD_Y - out_y * ST_Y + DL_Y - 1) / DL_Y);
                const int64_t y_max = sycl_min64(KH, (IH + PD_Y - out_y * ST_Y + DL_Y - 1) / DL_Y);
                const int64_t x_min = sycl_max64(0, (PD_X - out_x * ST_X + DL_X - 1) / DL_X);
                const int64_t x_max = sycl_min64(KW, (IW + PD_X - out_x * ST_X + DL_X - 1) / DL_X);

                #pragma unroll 4
                for (int64_t ky = y_min; ky < y_max; ++ky) {
                    // calculate_input_coord
                    const int64_t in_y = out_y * ST_Y + ky * DL_Y - PD_Y;

                    #pragma unroll 4
                    for (int64_t kx = x_min; kx < x_max; ++kx) {
                        const int64_t in_x = out_x * ST_X + kx * DL_X - PD_X;

                        // whcn_layout::input_index
                        const int64_t input_idx = n * (IC * IW * IH) + c_in * IW * IH + in_y * IW + in_x;
                        // whcn_layout::kernel_index
                        const int64_t kernel_idx = c_out * (IC * KH * KW) + c_in * (KH * KW) + ky * KW + kx;

                        const float input_val = X_D[input_idx];
                        const float kernel_val = static_cast<float>(K_D[kernel_idx]);
                        acc += input_val * kernel_val;
                    }
                }
            }

            // whcn_layout::output_index
            const int64_t output_idx = n * (OC * OW * OH) + c_out * OW * OH + out_y * OW + out_x;
            Y_D[output_idx] = acc;
        }
    );
}

void ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(input->ne[2] == kernel->ne[2]);

    const float * X_D = (const float *) input->data;
    float       * Y_D = (float *) dst->data;

    sycl::queue & stream = *ctx.stream();

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
    conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_sycl<sycl::half>(X_D, (const sycl::half *) kernel->data, Y_D, params, stream);
    } else {
        conv2d_sycl<float>(X_D, (const float *) kernel->data, Y_D, params, stream);
    }
}

// ============================================================================
// CONV_2D_DW (depthwise)
// ============================================================================

struct conv2d_dw_params {
    int in_w, in_h;
    int out_w, out_h;
    int kernel_w, kernel_h;
    int stride_x, stride_y;
    int padding_x, padding_y;
    int dilation_x, dilation_y;
    int channels, batches;
};

// whcn_layout kernel for depthwise conv2d
static void conv2d_dw_sycl_whcn(const float * input, const float * kernel_data, float * output,
                                  const conv2d_dw_params P, sycl::queue & stream) {
    const int total = P.batches * P.channels * P.out_h * P.out_w;
    const int block_size = SYCL_CONV2D_DW_BLOCK_SIZE;
    const int num_blocks = (total + block_size - 1) / block_size;

    // Extract all params into local variables for lambda capture
    const int in_w = P.in_w, in_h = P.in_h;
    const int out_w = P.out_w, out_h = P.out_h;
    const int kw = P.kernel_w, kh = P.kernel_h;
    const int stride_x = P.stride_x, stride_y = P.stride_y;
    const int padding_x = P.padding_x, padding_y = P.padding_y;
    const int dilation_x = P.dilation_x, dilation_y = P.dilation_y;
    const int channels = P.channels;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * block_size, block_size),
        [=](sycl::nd_item<1> item) {
            const int global_idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

            if (global_idx >= total) {
                return;
            }

            // whcn_layout::unpack_indices
            const int out_x_idx = global_idx % out_w;
            const int out_y_idx = (global_idx / out_w) % out_h;
            const int channel_idx = (global_idx / (out_w * out_h)) % channels;
            const int batch_idx = global_idx / (out_w * out_h * channels);

            float accumulator = 0.0f;

            // calculate_kernel_bounds
            const int y_min = sycl::max(0, (padding_y - out_y_idx * stride_y + dilation_y - 1) / dilation_y);
            const int y_max = sycl::min(kh, (in_h + padding_y - out_y_idx * stride_y + dilation_y - 1) / dilation_y);
            const int x_min = sycl::max(0, (padding_x - out_x_idx * stride_x + dilation_x - 1) / dilation_x);
            const int x_max = sycl::min(kw, (in_w + padding_x - out_x_idx * stride_x + dilation_x - 1) / dilation_x);

            for (int kern_y = y_min; kern_y < y_max; ++kern_y) {
                const int in_y_idx = out_y_idx * stride_y + kern_y * dilation_y - padding_y;

                for (int kern_x = x_min; kern_x < x_max; ++kern_x) {
                    const int in_x_idx = out_x_idx * stride_x + kern_x * dilation_x - padding_x;

                    // whcn_layout::input_index
                    const int input_idx = batch_idx * (channels * in_w * in_h) + channel_idx * in_w * in_h + in_y_idx * in_w + in_x_idx;
                    // whcn_layout::kernel_index (depthwise: no c_out dimension)
                    const int kernel_idx = channel_idx * kh * kw + kern_y * kw + kern_x;

                    accumulator += input[input_idx] * kernel_data[kernel_idx];
                }
            }

            // whcn_layout::output_index
            const int output_idx = batch_idx * (channels * out_w * out_h) + channel_idx * out_w * out_h + out_y_idx * out_w + out_x_idx;
            output[output_idx] = accumulator;
        }
    );
}

// cwhn_layout kernel for depthwise conv2d
static void conv2d_dw_sycl_cwhn(const float * input, const float * kernel_data, float * output,
                                  const conv2d_dw_params P, sycl::queue & stream) {
    const int total = P.batches * P.channels * P.out_h * P.out_w;
    const int block_size = SYCL_CONV2D_DW_BLOCK_SIZE;
    const int num_blocks = (total + block_size - 1) / block_size;

    // Extract all params into local variables for lambda capture
    const int in_w = P.in_w, in_h = P.in_h;
    const int out_w = P.out_w, out_h = P.out_h;
    const int kw = P.kernel_w, kh = P.kernel_h;
    const int stride_x = P.stride_x, stride_y = P.stride_y;
    const int padding_x = P.padding_x, padding_y = P.padding_y;
    const int dilation_x = P.dilation_x, dilation_y = P.dilation_y;
    const int channels = P.channels;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * block_size, block_size),
        [=](sycl::nd_item<1> item) {
            const int global_idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

            if (global_idx >= total) {
                return;
            }

            // cwhn_layout::unpack_indices
            const int channel_idx = global_idx % channels;
            const int out_x_idx = (global_idx / channels) % out_w;
            const int out_y_idx = (global_idx / (channels * out_w)) % out_h;
            const int batch_idx = global_idx / (channels * out_w * out_h);

            float accumulator = 0.0f;

            // calculate_kernel_bounds
            const int y_min = sycl::max(0, (padding_y - out_y_idx * stride_y + dilation_y - 1) / dilation_y);
            const int y_max = sycl::min(kh, (in_h + padding_y - out_y_idx * stride_y + dilation_y - 1) / dilation_y);
            const int x_min = sycl::max(0, (padding_x - out_x_idx * stride_x + dilation_x - 1) / dilation_x);
            const int x_max = sycl::min(kw, (in_w + padding_x - out_x_idx * stride_x + dilation_x - 1) / dilation_x);

            for (int kern_y = y_min; kern_y < y_max; ++kern_y) {
                const int in_y_idx = out_y_idx * stride_y + kern_y * dilation_y - padding_y;

                for (int kern_x = x_min; kern_x < x_max; ++kern_x) {
                    const int in_x_idx = out_x_idx * stride_x + kern_x * dilation_x - padding_x;

                    // cwhn_layout::input_index
                    const int input_idx = batch_idx * (channels * in_w * in_h) + (in_y_idx * in_w + in_x_idx) * channels + channel_idx;
                    // cwhn_layout::kernel_index (depthwise)
                    const int kernel_idx = (kern_y * kw + kern_x) * channels + channel_idx;

                    accumulator += input[input_idx] * kernel_data[kernel_idx];
                }
            }

            // cwhn_layout::output_index
            const int output_idx = batch_idx * (channels * out_w * out_h) + out_y_idx * (out_w * channels) + out_x_idx * channels + channel_idx;
            output[output_idx] = accumulator;
        }
    );
}

void ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const float * w_d = (const float *) kernel->data;
    const float * x_d = (const float *) input->data;
    float       * y_d = (float *) dst->data;

    sycl::queue & stream = *ctx.stream();

    const int32_t * p          = (const int32_t *) dst->op_params;
    const int       stride_x   = p[0];
    const int       stride_y   = p[1];
    const int       padding_x  = p[2];
    const int       padding_y  = p[3];
    const int       dilation_x = p[4];
    const int       dilation_y = p[5];

    const int in_w     = input->ne[0];
    const int in_h     = input->ne[1];
    const int kernel_w = kernel->ne[0];
    const int kernel_h = kernel->ne[1];
    const int out_w    = dst->ne[0];
    const int out_h    = dst->ne[1];
    const int channels = dst->ne[2];
    const int batches  = dst->ne[3];

    conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                stride_x, stride_y, padding_x, padding_y,
                                dilation_x, dilation_y, channels, batches };

    if (ggml_is_contiguous(input)) {
        conv2d_dw_sycl_whcn(x_d, w_d, y_d, params, stream);
    } else if (ggml_is_contiguous_channels(input)) {
        conv2d_dw_sycl_cwhn(x_d, w_d, y_d, params, stream);
    } else {
        GGML_ABORT("Unsupported memory layout for conv_2d_dw");
    }
}

// ============================================================================
// CONV_TRANSPOSE_2D
// ============================================================================

void ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float      * input_data  = (const float *) input->data;
    const sycl::half * kernel_data = (const sycl::half *) kernel->data;
    float            * output_data = (float *) dst->data;

    sycl::queue & stream = *ctx.stream();

    const int in_w         = input->ne[0];
    const int in_h         = input->ne[1];
    const int out_w        = dst->ne[0];
    const int out_h        = dst->ne[1];
    const int c_in         = input->ne[2];
    const int c_out        = kernel->ne[2];
    const int kernel_w     = kernel->ne[0];
    const int kernel_h     = kernel->ne[1];
    const int stride       = dst->op_params[0];
    const int batches      = input->ne[3];

    GGML_ASSERT(c_in == (int)kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    const int total = out_w * out_h * c_out * batches;
    const int block_size = SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    const int num_blocks = (total + block_size - 1) / block_size;

    stream.parallel_for(
        sycl::nd_range<1>(num_blocks * block_size, block_size),
        [=](sycl::nd_item<1> item) {
            const int global_idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

            if (global_idx >= total) {
                return;
            }

            const int out_x_idx = global_idx % out_w;
            const int out_y_idx = (global_idx / out_w) % out_h;
            const int c_idx     = (global_idx / (out_w * out_h)) % c_out;
            const int n_idx     = global_idx / (out_w * out_h * c_out);

            float accumulator = 0.0f;

            for (int c_in_idx = 0; c_in_idx < c_in; c_in_idx++) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    int iy = out_y_idx - kh;
                    if (iy < 0 || iy % stride) continue;
                    iy /= stride;
                    if (iy >= in_h) continue;

                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int ix = out_x_idx - kw;
                        if (ix < 0 || ix % stride) continue;
                        ix /= stride;
                        if (ix >= in_w) continue;

                        const int input_idx = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * iy + ix;
                        const int kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx + kernel_w * kh + kw;

                        const float input_val = input_data[input_idx];
                        const float kern_val  = static_cast<float>(kernel_data[kernel_idx]);

                        accumulator += input_val * kern_val;
                    }
                }
            }

            output_data[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y_idx + out_x_idx] = accumulator;
        }
    );
}
