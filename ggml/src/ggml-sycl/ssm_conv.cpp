#include "ssm_conv.hpp"
#include "presets.hpp"

// Maximum supported convolution kernel size.
// Used for fixed-size private arrays inside SYCL kernels.
#define SSM_CONV_MAX_D_CONV 9

// Number of threads per work-group (must evenly divide d_inner)
#define SSM_CONV_BLOCK_SIZE 128

// Time-step chunk size for the long-token path
#define SSM_CONV_SPLIT_N_T 32

// Short-token kernel: n_t <= SSM_CONV_SPLIT_N_T
// Each work-item processes one element of d_inner across all time steps,
// performing a 1D convolution with a circular buffer.
static void ssm_conv_f32_sycl(
        const float * src0, const float * src1,
        const int src0_nb0, const int src0_nb1, const int src0_nb2,
        const int src1_nb1,
        float * dst,
        const int dst_nb0, const int dst_nb1, const int dst_nb2,
        const int64_t nc, const int64_t nr, const int64_t n_t, const int64_t n_s,
        sycl::queue & stream) {

    const int threads = SSM_CONV_BLOCK_SIZE;
    GGML_ASSERT(nr % threads == 0);

    // Grid: (n_s, nr / threads)
    const sycl::range<2> global(n_s, nr);
    const sycl::range<2> local(1, threads);

    stream.parallel_for(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> item) {
            const int tid  = item.get_local_id(1);
            const int bidx = item.get_group(0);  // sequence index
            const int bidy = item.get_group(1);  // d_inner block index

            const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * threads * src0_nb1);
            const float * w_block = (const float *) ((const char *) src1 + bidy * threads * src1_nb1);
            float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * threads * dst_nb0);

            const int stride_x = src0_nb1 / (int) sizeof(float);
            const int stride_w = src1_nb1 / (int) sizeof(float);
            const int stride_y = dst_nb1  / (int) sizeof(float);

            // Private circular buffer and weights (fixed max size)
            float x[SSM_CONV_MAX_D_CONV] = {};
            float w[SSM_CONV_MAX_D_CONV] = {};

            // Load convolution weights
            for (int64_t j = 0; j < nc; j++) {
                w[j] = w_block[tid * stride_w + j];
            }

            // Process each time step
            for (int64_t i = 0; i < n_t; i++) {
                float sumf = 0.0f;

                if (i == 0) {
                    for (int64_t j = 0; j < nc; j++) {
                        x[j] = x_block[tid * stride_x + j];
                    }
                } else {
                    x[(i - 1) % nc] = x_block[tid * stride_x + i + nc - 1];
                }

                for (int64_t j = 0; j < nc; j++) {
                    sumf += x[(i + j) % nc] * w[j];
                }
                y_block[i * stride_y + tid] = sumf;
            }
        }
    );
}

// Long-token kernel: n_t > SSM_CONV_SPLIT_N_T
// Splits the time dimension into chunks to increase parallelism.
// Each work-item processes one element of d_inner across a chunk of time steps.
static void ssm_conv_long_token_f32_sycl(
        const float * src0, const float * src1,
        const int src0_nb0, const int src0_nb1, const int src0_nb2,
        const int src1_nb1,
        float * dst,
        const int dst_nb0, const int dst_nb1, const int dst_nb2,
        const int64_t nc, const int64_t nr, const int64_t n_t, const int64_t n_s,
        sycl::queue & stream) {

    const int threads = SSM_CONV_BLOCK_SIZE;
    GGML_ASSERT(nr % threads == 0);

    const int64_t split_n_t   = SSM_CONV_SPLIT_N_T;
    const int64_t n_t_blocks  = (n_t + split_n_t - 1) / split_n_t;

    // Grid: (n_s, nr / threads, n_t_blocks)
    const sycl::range<3> global(n_s, nr, n_t_blocks);
    const sycl::range<3> local(1, threads, 1);

    stream.parallel_for(
        sycl::nd_range<3>(global, local),
        [=](sycl::nd_item<3> item) {
            const int tid  = item.get_local_id(1);
            const int bidx = item.get_group(0);  // sequence index
            const int bidy = item.get_group(1);  // d_inner block index
            const int bidz = item.get_group(2);  // time chunk index

            const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2
                                        + bidy * threads * src0_nb1
                                        + bidz * split_n_t * src0_nb0);
            const float * w_block = (const float *) ((const char *) src1 + bidy * threads * src1_nb1);
            float *       y_block = (float *) ((char *) dst + bidx * dst_nb2
                                        + bidz * split_n_t * dst_nb1
                                        + bidy * threads * dst_nb0);

            const int stride_x = src0_nb1 / (int) sizeof(float);
            const int stride_w = src1_nb1 / (int) sizeof(float);
            const int stride_y = dst_nb1  / (int) sizeof(float);

            float x[SSM_CONV_MAX_D_CONV] = {};
            float w[SSM_CONV_MAX_D_CONV] = {};

            for (int64_t j = 0; j < nc; j++) {
                w[j] = w_block[tid * stride_w + j];
            }

            for (int64_t i = 0; i < split_n_t; i++) {
                if (bidz * split_n_t + i < n_t) {
                    float sumf = 0.0f;

                    if (i == 0) {
                        for (int64_t j = 0; j < nc; j++) {
                            x[j] = x_block[tid * stride_x + j];
                        }
                    } else {
                        x[(i - 1) % nc] = x_block[tid * stride_x + i + nc - 1];
                    }

                    for (int64_t j = 0; j < nc; j++) {
                        sumf += x[(i + j) % nc] * w[j];
                    }
                    y_block[i * stride_y + tid] = sumf;
                }
            }
        }
    );
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // conv_x
    const ggml_tensor * src1 = dst->src[1];  // conv1d.weight

    const int64_t nc  = src1->ne[0];         // d_conv
    const int64_t nr  = src0->ne[1];         // d_inner
    const int64_t n_t = dst->ne[1];          // tokens per sequence
    const int64_t n_s = dst->ne[2];          // number of sequences in the batch

    GGML_ASSERT(dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(nc <= SSM_CONV_MAX_D_CONV);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;

    sycl::queue & stream = *ctx.stream();

    if (n_t <= SSM_CONV_SPLIT_N_T) {
        ssm_conv_f32_sycl(src0_d, src1_d,
                           src0->nb[0], src0->nb[1], src0->nb[2],
                           src1->nb[1],
                           dst_d,
                           dst->nb[0], dst->nb[1], dst->nb[2],
                           nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_long_token_f32_sycl(src0_d, src1_d,
                                      src0->nb[0], src0->nb[1], src0->nb[2],
                                      src1->nb[1],
                                      dst_d,
                                      dst->nb[0], dst->nb[1], dst->nb[2],
                                      nc, nr, n_t, n_s, stream);
    }
}
