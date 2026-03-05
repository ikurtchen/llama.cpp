#include "ssm_scan.hpp"
#include "presets.hpp"

// Mamba-1 kernel: selective scan with shared memory for B and C
// Template params: splitD = threads per block, N = d_state
template <int splitD, int N>
static void ssm_scan_f32_sycl(
        const float * src0, const float * src1, const float * src2, const float * src3,
        const float * src4, const float * src5, const int32_t * src6, float * dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t d_inner, const int64_t n_tok,
        const int64_t n_seq, const int64_t n_head,
        sycl::queue & stream) {

    const int num_head_chunks = (n_head + splitD - 1) / splitD;

    const sycl::range<2> global(n_seq, num_head_chunks * splitD);
    const sycl::range<2> local(1, splitD);

    stream.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> smemB(sycl::range<1>(N), cgh);
        sycl::local_accessor<float, 1> smemC(sycl::range<1>(N), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) {
                const int seq_idx    = item.get_group(0);  // blockIdx.x
                const int head_chunk = item.get_group(1);  // blockIdx.y
                const int tid        = item.get_local_id(1); // threadIdx.x

                const float * s0_block = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_chunk * splitD * src0_nb2);
                const float * x_block  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + head_chunk * splitD * sizeof(float));
                const float * dt_block = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_chunk * splitD * sizeof(float));
                const float * A_block  = (const float *) ((const char *) src3 + head_chunk * splitD * src3_nb1);
                const float * B_block  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3));
                const float * C_block  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3));
                float *       y_block  = (float *) ((char *) dst + (seq_idx * d_inner * n_tok * sizeof(float)) + head_chunk * splitD * sizeof(float));
                float *       s_block  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_chunk * splitD * src0_nb2);

                const int stride_x  = src1_nb2 / sizeof(float);
                const int stride_dt = src2_nb1 / sizeof(float);
                const int stride_B  = src4_nb2 / sizeof(float);
                const int stride_C  = src5_nb2 / sizeof(float);
                const int stride_y  = d_inner;

                const int stride_s0 = src0_nb2 / sizeof(float);
                const int stride_A  = src3_nb1 / sizeof(float);

                // Private register arrays
                float regA[N];
                float regs0[N];

                // Load A and s0 (non-CUB path)
                for (int n = 0; n < N; ++n) {
                    regA[n]  = A_block[tid * stride_A + n];
                    regs0[n] = s0_block[tid * stride_s0 + n];
                }

                // Sequential recurrence over time steps
                for (int64_t i = 0; i < n_tok; i++) {
                    if (tid < N) {
                        smemB[tid] = B_block[i * stride_B + tid];
                        smemC[tid] = C_block[i * stride_C + tid];
                    }
                    sycl::group_barrier(item.get_group());

                    float dt_soft_plus = dt_block[i * stride_dt + tid];
                    if (dt_soft_plus <= 20.0f) {
                        dt_soft_plus = sycl::log1p(sycl::exp(dt_soft_plus));
                    }
                    float x_dt = x_block[i * stride_x + tid] * dt_soft_plus;

                    float sumf = 0.0f;
                    for (int n = 0; n < N; n++) {
                        float state = regs0[n] * sycl::exp(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
                        sumf += state * smemC[n];
                        regs0[n] = state;
                    }
                    y_block[i * stride_y + tid] = sumf;
                }

                // Store final states
                const int stride_s = stride_s0;
                for (int n = 0; n < N; ++n) {
                    s_block[tid * stride_s + n] = regs0[n];
                }
            }
        );
    });
}

// Mamba-2 (grouped) kernel: uses warp-level reduction for output accumulation
// Template params: c_factor = states per thread, d_state = total state dimension
template <int c_factor, int d_state>
static void ssm_scan_f32_group_sycl(
        const float * src0, const float * src1, const float * src2, const float * src3,
        const float * src4, const float * src5, const int32_t * src6, float * dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t n_head, const int64_t d_head,
        const int64_t n_group, const int64_t n_tok,
        const int64_t n_seq, const int64_t n_blocks,
        sycl::queue & stream) {

    constexpr int threads = d_state;

    const sycl::range<2> global(n_seq, n_blocks * threads);
    const sycl::range<2> local(1, threads);

    stream.parallel_for(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const int tid      = item.get_local_id(1); // threadIdx.x
            const int block_id = item.get_group(1);    // blockIdx.x
            const int seq_idx  = item.get_group(0);    // blockIdx.y

            const int warp     = tid / WARP_SIZE;
            const int lane     = tid % WARP_SIZE;
            const int warp_idx = block_id * c_factor + warp;

            const int head_idx  = warp_idx / d_head;
            const int head_off  = (warp_idx % d_head) * sizeof(float);

            const int group_off = (head_idx / (n_head / n_group)) * d_state * sizeof(float);

            const float * s0_warp = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
            const float * x_warp  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + (warp_idx * sizeof(float)));
            const float * dt_warp = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
            const float * A_warp  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
            const float * B_warp  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
            const float * C_warp  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
            float *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
            float *       s_warp  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

            // Strides across n_seq_tokens
            const int stride_x  = src1_nb2 / sizeof(float);
            const int stride_dt = src2_nb1 / sizeof(float);
            const int stride_B  = src4_nb2 / sizeof(float);
            const int stride_C  = src5_nb2 / sizeof(float);
            const int stride_y  = n_head * d_head;

            float state[c_factor];
            float state_sum = 0.0f;

            // Load initial states
            for (int j = 0; j < c_factor; j++) {
                state[j] = s0_warp[WARP_SIZE * j + lane];
            }

            for (int64_t i = 0; i < n_tok; i++) {
                const float dt_val = dt_warp[i * stride_dt];
                const float dt_soft_plus = (dt_val <= 20.0f ? sycl::log1p(sycl::exp(dt_val)) : dt_val);

                state_sum = 0.0f;
                const float dA   = sycl::exp(dt_soft_plus * A_warp[0]);
                const float x_dt = x_warp[i * stride_x] * dt_soft_plus;

                for (int j = 0; j < c_factor; j++) {
                    const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
                    const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
                    state[j] = (state[j] * dA) + (B_val * x_dt);
                    state_sum += state[j] * C_val;
                }

                // Parallel accumulation for output via warp reduction
                state_sum = warp_reduce_sum(state_sum);

                if (lane == 0) {
                    y_warp[i * stride_y] = state_sum;
                }
            }

            // Write back the state
            for (int j = 0; j < c_factor; j++) {
                s_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    );
}

void ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // s
    const ggml_tensor * src1 = dst->src[1]; // x
    const ggml_tensor * src2 = dst->src[2]; // dt
    const ggml_tensor * src3 = dst->src[3]; // A
    const ggml_tensor * src4 = dst->src[4]; // B
    const ggml_tensor * src5 = dst->src[5]; // C
    const ggml_tensor * src6 = dst->src[6]; // ids

    const int64_t nc  = src0->ne[0]; // d_state
    const int64_t nr  = src0->ne[1]; // head_dim or 1
    const int64_t nh  = src1->ne[1]; // n_head
    const int64_t ng  = src4->ne[1]; // n_group
    const int64_t n_t = src1->ne[2]; // number of tokens per sequence
    const int64_t n_s = src1->ne[3]; // number of sequences in the batch

    const int64_t s_off = ggml_nelements(src1) * sizeof(float);

    GGML_ASSERT(ggml_nelements(src1) + nc*nr*nh*n_s == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src6->nb[0] == sizeof(int32_t));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(src3->type == GGML_TYPE_F32);
    GGML_ASSERT(src4->type == GGML_TYPE_F32);
    GGML_ASSERT(src5->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const float *   src0_d = (const float *)   src0->data;
    const float *   src1_d = (const float *)   src1->data;
    const float *   src2_d = (const float *)   src2->data;
    const float *   src3_d = (const float *)   src3->data;
    const float *   src4_d = (const float *)   src4->data;
    const float *   src5_d = (const float *)   src5->data;
    const int32_t * src6_d = (const int32_t *) src6->data;
    float *         dst_d  = (float *)         dst->data;

    sycl::queue & stream = *ctx.stream();

    // Extract strides as ints (matching CUDA)
    const int src0_nb2 = src0->nb[2];
    const int src0_nb3 = src0->nb[3];
    const int src1_nb2 = src1->nb[2];
    const int src1_nb3 = src1->nb[3];
    const int src2_nb1 = src2->nb[1];
    const int src2_nb2 = src2->nb[2];
    const int src3_nb1 = src3->nb[1];
    const int src4_nb2 = src4->nb[2];
    const int src4_nb3 = src4->nb[3];
    const int src5_nb2 = src5->nb[2];
    const int src5_nb3 = src5->nb[3];

    if (src3_nb1 == sizeof(float)) {
        // Mamba-2 (grouped)
        const int64_t d_head = nr;

        if (nc == 128) {
            constexpr int threads   = 128;
            constexpr int num_warps = threads / WARP_SIZE;
            constexpr int c_factor  = 128 / WARP_SIZE;

            const int64_t n_blocks = (nh * d_head + (num_warps - 1)) / num_warps;

            ssm_scan_f32_group_sycl<c_factor, 128>(
                src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                src4_nb2, src4_nb3, src5_nb2, src5_nb3,
                s_off, nh, d_head, ng, n_t, n_s, n_blocks, stream);
        } else if (nc == 256) {
            constexpr int threads   = 256;
            constexpr int num_warps = threads / WARP_SIZE;
            constexpr int c_factor  = 256 / WARP_SIZE;

            const int64_t n_blocks = (nh * d_head + (num_warps - 1)) / num_warps;

            ssm_scan_f32_group_sycl<c_factor, 256>(
                src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                src4_nb2, src4_nb3, src5_nb2, src5_nb3,
                s_off, nh, d_head, ng, n_t, n_s, n_blocks, stream);
        } else {
            GGML_ABORT("SYCL SSM_SCAN: doesn't support d_state!=(128 or 256).");
        }
    } else {
        // Mamba-1
        constexpr int splitD = 128;
        constexpr int N      = 16;
        GGML_ASSERT(nh % splitD == 0);
        GGML_ASSERT(nr == 1);
        GGML_ASSERT(ng == 1);
        GGML_ASSERT(nc == N);

        ssm_scan_f32_sycl<splitD, N>(
            src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
            src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
            src4_nb2, src4_nb3, src5_nb2, src5_nb3,
            s_off, nh, n_t, n_s, nh, stream);
    }
}
