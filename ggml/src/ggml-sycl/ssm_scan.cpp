#include "ssm_scan.hpp"

#include "common.hpp"

// Mamba-1 selective scan kernel (non-CUB path, L_template=0 for runtime L)
// Grid mapping: nd_range<3> with range<3>(n_seq, n_groups, splitD), local={1, 1, splitD}
//   blockIdx.x = item.get_group(0) => seq index
//   blockIdx.y = item.get_group(1) => head group index
//   threadIdx.x = item.get_local_id(2) => thread within block
template <int splitD, int N>
static void ssm_scan_f32(
        const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ src2,
        const float * __restrict__ src3, const float * __restrict__ src4, const float * __restrict__ src5,
        const int32_t * __restrict__ src6, float * __restrict__ dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t d_inner, const int64_t L,
        const sycl::nd_item<3> & item,
        float * smemB, float * smemC) {

    const int tidx     = item.get_local_id(2);
    const int block_x  = item.get_group(0);  // seq index
    const int block_y  = item.get_group(1);  // head group index

    const float * s0_block = (const float *)((const char *)src0 + src6[block_x] * src0_nb3 + block_y * splitD * src0_nb2);
    const float * x_block  = (const float *)((const char *)src1 + (block_x * src1_nb3) + block_y * splitD * sizeof(float));
    const float * dt_block = (const float *)((const char *)src2 + (block_x * src2_nb2) + block_y * splitD * sizeof(float));
    const float * A_block  = (const float *)((const char *)src3 + block_y * splitD * src3_nb1);
    const float * B_block  = (const float *)((const char *)src4 + (block_x * src4_nb3));
    const float * C_block  = (const float *)((const char *)src5 + (block_x * src5_nb3));
    float *       y_block  = (float *)((char *)dst + (block_x * d_inner * L * sizeof(float)) + block_y * splitD * sizeof(float));
    float *       s_block  = (float *)((char *)dst + s_off + block_x * src0_nb3 + block_y * splitD * src0_nb2);

    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_y  = d_inner;

    // Non-CUB path: load A and s0 directly
    const int stride_s0 = src0_nb2 / sizeof(float);
    const int stride_A  = src3_nb1 / sizeof(float);

    float regA[N];
    float regs0[N];

    for (int n = 0; n < N; ++n) {
        regA[n]  = A_block[tidx * stride_A + n];
        regs0[n] = s0_block[tidx * stride_s0 + n];
    }

    for (int64_t i = 0; i < L; i++) {
        if (tidx < N) {
            smemB[tidx] = B_block[i * stride_B + tidx];
            smemC[tidx] = C_block[i * stride_C + tidx];
        }
        sycl::group_barrier(item.get_group());

        float dt_soft_plus = dt_block[i * stride_dt + tidx];
        if (dt_soft_plus <= 20.0f) {
            dt_soft_plus = sycl::log1p(sycl::exp(dt_soft_plus));
        }
        float x_dt = x_block[i * stride_x + tidx] * dt_soft_plus;

        float sumf = 0.0f;
        for (int n = 0; n < N; n++) {
            float state = regs0[n] * sycl::exp(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
            sumf += state * smemC[n];
            regs0[n] = state;
        }
        y_block[i * stride_y + tidx] = sumf;
    }

    // Non-CUB path: store state directly
    const int stride_s = stride_s0;
    for (int n = 0; n < N; ++n) {
        s_block[tidx * stride_s + n] = regs0[n];
    }
}

// Mamba-2 selective scan kernel (warp-based with warp_reduce_sum)
// Grid mapping: nd_range<2> with global={(n_head*d_head+num_warps-1)/num_warps * d_state, n_seq * d_state}
//   but more simply: nd_range<2> with global={num_blocks * d_state, n_seq * d_state}
//   Actually use nd_range<2>: dim0 = block_x (covers n_head*d_head), dim1 = block_y (covers n_seq)
//   local = {d_state, 1}
template <int c_factor, int d_state>
static void ssm_scan_f32_group(
        const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ src2,
        const float * __restrict__ src3, const float * __restrict__ src4, const float * __restrict__ src5,
        const int32_t * __restrict__ src6, float * __restrict__ dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t n_head, const int64_t d_head,
        const int64_t n_group, const int64_t n_tok,
        const sycl::nd_item<2> & item) {

    const int tidx     = item.get_local_id(0);
    const int warp     = tidx / WARP_SIZE;
    const int lane     = tidx % WARP_SIZE;
    const int warp_idx = item.get_group(0) * c_factor + warp;

    // Guard against out-of-bounds warps from grid rounding
    if (warp_idx >= n_head * d_head) {
        return;
    }

    const int head_idx =  warp_idx / d_head;
    const int head_off = (warp_idx % d_head) * sizeof(float);
    const int seq_idx  = item.get_group(1);

    const int group_off = (head_idx / (n_head / n_group)) * d_state * sizeof(float);

    const float * s0_warp = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    const float * x_warp  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + (warp_idx * sizeof(float)));
    const float * dt_warp = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
    const float * A_warp  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
    const float * B_warp  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
    const float * C_warp  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
    float *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    float *       s_warp  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    // strides across n_seq_tokens
    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_y  = n_head * d_head;

    float state[c_factor];
    float state_sum = 0.0f;

    for (int j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        const float dt_soft_plus = (dt_warp[i * stride_dt] <= 20.0f ? sycl::log1p(sycl::exp(dt_warp[i * stride_dt])) : dt_warp[i * stride_dt]);

        state_sum = 0.0f;
        const float dA   = sycl::exp(dt_soft_plus * A_warp[0]);
        const float x_dt = x_warp[i * stride_x] * dt_soft_plus;
        for (int j = 0; j < c_factor; j++) {
            const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        // parallel accumulation for output (template version, no nd_item param needed)
        state_sum = warp_reduce_sum<WARP_SIZE>(state_sum);

        if (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }
    }

    // write back the state
    for (int j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

static void ssm_scan_f32_sycl(
        const float * src0, const float * src1, const float * src2, const float * src3,
        const float * src4, const float * src5, const int32_t * src6, float * dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3, const int src2_nb1,
        const int src2_nb2, const int src3_nb1, const int src4_nb2, const int src4_nb3, const int src5_nb2,
        const int src5_nb3, const int64_t s_off, const int64_t d_state, const int64_t head_dim,
        const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq,
        dpct::queue_ptr stream) {

    if (src3_nb1 == sizeof(float)) {
        // Mamba-2
        if (d_state == 128) {
            constexpr int threads   = 128;
            constexpr int num_warps = threads / WARP_SIZE;
            constexpr int c_factor  = 128 / WARP_SIZE;

            const int num_blocks_x = (n_head * head_dim + num_warps - 1) / num_warps;
            const sycl::range<2> global_range(num_blocks_x * threads, n_seq);
            const sycl::range<2> local_range(threads, 1);

            stream->submit([&](sycl::handler & cgh) {
                cgh.parallel_for(
                    sycl::nd_range<2>(global_range, local_range),
                    [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        ssm_scan_f32_group<c_factor, 128>(
                            src0, src1, src2, src3, src4, src5, src6, dst,
                            src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                            src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok,
                            item);
                    });
            });
        } else if (d_state == 256) {
            constexpr int threads   = 256;
            constexpr int num_warps = threads / WARP_SIZE;
            constexpr int c_factor  = 256 / WARP_SIZE;

            const int num_blocks_x = (n_head * head_dim + num_warps - 1) / num_warps;
            const sycl::range<2> global_range(num_blocks_x * threads, n_seq);
            const sycl::range<2> local_range(threads, 1);

            stream->submit([&](sycl::handler & cgh) {
                cgh.parallel_for(
                    sycl::nd_range<2>(global_range, local_range),
                    [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        ssm_scan_f32_group<c_factor, 256>(
                            src0, src1, src2, src3, src4, src5, src6, dst,
                            src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                            src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok,
                            item);
                    });
            });
        } else {
            GGML_ABORT("doesn't support d_state!=(128 or 256).");
        }
    } else {
        // Mamba-1
        constexpr int threads = 128;
        GGML_ASSERT(n_head % threads == 0);
        GGML_ASSERT(head_dim == 1);
        GGML_ASSERT(n_group == 1);

        const int n_groups_y = (n_head + threads - 1) / threads;

        if (d_state == 16) {
            // Use nd_range<3>: dim0=n_seq, dim1=n_groups_y, dim2=threads
            const sycl::range<3> global_range(n_seq, n_groups_y, threads);
            const sycl::range<3> local_range(1, 1, threads);

            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> smemB_acc(sycl::range<1>(16), cgh);
                sycl::local_accessor<float, 1> smemC_acc(sycl::range<1>(16), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(global_range, local_range),
                    [=](sycl::nd_item<3> item) {
                        ssm_scan_f32<threads, 16>(
                            src0, src1, src2, src3, src4, src5, src6, dst,
                            src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                            src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok,
                            item,
                            smemB_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                            smemC_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
        } else {
            GGML_ABORT("doesn't support d_state!=16.");
        }
    }
}

void ggml_sycl_op_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // s
    const struct ggml_tensor * src1 = dst->src[1];  // x
    const struct ggml_tensor * src2 = dst->src[2];  // dt
    const struct ggml_tensor * src3 = dst->src[3];  // A
    const struct ggml_tensor * src4 = dst->src[4];  // B
    const struct ggml_tensor * src5 = dst->src[5];  // C
    const struct ggml_tensor * src6 = dst->src[6];  // ids

    const int64_t nc  = src0->ne[0];  // d_state
    const int64_t nr  = src0->ne[1];  // head_dim or 1
    const int64_t nh  = src1->ne[1];  // n_head
    const int64_t ng  = src4->ne[1];  // n_group
    const int64_t n_t = src1->ne[2];  // number of tokens per sequence
    const int64_t n_s = src1->ne[3];  // number of sequences in the batch

    const int64_t s_off = ggml_nelements(src1) * sizeof(float);

    GGML_ASSERT(ggml_nelements(src1) + nc*nr*nh*n_s == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src6->nb[0] == sizeof(int32_t));

    const float *   src0_d = (const float *)   src0->data;
    const float *   src1_d = (const float *)   src1->data;
    const float *   src2_d = (const float *)   src2->data;
    const float *   src3_d = (const float *)   src3->data;
    const float *   src4_d = (const float *)   src4->data;
    const float *   src5_d = (const float *)   src5->data;
    const int32_t * src6_d = (const int32_t *) src6->data;
    float *         dst_d  = (float *)         dst->data;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    ssm_scan_f32_sycl(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                       src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
                       src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
                       s_off, nc, nr, nh, ng, n_t, n_s, stream);
}
