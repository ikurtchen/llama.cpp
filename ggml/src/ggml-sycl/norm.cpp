#include "norm.hpp"

// ---------------------------------------------------------------------------
// GROUP_NORM baseline kernel (3-pass: sum, mean-subtract + variance, normalize)
// ---------------------------------------------------------------------------
static void group_norm_f32(const float* x, float* dst, const int group_size, const int ne_elements, const float eps,
    const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {
    int start = item_ct1.get_group(2) * group_size;
    int end = start + group_size;
    const int nthreads = item_ct1.get_local_range(2);
    const int nwarps = nthreads / WARP_SIZE;
    start += item_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_SIZE;

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        item_ct1.barrier();
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        item_ct1.barrier();
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float variance = tmp / group_size;
    float scale = sycl::rsqrt(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// ---------------------------------------------------------------------------
// GROUP_NORM baseline launch function (adaptive block sizing)
// ---------------------------------------------------------------------------
static void group_norm_f32_sycl(const float* x, float* dst,
    const int num_groups, const float eps, const int group_size,
    const int ne_elements, queue_ptr stream, int device) {
    if (group_size < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            const float eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(
                        x, dst, group_size, ne_elements, eps_ct4, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);

            const float eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(x, dst, group_size, ne_elements,
                        eps_ct4, item_ct1,
                        get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

// ---------------------------------------------------------------------------
// L2_NORM baseline kernel
// ---------------------------------------------------------------------------
static void l2_norm_f32(const float* x, float* dst, const int ncols, const float eps,
    const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
    const int tid = item_ct1.get_local_id(2);
    const int nthreads = item_ct1.get_local_range(2);
    const int nwarps = nthreads / WARP_SIZE;
    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row * ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        size_t nreduce = nwarps / WARP_SIZE;
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float scale = sycl::rsqrt(sycl::max(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[row * ncols + col] = scale * x[row * ncols + col];
    }
}

// ---------------------------------------------------------------------------
// L2_NORM baseline launch function (adaptive block sizing)
// ---------------------------------------------------------------------------
static void l2_norm_f32_sycl(const float* x, float* dst, const int ncols,
    const int nrows, const float eps,
    queue_ptr stream, int device) {
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    l2_norm_f32(x, dst, ncols, eps, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    l2_norm_f32(x, dst, ncols, eps, item_ct1,
                        get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

// ---------------------------------------------------------------------------
// NORM (LayerNorm without affine)
// ---------------------------------------------------------------------------
void ggml_sycl_op_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float *       dst_d  = (float *) dst->data;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    queue_ptr stream = ctx.stream();

    const int ncols     = ne00;
    const int nrows     = ne01;
    const int nchannels = ne02;
    const int nsamples  = ne03;

    const int block_size = 256;
    const int num_warps  = block_size / WARP_SIZE;
    const int num_groups = nrows * nchannels * nsamples;

    sycl::range<1> global_range(num_groups * block_size);
    sycl::range<1> local_range(block_size);

    stream->submit([&](sycl::handler & h) {
        // sycl::float2 shared memory for the two-component reduction
        sycl::local_accessor<sycl::float2, 1> shared_vals(num_warps, h);

        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int group_id = item.get_group(0);
                const int tid      = item.get_local_id(0);

                const int row     = group_id % nrows;
                const int channel = (group_id / nrows) % nchannels;
                const int sample  = group_id / (nrows * nchannels);

                const float * x_row = src0_d + sample * s03 + channel * s02 + row * s01;
                float * dst_row = dst_d + ((sample * nchannels + channel) * nrows + row) * ncols;

                // Accumulate sum and sum-of-squares in a float2
                sycl::float2 mean_var(0.0f, 0.0f);
                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = x_row[col];
                    mean_var.x() += xi;
                    mean_var.y() += xi * xi;
                }

                // Block-wide float2 reduction
                sycl::float2 * sh = shared_vals.get_multi_ptr<sycl::access::decorated::no>().get_raw();
                mean_var = block_reduce<block_reduce_method::SUM>(mean_var, sh, item);

                // Broadcast from sub-group 0 to all work-items
                if (tid == 0) {
                    sh[0] = mean_var;
                }
                item.barrier(sycl::access::fence_space::local_space);
                mean_var = sh[0];

                const float mean    = mean_var.x() / ncols;
                const float var     = mean_var.y() / ncols - mean * mean;
                const float inv_std = sycl::rsqrt(var + eps);

                for (int col = tid; col < ncols; col += block_size) {
                    dst_row[col] = (x_row[col] - mean) * inv_std;
                }
            }
        );
    });
}

// ---------------------------------------------------------------------------
// GROUP_NORM
// ---------------------------------------------------------------------------
void ggml_sycl_op_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];
    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int group_size = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, num_groups, eps, group_size, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
}

// ---------------------------------------------------------------------------
// RMS_NORM
// ---------------------------------------------------------------------------
void ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float *       dst_d  = (float *) dst->data;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    // Compute element strides from byte strides
    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;  // stride between rows (elements)
    const int64_t s02 = nb02 / ts0;  // stride between channels
    const int64_t s03 = nb03 / ts0;  // stride between samples

    queue_ptr stream = ctx.stream();

    const int ncols     = ne00;
    const int nrows     = ne01;
    const int nchannels = ne02;
    const int nsamples  = ne03;

    // With WARP_SIZE=16 (Intel), the 2-level block_reduce supports at most
    // WARP_SIZE * WARP_SIZE = 256 work-items. Always use 256 to stay within
    // the reduction's capability (CUDA uses WARP_SIZE=32 so 1024 works there).
    const int block_size = 256;
    const int num_warps  = block_size / WARP_SIZE;

    // Total number of work-groups: one per (row, channel, sample) combination
    const int num_groups = nrows * nchannels * nsamples;

    // Use 1D nd_range so block_reduce (which uses get_local_id(0)) works correctly
    sycl::range<1> global_range(num_groups * block_size);
    sycl::range<1> local_range(block_size);

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> shared_vals(num_warps, h);

        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int group_id = item.get_group(0);
                const int tid      = item.get_local_id(0);

                // Decompose flat group_id into (sample, channel, row)
                // Layout: group_id = sample * (nchannels * nrows) + channel * nrows + row
                const int row     = group_id % nrows;
                const int channel = (group_id / nrows) % nchannels;
                const int sample  = group_id / (nrows * nchannels);

                // Source uses non-contiguous strides
                const float * x_row = src0_d + sample * s03 + channel * s02 + row * s01;

                // Destination is always contiguous
                float * dst_row = dst_d + ((sample * nchannels + channel) * nrows + row) * ncols;

                // Pass 1: accumulate sum of squares across columns
                float tmp = 0.0f;
                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = x_row[col];
                    tmp += xi * xi;
                }

                // Block-wide sum reduction through local memory
                float * sh = shared_vals.get_multi_ptr<sycl::access::decorated::no>().get_raw();
                tmp = block_reduce<block_reduce_method::SUM>(tmp, sh, item);

                // block_reduce returns the correct value only in sub-group 0.
                // Broadcast to all work-items via shared memory.
                if (tid == 0) {
                    sh[0] = tmp;
                }
                item.barrier(sycl::access::fence_space::local_space);
                tmp = sh[0];

                // Compute RMS normalization scale
                const float mean  = tmp / ncols;
                const float scale = sycl::rsqrt(mean + eps);

                // Pass 2: write normalized output
                for (int col = tid; col < ncols; col += block_size) {
                    dst_row[col] = scale * x_row[col];
                }
            }
        );
    });
}

// ---------------------------------------------------------------------------
// RMS_NORM_BACK (backward pass for RMS normalization)
// ---------------------------------------------------------------------------
void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients (dy)
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass (x)

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float *       dst_d   = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00  = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    queue_ptr stream = ctx.stream();

    const int ncols = ne00;

    const int block_size = 256;
    const int num_warps  = block_size / WARP_SIZE;

    sycl::range<1> global_range(nrows * block_size);
    sycl::range<1> local_range(block_size);

    stream->submit([&](sycl::handler & h) {
        // float2 shared memory for the two-component reduction (sum_xx, sum_xg)
        sycl::local_accessor<sycl::float2, 1> shared_vals(num_warps, h);

        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item.get_group(0);
                const int tid = item.get_local_id(0);

                const float * grad_row = grad_d  + static_cast<int64_t>(row) * ncols;
                const float * x_row    = src0f_d + static_cast<int64_t>(row) * ncols;
                float *       dst_row  = dst_d   + static_cast<int64_t>(row) * ncols;

                // Pass 1: accumulate sum of squares (sum_xx) and sum of x*grad (sum_xg)
                sycl::float2 sums(0.0f, 0.0f);
                for (int col = tid; col < ncols; col += block_size) {
                    const float xfi = x_row[col];
                    sums.x() += xfi * xfi;
                    sums.y() += xfi * grad_row[col];
                }

                // Block-wide float2 reduction
                sycl::float2 * sh = shared_vals.get_multi_ptr<sycl::access::decorated::no>().get_raw();
                sums = block_reduce<block_reduce_method::SUM>(sums, sh, item);

                // Broadcast from work-item 0 to all work-items via shared memory
                if (tid == 0) {
                    sh[0] = sums;
                }
                item.barrier(sycl::access::fence_space::local_space);
                sums = sh[0];

                const float sum_xx = sums.x();
                const float sum_xg = sums.y();

                const float mean_eps   = sum_xx / ncols + eps;
                const float sum_eps    = sum_xx + ncols * eps;

                const float scale_grad = sycl::rsqrt(mean_eps);
                const float scale_x    = -scale_grad * sum_xg / sum_eps;

                // Pass 2: write gradient output
                for (int col = tid; col < ncols; col += block_size) {
                    dst_row[col] = scale_grad * grad_row[col] + scale_x * x_row[col];
                }
            }
        );
    });
}

// ---------------------------------------------------------------------------
// L2_NORM
// ---------------------------------------------------------------------------
void ggml_sycl_op_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int64_t ne00 = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float * dst_dd = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    l2_norm_f32_sycl(src0_dd, dst_dd, ne00, nrows, eps, main_stream, ctx.device);

}
