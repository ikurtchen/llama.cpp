#include "norm.hpp"

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
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float *       dst_d  = (float *) dst->data;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int num_groups  = dst->op_params[0];
    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const int group_size   = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    const int ne_elements  = ggml_nelements(src0);
    const int total_groups = num_groups * src0->ne[3];

    queue_ptr stream = ctx.stream();

    const int block_size = 256;
    const int num_warps  = block_size / WARP_SIZE;

    sycl::range<1> global_range(total_groups * block_size);
    sycl::range<1> local_range(block_size);

    stream->submit([&](sycl::handler & h) {
        // float2 shared memory for the two-component reduction (sum, sum_sq)
        sycl::local_accessor<sycl::float2, 1> shared_vals(num_warps, h);

        h.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int gid   = item.get_group(0);   // group index
                const int tid   = item.get_local_id(0);

                const int start = gid * group_size + tid;
                const int end   = sycl::min(gid * group_size + group_size, ne_elements);

                // --- Pass 1: accumulate sum and sum-of-squares in float2 ---
                // Uses E[x] and E[x²] to compute variance as Var = E[x²] - E[x]²
                // This eliminates the need for a separate mean-subtraction pass.
                sycl::float2 acc(0.0f, 0.0f);
                for (int j = start; j < end; j += block_size) {
                    const float xi = src0_d[j];
                    acc.x() += xi;
                    acc.y() += xi * xi;
                }

                sycl::float2 * sh = shared_vals.get_multi_ptr<sycl::access::decorated::no>().get_raw();
                acc = block_reduce<block_reduce_method::SUM>(acc, sh, item);

                // Broadcast from sub-group 0 to all work-items
                if (tid == 0) {
                    sh[0] = acc;
                }
                item.barrier(sycl::access::fence_space::local_space);
                acc = sh[0];

                const float mean     = acc.x() / group_size;
                const float variance = acc.y() / group_size - mean * mean;
                const float scale    = sycl::rsqrt(variance + eps);

                // --- Pass 2: normalize and write output ---
                for (int j = start; j < end; j += block_size) {
                    dst_d[j] = (src0_d[j] - mean) * scale;
                }
            }
        );
    });
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

                // Compute L2 normalization scale
                // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
                const float scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

                // Pass 2: write normalized output
                for (int col = tid; col < ncols; col += block_size) {
                    dst_row[col] = scale * x_row[col];
                }
            }
        );
    });
}
