#include "norm.hpp"

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
