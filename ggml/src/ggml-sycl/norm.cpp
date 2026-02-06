#include "norm.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"

#define SYCL_RMS_NORM_BLOCK_SIZE 256

// Block reduction for sum using shared memory
template <int block_size>
static inline float block_reduce_sum(float val, float * shared_mem, const sycl::nd_item<3> & item_ct1) {
    const int tid = item_ct1.get_local_id(2);
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // First, reduce within warp
    val = warp_reduce_sum(val, item_ct1);

    // Write reduced warp value to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Final reduction in first warp
    if (warp_id == 0) {
        val = (tid < (block_size / WARP_SIZE)) ? shared_mem[lane_id] : 0.0f;
        val = warp_reduce_sum(val, item_ct1);
    }

    return val;
}

template <int block_size>
static void rms_norm_f32_kernel(
        const float * x,
        float * dst,
        const int ncols,
        const int64_t stride_row,
        const int64_t stride_channel,
        const int64_t stride_sample,
        const float eps,
        const sycl::nd_item<3> & item_ct1,
        float * shared_sum) {

    const int nrows     = item_ct1.get_group_range(2);
    const int nchannels = item_ct1.get_group_range(1);

    const int row       = item_ct1.get_group(2);
    const int channel   = item_ct1.get_group(1);
    const int sample    = item_ct1.get_group(0);
    const int tid       = item_ct1.get_local_id(2);

    // Compute input offset
    const float * x_row = x + sample * stride_sample + channel * stride_channel + row * stride_row;
    // Compute output offset (contiguous output)
    float * dst_row = dst + ((sample * nchannels + channel) * nrows + row) * ncols;

    // Compute partial sum of squares
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x_row[col];
        tmp += xi * xi;
    }

    // Block reduction to get total sum of squares
    tmp = block_reduce_sum<block_size>(tmp, shared_sum, item_ct1);

    // Compute scale factor
    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    // Apply normalization
    for (int col = tid; col < ncols; col += block_size) {
        dst_row[col] = scale * x_row[col];
    }
}

static void rms_norm_f32_sycl(
        const float * x,
        float * dst,
        const int ncols,
        const int nrows,
        const int nchannels,
        const int nsamples,
        const int64_t stride_row,
        const int64_t stride_channel,
        const int64_t stride_sample,
        const float eps,
        dpct::queue_ptr stream) {

    const sycl::range<3> blocks_num(nsamples, nchannels, nrows);

    if (ncols < 1024) {
        constexpr int block_size = SYCL_RMS_NORM_BLOCK_SIZE;
        const sycl::range<3> block_dims(1, 1, block_size);
        const size_t shared_mem_size = (block_size / WARP_SIZE) * sizeof(float);

        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> shared_sum_acc(sycl::range<1>(block_size / WARP_SIZE), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(blocks_num * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rms_norm_f32_kernel<block_size>(
                        x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                        item_ct1, shared_sum_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    } else {
        constexpr int block_size = 1024;
        const sycl::range<3> block_dims(1, 1, block_size);

        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> shared_sum_acc(sycl::range<1>(block_size / WARP_SIZE), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(blocks_num * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rms_norm_f32_kernel<block_size>(
                        x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                        item_ct1, shared_sum_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    // TODO: Implement layer norm
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    // TODO: Implement group norm
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    rms_norm_f32_sycl(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    // TODO: Implement rms_norm backward pass
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    // TODO: Implement L2 norm
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
}
