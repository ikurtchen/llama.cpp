#include "norm.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"

static __dpct_inline__ uint32_t fastmodulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    return fast_div_modulo(n, fastdiv_values).y();
}

static void norm_f32(const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const sycl::nd_item<3>& item_ct1, int block_size) {

    const int nrows = item_ct1.get_group_range(2);
    const int nchannels = item_ct1.get_group_range(1);

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int tid = item_ct1.get_local_id(2);

    const auto strided_offset = calculate_offset<3>({stride_sample, stride_channel, stride_row}, {sample, channel, row});
    const auto packed_offset = calculate_offset<3>({nchannels * nrows * ncols, nrows * ncols, ncols}, {sample, channel, row});

    x += strided_offset;
    dst += packed_offset;

    float sum_x = 0.f;
    float sum_x2 = 0.f;

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        sum_x += xi;
        sum_x2 += xi * xi;
    }

    const float mean = sycl::reduce_over_group(item_ct1.get_group(), sum_x, sycl::plus<float>()) / ncols;
    const float var = sycl::reduce_over_group(item_ct1.get_group(), sum_x2, sycl::plus<float>()) / ncols - mean * mean;
    const float inv_std = sycl::rsqrt(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

static void group_norm_f32(const float* x, float* dst, const int group_size, const int ne_elements, const float eps,
    const sycl::nd_item<3>& item_ct1, int block_size) {
    int start = item_ct1.get_group(2) * group_size;
    int end = start + group_size;
    start += item_ct1.get_local_id(2);

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    float mean = sycl::reduce_over_group(item_ct1.get_group(), tmp, sycl::plus<float>()) / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    float variance = sycl::reduce_over_group(item_ct1.get_group(), tmp, sycl::plus<float>()) / group_size;
    float scale = sycl::rsqrt(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

static void rms_norm_f32(const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const sycl::nd_item<3>& item_ct1, int block_size) {

    const int nrows = item_ct1.get_group_range(2);
    const int nchannels = item_ct1.get_group_range(1);

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int tid = item_ct1.get_local_id(2);

    const auto strided_offset = calculate_offset<3>({stride_sample, stride_channel, stride_row}, {sample, channel, row});
    const auto packed_offset = calculate_offset<3>({nchannels * nrows * ncols, nrows * ncols, ncols}, {sample, channel, row});

    x   += strided_offset;
    dst += packed_offset;

    float tmp = 0.0f;

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    const float mean = sycl::reduce_over_group(item_ct1.get_group(), tmp, sycl::plus<float>()) / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

template<bool do_multiply = false, bool do_add = false>
static void rms_norm_fused_f32(const float* x, float* dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float* mul, const int64_t mul_stride_row, const int64_t mul_stride_channel,
        const int64_t mul_stride_sample, const sycl::uint3 mul_ncols_packed, const sycl::uint3 mul_nrows_packed,
        const sycl::uint3 mul_nchannels_packed, const sycl::uint3 mul_nsamples_packed, const float* add,
        const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const sycl::uint3 add_ncols_packed, const sycl::uint3 add_nrows_packed, const sycl::uint3 add_nchannels_packed,
        const sycl::uint3 add_nsamples_packed, const sycl::nd_item<3>& item_ct1, int block_size) {

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    const int nrows = item_ct1.get_group_range(2);
    const int nchannels = item_ct1.get_group_range(1);

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int tid = item_ct1.get_local_id(2);

    const auto strided_offset = calculate_offset<3>({stride_sample, stride_channel, stride_row}, {sample, channel, row});
    const auto packed_offset = calculate_offset<3>({nchannels * nrows * ncols, nrows * ncols, ncols}, {sample, channel, row});

    x   += strided_offset;
    dst += packed_offset;

    if constexpr (do_multiply) {
        const uint32_t mul_row = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const uint32_t add_row = fastmodulo(row, add_nrows_packed);
        const uint32_t add_channel = fastmodulo(channel, add_nchannels_packed);
        const uint32_t add_sample = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f;

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    const float mean = sycl::reduce_over_group(item_ct1.get_group(), tmp, sycl::plus<float>()) / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const uint32_t mul_col = fastmodulo(col, mul_ncols_packed);
            const uint32_t add_col = fastmodulo(col, add_ncols_packed);
            dst[col] = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const uint32_t mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col] = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }
    }
}

static void l2_norm_f32(const float* x, float* dst, const int ncols, const float eps,
    const sycl::nd_item<3>& item_ct1, int block_size) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
    const int tid = item_ct1.get_local_id(2);
    float tmp = 0.0f;

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row * ncols + col];
        tmp += xi * xi;
    }

    const float sum = sycl::reduce_over_group(item_ct1.get_group(), tmp, sycl::plus<float>());
    const float scale = sycl::rsqrt(sycl::max(sum, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[row * ncols + col] = scale * x[row * ncols + col];
    }
}

static void norm_f32_sycl(const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const float eps, queue_ptr stream, int device) {

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
    const int block_size = ncols < 1024 ? 256 : work_group_size;
    const sycl::range<3> block_dims(1, 1, block_size);

    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global_dims * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                norm_f32(x, dst, ncols, stride_row, stride_channel, stride_sample, eps, item_ct1, block_size);
            });
        });
}

static void group_norm_f32_sycl(const float* x, float* dst,
    const int num_groups, const float eps, const int group_size,
    const int ne_elements, queue_ptr stream, int device) {
    const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
    const int block_size = group_size < 1024 ? 256 : work_group_size;
    const sycl::range<3> block_dims(1, 1, block_size);

    stream->submit([&](sycl::handler& cgh) {
        const float eps_ct4 = eps;
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                block_dims),
            [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                group_norm_f32(
                    x, dst, group_size, ne_elements, eps_ct4, item_ct1, block_size);
            });
        });
}

static void rms_norm_f32_sycl(const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, queue_ptr stream, int device) {

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
    const int block_size = ncols < 1024 ? 256 : work_group_size;
    const sycl::range<3> block_dims(1, 1, block_size);

    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global_dims * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                rms_norm_f32(x, dst, ncols, stride_row, stride_channel, stride_sample, eps, item_ct1, block_size);
            });
        });
}

static void rms_norm_fused_f32_sycl(const float* x, const float* mul, const float* add, float* dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint32_t mul_ncols, const uint32_t mul_nrows, const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint32_t add_ncols, const uint32_t add_nrows, const uint32_t add_nchannels, const uint32_t add_nsamples,
        const float eps, queue_ptr stream, int device) {

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
    const int block_size = ncols < 1024 ? 256 : work_group_size;
    const sycl::range<3> block_dims(1, 1, block_size);

    if (mul == nullptr) {
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols, stride_row, stride_channel, stride_sample, eps, item_ct1, block_size);
                });
            });
        return;
    }

    const sycl::uint3 mul_ncols_packed = init_fastdiv_values(mul_ncols);
    const sycl::uint3 mul_nrows_packed = init_fastdiv_values(mul_nrows);
    const sycl::uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const sycl::uint3 mul_nsamples_packed = init_fastdiv_values(mul_nsamples);

    if (add == nullptr) {
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_fused_f32<true, false>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                        mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                        nullptr, 0, 0, 0, sycl::uint3(0,0,0), sycl::uint3(0,0,0), sycl::uint3(0,0,0), sycl::uint3(0,0,0),
                        item_ct1, block_size);
                });
            });
    } else {
        const sycl::uint3 add_ncols_packed = init_fastdiv_values(add_ncols);
        const sycl::uint3 add_nrows_packed = init_fastdiv_values(add_nrows);
        const sycl::uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const sycl::uint3 add_nsamples_packed = init_fastdiv_values(add_nsamples);

        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_fused_f32<true, true>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample,
                        mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                        add, add_stride_row, add_stride_channel, add_stride_sample,
                        add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed,
                        item_ct1, block_size);
                });
            });
    }
}

static void l2_norm_f32_sycl(const float* x, float* dst, const int ncols,
    const int nrows, const float eps,
    queue_ptr stream, int device) {
    const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
    const int block_size = ncols < 1024 ? 256 : work_group_size;
    const sycl::range<3> block_dims(1, 1, block_size);

    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims,
                block_dims),
            [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                l2_norm_f32(x, dst, ncols, eps, item_ct1, block_size);
            });
        });
}

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03, s01, s02, s03, eps, main_stream, ctx.device);
}

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int group_size = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, num_groups, eps, group_size, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
}

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS
    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03, s01, s02, s03, eps, main_stream, ctx.device);
}

void ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor) {
    const ggml_tensor* rms_norm_src = dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float* src0_d = (const float*)rms_norm_src->data;
    const float* mul_d = nullptr;
    const ggml_tensor* mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (const float*)mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d = (const float*)mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float* dst_d = (float*)mul_tensor->data;
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const uint32_t mul_ncols = mul_src->ne[0];
    const uint32_t mul_nrows = mul_src->ne[1];
    const uint32_t mul_nchannels = mul_src->ne[2];
    const uint32_t mul_nsamples = mul_src->ne[3];

    rms_norm_fused_f32_sycl(src0_d, mul_d, nullptr, dst_d,
        ne00, ne01, ne02, ne03,
        s01, s02, s03,
        mul_s01, mul_s02, mul_s03,
        mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
        0, 0, 0,
        0, 0, 0, 0,
        eps, main_stream, ctx.device);
}

void ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* add_tensor) {
    const ggml_tensor* rms_norm_src = dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float* src0_d = (const float*)rms_norm_src->data;
    const float* mul_d = nullptr;
    const ggml_tensor* mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (const float*)mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d = (const float*)mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float* add_d = nullptr;
    const ggml_tensor* add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d = (const float*)add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d = (const float*)add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float* dst_d = (float*)add_tensor->data;
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const uint32_t mul_ncols = mul_src->ne[0];
    const uint32_t mul_nrows = mul_src->ne[1];
    const uint32_t mul_nchannels = mul_src->ne[2];
    const uint32_t mul_nsamples = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const uint32_t add_ncols = add_src->ne[0];
    const uint32_t add_nrows = add_src->ne[1];
    const uint32_t add_nchannels = add_src->ne[2];
    const uint32_t add_nsamples = add_src->ne[3];

    rms_norm_fused_f32_sycl(src0_d, mul_d, add_d, dst_d,
        ne00, ne01, ne02, ne03,
        s01, s02, s03,
        mul_s01, mul_s02, mul_s03,
        mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
        add_s01, add_s02, add_s03,
        add_ncols, add_nrows, add_nchannels, add_nsamples,
        eps, main_stream, ctx.device);
}

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    float eps = 1e-5f;
    std::memcpy(&eps, dst->op_params, sizeof(float));
    if (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    const float * g_base  = static_cast<const float *>(dst->src[0]->data); // dz
    const float * x_base  = static_cast<const float *>(dst->src[1]->data); // x
          float * dx_base = static_cast<      float *>(dst->data);

    const int64_t D  = dst->ne[0];
    const int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (void) n3;
    const int64_t N  = ggml_nrows(dst);
    if (D == 0 || N == 0) return;

    const ggml_tensor *G = dst->src[0];
    const ggml_tensor *X = dst->src[1];
    const int ts = (int) ggml_type_size(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    const int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    const int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    const int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::queue_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    const int device_max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    auto roundup = [](int v, int m) { return ((v + m - 1) / m) * m; };
    int wg_cap = 256;
    if (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    int WG = std::max(WARP_SIZE, std::min(roundup((int)std::min<int64_t>(D, wg_cap), WARP_SIZE), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        const int nwarps_loc = std::max(1, WG / WARP_SIZE);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        auto l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        auto l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item_ct1.get_group(2);
                const int tid = item_ct1.get_local_id(2);

                const int64_t i1 = row % n1;
                const int64_t i2 = (row / n1) % n2;
                const int64_t i3 = row / (n1 * n2);

                const float *__restrict x_row = x_base + i3 * xs3 + i2 * xs2 + i1 * xs1;
                const float *__restrict g_row = g_base + i3 * gs3 + i2 * gs2 + i1 * gs1;
                float *__restrict d_row       = dx_base + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                float sum_xx = 0.f, sum_xg = 0.f;
#ifndef GGML_SYCL_RMS_BACK_FAST
                float c_xx = 0.f, c_xg = 0.f;
#endif
                for (int64_t col = tid; col < D; col += WG) {
                    const float xv = x_row[col];
                    const float gv = g_row[col];
#ifdef GGML_SYCL_RMS_BACK_FAST
                    sum_xx += xv * xv;
                    sum_xg += xv * gv;
#else
                    float y1 = xv * xv - c_xx;
                    float t1 = sum_xx + y1;
                    c_xx = (t1 - sum_xx) - y1;
                    sum_xx = t1;

                    float y2 = xv * gv - c_xg;
                    float t2 = sum_xg + y2;
                    c_xg = (t2 - sum_xg) - y2;
                    sum_xg = t2;
#endif
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(sum_xx,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xx
#else
                    0.f
#endif
                );
                sycl::float2 xg = sycl::float2(sum_xg,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xg
#else
                    0.f
#endif
                );
                xx = warp_reduce_sum(xx, item_ct1);
                xg = warp_reduce_sum(xg, item_ct1);

                // cross-warp reduction using local memory (single barrier)
                const auto sub_group = item_ct1.get_sub_group();
                const auto sg_id     = sub_group.get_group_linear_id();
                const auto wi_in_sg  = sub_group.get_local_linear_id();
                const int nthreads   = item_ct1.get_local_range(2);
                const int nwarps     = nthreads / WARP_SIZE;

                sycl::float2 xx_total = xx;
                sycl::float2 xg_total = xg;
                if (nwarps > 1) {
                    if (wi_in_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    if (sg_id == 0) {
                        const unsigned wi_u = wi_in_sg;
                        sycl::float2 xx_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_total = warp_reduce_sum(xx_first, item_ct1);
                        xg_total = warp_reduce_sum(xg_first, item_ct1);
                    } else {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_total = xx;
                        xg_total = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                float inv_r = 0.f;
                float coeff = 0.f;
                if (tid == 0) {
                    const float sum_xx_f  = xx_total.x() + xx_total.y();
                    const float sum_xdz_f = xg_total.x() + xg_total.y();
                    const float mean_eps  = sum_xx_f / (float) D + eps;
                    const float sum_eps   = sum_xx_f + eps * (float) D;
                    inv_r = sycl::rsqrt(mean_eps);
                    coeff = -sum_xdz_f / sum_eps;
                }
                inv_r = sycl::group_broadcast(item_ct1.get_group(), inv_r);
                coeff = sycl::group_broadcast(item_ct1.get_group(), coeff);

                for (int64_t col = tid; col < D; col += WG) {
                    d_row[col] = (g_row[col] + coeff * x_row[col]) * inv_r;
                }
            });
    });

}

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int64_t ne00 = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float * dst_dd = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    l2_norm_f32_sycl(src0_dd, dst_dd, ne00, nrows, eps, main_stream, ctx.device);

}
