// SYCL norm kernels. Mirrors ggml-cuda/norm.cu's block-per-row reduction shape,
// but is hand-written for SYCL 2020 with sub-group and work-group collectives.
#include "norm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

static size_t ggml_sycl_pick_norm_wg(const sycl::queue & q, int64_t extent) {
    const size_t max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const size_t target = extent < 256 ? 32 : 256;

    for (size_t cand : { size_t(256), size_t(128), size_t(64), size_t(32), size_t(16) }) {
        if (cand <= target && cand <= max_wg) {
            return cand;
        }
    }

    GGML_ASSERT(max_wg > 0);
    return max_wg;
}

template <typename T>
static inline T ggml_sycl_work_group_sum(sycl::nd_item<1> item, sycl::local_accessor<T, 1> scratch, T value) {
    const sycl::sub_group sg = item.get_sub_group();
    const uint32_t lid       = item.get_local_id(0);
    const uint32_t sg_lid    = sg.get_local_id()[0];
    const uint32_t sg_size   = sg.get_local_range()[0];
    const uint32_t sg_id     = lid / sg_size;
    const uint32_t wg_size   = item.get_local_range()[0];
    const uint32_t sg_count  = (wg_size + sg_size - 1) / sg_size;

    const T sg_total = sycl::reduce_over_group(sg, value, sycl::plus<T>());
    if (sg_lid == 0) {
        scratch[sg_id] = sg_total;
    }
    sycl::group_barrier(item.get_group());

    if (sg_id == 0) {
        const T partial = lid < sg_count ? scratch[lid] : T(0);
        const T wg_total = sycl::reduce_over_group(sg, partial, sycl::plus<T>());
        if (sg_lid == 0) {
            scratch[0] = wg_total;
        }
    }
    sycl::group_barrier(item.get_group());

    return scratch[0];
}

static void norm_f32(sycl::queue & q,
                     const float * x, float * dst,
                     int32_t ncols,
                     int64_t nrows, int64_t nchannels, int64_t nsamples,
                     int64_t x_stride_row, int64_t x_stride_channel, int64_t x_stride_sample,
                     int64_t y_stride_row, int64_t y_stride_channel, int64_t y_stride_sample,
                     float eps) {
    const size_t wg = ggml_sycl_pick_norm_wg(q, ncols);
    const size_t total_rows = (size_t) (nrows * nchannels * nsamples);

    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> scratch_sum   (sycl::range<1>(wg), h);
        sycl::local_accessor<float, 1> scratch_sumsq (sycl::range<1>(wg), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_rows * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
            const int64_t row_id           = (int64_t) item.get_group(0);
            const int64_t rows_per_sample  = nrows * nchannels;
            const int64_t sample           = row_id / rows_per_sample;
            const int64_t sample_row       = row_id - sample * rows_per_sample;
            const int64_t channel          = sample_row / nrows;
            const int64_t row              = sample_row - channel * nrows;
            const size_t  lid              = item.get_local_id(0);

            const float * row_x = x   + sample * x_stride_sample + channel * x_stride_channel + row * x_stride_row;
            float *       row_y = dst + sample * y_stride_sample + channel * y_stride_channel + row * y_stride_row;

            float sum = 0.0f;
            float sumsq = 0.0f;
            for (int32_t col = (int32_t) lid; col < ncols; col += (int32_t) wg) {
                const float xi = row_x[col];
                sum   += xi;
                sumsq += xi * xi;
            }

            const float total_sum   = ggml_sycl_work_group_sum(item, scratch_sum, sum);
            const float total_sumsq = ggml_sycl_work_group_sum(item, scratch_sumsq, sumsq);
            const float mean        = total_sum / ncols;
            const float variance    = total_sumsq / ncols - mean * mean;
            const float inv_std     = sycl::rsqrt(variance + eps);

            for (int32_t col = (int32_t) lid; col < ncols; col += (int32_t) wg) {
                row_y[col] = (row_x[col] - mean) * inv_std;
            }
        });
    });
}

static void rms_norm_f32(sycl::queue & q,
                         const float * x, float * dst,
                         int32_t ncols,
                         int64_t nrows, int64_t nchannels, int64_t nsamples,
                         int64_t x_stride_row, int64_t x_stride_channel, int64_t x_stride_sample,
                         int64_t y_stride_row, int64_t y_stride_channel, int64_t y_stride_sample,
                         float eps) {
    const size_t wg = ggml_sycl_pick_norm_wg(q, ncols);
    const size_t total_rows = (size_t) (nrows * nchannels * nsamples);

    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> scratch_sum(sycl::range<1>(wg), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_rows * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
            const int64_t row_id           = (int64_t) item.get_group(0);
            const int64_t rows_per_sample  = nrows * nchannels;
            const int64_t sample           = row_id / rows_per_sample;
            const int64_t sample_row       = row_id - sample * rows_per_sample;
            const int64_t channel          = sample_row / nrows;
            const int64_t row              = sample_row - channel * nrows;
            const size_t  lid              = item.get_local_id(0);

            const float * row_x = x   + sample * x_stride_sample + channel * x_stride_channel + row * x_stride_row;
            float *       row_y = dst + sample * y_stride_sample + channel * y_stride_channel + row * y_stride_row;

            float sumsq = 0.0f;
            for (int32_t col = (int32_t) lid; col < ncols; col += (int32_t) wg) {
                const float xi = row_x[col];
                sumsq += xi * xi;
            }

            const float total_sumsq = ggml_sycl_work_group_sum(item, scratch_sum, sumsq);
            const float scale       = sycl::rsqrt(total_sumsq / ncols + eps);

            for (int32_t col = (int32_t) lid; col < ncols; col += (int32_t) wg) {
                row_y[col] = scale * row_x[col];
            }
        });
    });
}

static void group_norm_f32(sycl::queue & q,
                           const float * x, float * dst,
                           int64_t n_groups_total, int32_t group_size, int64_t ne_elements,
                           float eps) {
    const size_t wg = ggml_sycl_pick_norm_wg(q, group_size);

    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> scratch_sum(sycl::range<1>(wg), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_groups_total * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
            const int64_t group = (int64_t) item.get_group(0);
            const size_t  lid   = item.get_local_id(0);
            const int64_t base  = group * group_size;
            const int64_t end   = base + group_size < ne_elements ? base + group_size : ne_elements;

            float sum = 0.0f;
            for (int64_t j = base + (int64_t) lid; j < end; j += (int64_t) wg) {
                sum += x[j];
            }

            const float mean = ggml_sycl_work_group_sum(item, scratch_sum, sum) / group_size;

            float sumsq = 0.0f;
            for (int64_t j = base + (int64_t) lid; j < end; j += (int64_t) wg) {
                const float centered = x[j] - mean;
                dst[j] = centered;
                sumsq += centered * centered;
            }

            const float variance = ggml_sycl_work_group_sum(item, scratch_sum, sumsq) / group_size;
            const float scale    = sycl::rsqrt(variance + eps);

            for (int64_t j = base + (int64_t) lid; j < end; j += (int64_t) wg) {
                dst[j] *= scale;
            }
        });
    });
}

static void l2_norm_f32(sycl::queue & q,
                        const float * x, float * dst,
                        int32_t ncols,
                        int64_t nrows, int64_t nchannels, int64_t nsamples,
                        int64_t x_stride_row, int64_t x_stride_channel, int64_t x_stride_sample,
                        int64_t y_stride_row, int64_t y_stride_channel, int64_t y_stride_sample,
                        float eps) {
    const size_t wg = ggml_sycl_pick_norm_wg(q, ncols);
    const size_t total_rows = (size_t) (nrows * nchannels * nsamples);

    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> scratch_sum(sycl::range<1>(wg), h);

        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_rows * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
            const int64_t row_id           = (int64_t) item.get_group(0);
            const int64_t rows_per_sample  = nrows * nchannels;
            const int64_t sample           = row_id / rows_per_sample;
            const int64_t sample_row       = row_id - sample * rows_per_sample;
            const int64_t channel          = sample_row / nrows;
            const int64_t row              = sample_row - channel * nrows;
            const size_t  lid              = item.get_local_id(0);

            const float * row_x = x   + sample * x_stride_sample + channel * x_stride_channel + row * x_stride_row;
            float *       row_y = dst + sample * y_stride_sample + channel * y_stride_channel + row * y_stride_row;

            float sumsq = 0.0f;
            for (int32_t col = (int32_t) lid; col < ncols; col += (int32_t) wg) {
                const float xi = row_x[col];
                sumsq += xi * xi;
            }

            const float total_sumsq = ggml_sycl_work_group_sum(item, scratch_sum, sumsq);
            const float scale       = sycl::rsqrt(sycl::fmax(total_sumsq, eps * eps));

            for (int32_t col = (int32_t) lid; col < ncols; col += (int32_t) wg) {
                row_y[col] = scale * row_x[col];
            }
        });
    });
}

static bool ggml_sycl_supports_rowwise_norm(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_is_contiguous_rows(src0) &&
           ggml_is_contiguous_rows(op);
}

} // namespace

void ggml_sycl_op_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(dst));

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    GGML_ASSERT(ne00 <= std::numeric_limits<int32_t>::max());
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    norm_f32(ctx.stream(),
             (const float *) src0->data,
             (float *) dst->data,
             (int32_t) ne00,
             ne01, ne02, ne03,
             (int64_t) (nb01 / sizeof(float)),
             (int64_t) (nb02 / sizeof(float)),
             (int64_t) (nb03 / sizeof(float)),
             (int64_t) (nb1  / sizeof(float)),
             (int64_t) (nb2  / sizeof(float)),
             (int64_t) (nb3  / sizeof(float)),
             eps);
}

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(dst));

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    GGML_ASSERT(ne00 <= std::numeric_limits<int32_t>::max());
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    rms_norm_f32(ctx.stream(),
                 (const float *) src0->data,
                 (float *) dst->data,
                 (int32_t) ne00,
                 ne01, ne02, ne03,
                 (int64_t) (nb01 / sizeof(float)),
                 (int64_t) (nb02 / sizeof(float)),
                 (int64_t) (nb03 / sizeof(float)),
                 (int64_t) (nb1  / sizeof(float)),
                 (int64_t) (nb2  / sizeof(float)),
                 (int64_t) (nb3  / sizeof(float)),
                 eps);
}

void ggml_sycl_op_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int32_t num_groups = 0;
    std::memcpy(&num_groups, dst->op_params, sizeof(num_groups));

    float eps;
    std::memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(num_groups > 0);
    GGML_ASSERT(eps >= 0.0f);

    const int32_t n_channels = (int32_t) src0->ne[2];
    const int32_t n_channels_per_group = (n_channels + num_groups - 1) / num_groups;
    const int64_t group_size64 = src0->ne[0] * src0->ne[1] * n_channels_per_group;

    GGML_ASSERT(group_size64 > 0);
    GGML_ASSERT(group_size64 <= std::numeric_limits<int32_t>::max());
    const int32_t group_size = (int32_t) group_size64;
    group_norm_f32(ctx.stream(),
                   (const float *) src0->data,
                   (float *) dst->data,
                   (int64_t) num_groups * src0->ne[3],
                   group_size,
                   ggml_nelements(src0),
                   eps);
}

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(dst));

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    GGML_ASSERT(ne00 <= std::numeric_limits<int32_t>::max());
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    l2_norm_f32(ctx.stream(),
                (const float *) src0->data,
                (float *) dst->data,
                (int32_t) ne00,
                ne01, ne02, ne03,
                (int64_t) (nb01 / sizeof(float)),
                (int64_t) (nb02 / sizeof(float)),
                (int64_t) (nb03 / sizeof(float)),
                (int64_t) (nb1  / sizeof(float)),
                (int64_t) (nb2  / sizeof(float)),
                (int64_t) (nb3  / sizeof(float)),
                eps);
}

bool ggml_sycl_supports_norm(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_sycl_supports_rowwise_norm(op);
        case GGML_OP_GROUP_NORM:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        default:
            return false;
    }
}
