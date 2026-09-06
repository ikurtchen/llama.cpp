// SYCL softmax. Mirrors ggml-cuda/softmax.cu's row-wise reduction algorithm,
// but is hand-written for SYCL 2020 with work-group collectives.
#include "softmax.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

struct soft_max_params {
    int64_t ncols;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;
    int64_t ne12;
    int64_t ne13;
    uint32_t n_head_log2;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

template <typename T>
inline float ggml_sycl_t2f32(T v) {
    return (float) v;
}

inline float ggml_sycl_get_alibi_slope(float max_bias, uint32_t h, uint32_t n_head_log2, float m0, float m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    const float base = h < n_head_log2 ? m0 : m1;
    const int exph   = h < n_head_log2 ? (int) h + 1 : 2 * ((int) h - (int) n_head_log2) + 1;
    return sycl::pow(base, (float) exph);
}

static size_t ggml_sycl_pick_softmax_wg(const sycl::queue & q, int64_t ncols) {
    const size_t max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const size_t target = ncols >= 256 ? 256 : (ncols >= 128 ? 128 : (ncols >= 64 ? 64 : 32));

    for (size_t cand : { size_t(256), size_t(128), size_t(64), size_t(32), size_t(16) }) {
        if (cand <= target && cand <= max_wg) {
            return cand;
        }
    }

    GGML_ASSERT(max_wg > 0);
    return std::min(max_wg, size_t(16));
}

template <typename MaskT, bool has_mask, bool has_sinks>
static void soft_max_f32(sycl::queue & q,
                         const float * x,
                         const MaskT * mask,
                         const float * sinks,
                         float * dst,
                         const soft_max_params params) {
    const size_t wg = ggml_sycl_pick_softmax_wg(q, params.ncols);
    const size_t nrows = (size_t) (params.ne01 * params.ne02 * params.ne03);

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nrows * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> item) {
        const int64_t row = (int64_t) item.get_group(0);
        const int64_t lid = (int64_t) item.get_local_id(0);

        const int64_t rows_per_plane = params.ne01 * params.ne02;
        const int64_t i03 = row / rows_per_plane;
        const int64_t row_in_plane = row - i03 * rows_per_plane;
        const int64_t i02 = row_in_plane / params.ne01;
        const int64_t i01 = row_in_plane - i02 * params.ne01;

        const float * row_x = x + row * params.ncols;
        float * row_dst     = dst + row * params.ncols;

        const MaskT * row_mask = nullptr;
        if constexpr (has_mask) {
            const int64_t i11 = i01;
            const int64_t i12 = i02 % params.ne12;
            const int64_t i13 = i03 % params.ne13;
            row_mask = mask + (i11 * params.nb11 + i12 * params.nb12 + i13 * params.nb13) / (int64_t) sizeof(MaskT);
        }

        const float slope = ggml_sycl_get_alibi_slope(params.max_bias, (uint32_t) i02, params.n_head_log2, params.m0, params.m1);

        float local_max = -std::numeric_limits<float>::infinity();
        if constexpr (has_sinks) {
            local_max = sinks[i02];
        }

        for (int64_t col = lid; col < params.ncols; col += (int64_t) wg) {
            float val = row_x[col] * params.scale;
            if constexpr (has_mask) {
                val += slope * ggml_sycl_t2f32(row_mask[col]);
            }
            local_max = sycl::fmax(local_max, val);
        }

        const float max_val = sycl::reduce_over_group(item.get_group(), local_max, sycl::maximum<float>());

        float local_sum = 0.0f;
        for (int64_t col = lid; col < params.ncols; col += (int64_t) wg) {
            float val = row_x[col] * params.scale;
            if constexpr (has_mask) {
                val += slope * ggml_sycl_t2f32(row_mask[col]);
            }
            const float e = sycl::exp(val - max_val);
            row_dst[col] = e;
            local_sum += e;
        }

        float sum = sycl::reduce_over_group(item.get_group(), local_sum, sycl::plus<float>());
        if constexpr (has_sinks) {
            sum += sycl::exp(sinks[i02] - max_val);
        }

        const float inv_sum = 1.0f / sum;
        for (int64_t col = lid; col < params.ncols; col += (int64_t) wg) {
            row_dst[col] *= inv_sum;
        }
    });
}

static void soft_max_back_f32(sycl::queue & q,
                              const float * dy,
                              const float * y,
                              float * dx,
                              int64_t ncols,
                              int64_t nrows,
                              float scale) {
    const size_t wg = ggml_sycl_pick_softmax_wg(q, ncols);

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nrows * wg), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> item) {
        const int64_t row = (int64_t) item.get_group(0);
        const int64_t lid = (int64_t) item.get_local_id(0);

        const float * row_dy = dy + row * ncols;
        const float * row_y  = y  + row * ncols;
        float * row_dx       = dx + row * ncols;

        float dot_local = 0.0f;
        for (int64_t col = lid; col < ncols; col += (int64_t) wg) {
            dot_local += row_y[col] * row_dy[col];
        }

        const float dot = sycl::reduce_over_group(item.get_group(), dot_local, sycl::plus<float>());

        for (int64_t col = lid; col < ncols; col += (int64_t) wg) {
            row_dx[col] = scale * (row_dy[col] - dot) * row_y[col];
        }
    });
}

template <typename MaskT>
static void ggml_sycl_op_soft_max_mask(ggml_backend_sycl_context & ctx,
                                       ggml_tensor * dst,
                                       const MaskT * mask) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src2 = dst->src[2];

    const float scale    = ggml_get_op_params_f32(dst, 0);
    const float max_bias = ggml_get_op_params_f32(dst, 1);

    const uint32_t n_head      = (uint32_t) src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) std::floor(std::log2((float) n_head));

    soft_max_params params = {
        /* ncols       = */ src0->ne[0],
        /* ne01        = */ src0->ne[1],
        /* ne02        = */ src0->ne[2],
        /* ne03        = */ src0->ne[3],
        /* nb11        = */ dst->src[1]->nb[1],
        /* nb12        = */ dst->src[1]->nb[2],
        /* nb13        = */ dst->src[1]->nb[3],
        /* ne12        = */ dst->src[1]->ne[2],
        /* ne13        = */ dst->src[1]->ne[3],
        /* n_head_log2 = */ n_head_log2,
        /* scale       = */ scale,
        /* max_bias    = */ max_bias,
        /* m0          = */ std::pow(2.0f, -(max_bias) / n_head_log2),
        /* m1          = */ std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2),
    };

    sycl::queue & q = ctx.stream();
    const float * sinks = src2 ? (const float *) src2->data : nullptr;

    if (src2) {
        soft_max_f32<MaskT, true, true>(q, (const float *) src0->data, mask, sinks, (float *) dst->data, params);
    } else {
        soft_max_f32<MaskT, true, false>(q, (const float *) src0->data, mask, nullptr, (float *) dst->data, params);
    }
}

static void ggml_sycl_op_soft_max_nomask(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src2 = dst->src[2];

    const float scale    = ggml_get_op_params_f32(dst, 0);
    const float max_bias = ggml_get_op_params_f32(dst, 1);

    const uint32_t n_head      = (uint32_t) src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) std::floor(std::log2((float) n_head));

    soft_max_params params = {
        /* ncols       = */ src0->ne[0],
        /* ne01        = */ src0->ne[1],
        /* ne02        = */ src0->ne[2],
        /* ne03        = */ src0->ne[3],
        /* nb11        = */ 1,
        /* nb12        = */ 1,
        /* nb13        = */ 1,
        /* ne12        = */ 1,
        /* ne13        = */ 1,
        /* n_head_log2 = */ n_head_log2,
        /* scale       = */ scale,
        /* max_bias    = */ max_bias,
        /* m0          = */ std::pow(2.0f, -(max_bias) / n_head_log2),
        /* m1          = */ std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2),
    };

    sycl::queue & q = ctx.stream();
    const float * sinks = src2 ? (const float *) src2->data : nullptr;

    if (src2) {
        soft_max_f32<float, false, true>(q, (const float *) src0->data, nullptr, sinks, (float *) dst->data, params);
    } else {
        soft_max_f32<float, false, false>(q, (const float *) src0->data, nullptr, nullptr, (float *) dst->data, params);
    }
}

} // namespace

void ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float max_bias = ggml_get_op_params_f32(dst, 1);

    if (src2) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(src2));
        GGML_ASSERT(src2->ne[0] == src0->ne[2]);
    }

    if (src1 == nullptr) {
        GGML_ASSERT(max_bias == 0.0f);
        ggml_sycl_op_soft_max_nomask(ctx, dst);
        return;
    }

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src1->ne[0] == src0->ne[0]);
    GGML_ASSERT(src1->ne[1] >= src0->ne[1]);
    GGML_ASSERT(src0->ne[2] % src1->ne[2] == 0);
    GGML_ASSERT(src0->ne[3] % src1->ne[3] == 0);
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));

    if (src1->type == GGML_TYPE_F16) {
        ggml_sycl_op_soft_max_mask<sycl::half>(ctx, dst, (const sycl::half *) src1->data);
    } else {
        ggml_sycl_op_soft_max_mask<float>(ctx, dst, (const float *) src1->data);
    }
}

bool ggml_sycl_supports_soft_max(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];

    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src0, op) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        return false;
    }

    const float max_bias = ggml_get_op_params_f32(op, 1);
    if (src1 == nullptr) {
        if (max_bias != 0.0f) {
            return false;
        }
    } else {
        if ((src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) ||
            !ggml_is_contiguous(src1) ||
            src1->nb[0] != ggml_type_size(src1->type) ||
            src1->ne[0] != src0->ne[0] ||
            src1->ne[1] < src0->ne[1] ||
            src0->ne[2] % src1->ne[2] != 0 ||
            src0->ne[3] % src1->ne[3] != 0) {
            return false;
        }
    }

    if (src2 != nullptr) {
        if (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2) || src2->ne[0] != src0->ne[2]) {
            return false;
        }
    }

    return true;
}

void ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float scale    = ggml_get_op_params_f32(dst, 0);
    const float max_bias = ggml_get_op_params_f32(dst, 1);

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32(ctx.stream(),
                      (const float *) src0->data,
                      (const float *) src1->data,
                      (float *) dst->data,
                      src0->ne[0],
                      ggml_nrows(src0),
                      scale);
}

bool ggml_sycl_supports_soft_max_back(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src0, op) || !ggml_are_same_shape(src1, op)) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }

    const float max_bias = ggml_get_op_params_f32(op, 1);
    return max_bias == 0.0f;
}
