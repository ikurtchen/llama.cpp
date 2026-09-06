// SYCL rope op (rotary position embedding). Mirrors ggml-cuda/rope.cu's algorithm (YaRN-scaled
// rotary embedding, NORMAL and NEOX pairing schemes) but hand-written for SYCL, not translated.
// mrope/imrope/vision modes (used by the Qwen3-VL vision tower, not the dense text path) and the
// ROPE+VIEW+SET_ROWS fusion are not yet ported -- tracked as residual, not silently dropped.
#include "rope.hpp"

#include <sycl/sycl.hpp>

namespace {

inline float rope_yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / sycl::fmax(0.001f, high - low);
    return 1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
inline void rope_yarn(float theta_extrap, float freq_scale, float corr_low, float corr_high, int i0,
                       float ext_factor, float mscale, float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(corr_low, corr_high, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
}

// One work-item per (row, i0-pair). NORMAL mode rotates adjacent elements (i0, i0+1); NEOX mode
// rotates elements n_dims/2 apart (i0/2, i0/2 + n_dims/2), matching ggml's two pairing schemes.
template <typename T, bool is_neox>
void rope_kernel(sycl::queue & q, const T * x, T * dst,
                  int64_t ne00, int64_t ne01, int64_t ne02,
                  int64_t s01, int64_t s02, int64_t s03, int64_t s1, int64_t s2, int64_t s3,
                  int n_dims, int n_offs, const int32_t * pos, float freq_scale, float ext_factor,
                  float attn_factor, float corr_low, float corr_high, float theta_scale,
                  const float * freq_factors, int64_t nrows) {
    const int64_t half = ne00 / 2;
    const int64_t block_x = 64, block_y = 4;
    const int64_t nbx = (nrows + block_x - 1) / block_x;
    const int64_t nby = (half  + block_y - 1) / block_y;

    q.parallel_for(sycl::nd_range<2>(sycl::range<2>(nbx * block_x, nby * block_y),
                                      sycl::range<2>(block_x, block_y)),
        [=](sycl::nd_item<2> item) {
        const int64_t row_dst = item.get_global_id(0);
        const int64_t i0      = 2 * item.get_global_id(1);
        if (row_dst >= nrows || i0 >= ne00) {
            return;
        }

        const int64_t i3 = row_dst / (ne01 * ne02);
        const int64_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
        const int64_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

        const int64_t idst = i1 * s1 + i2 * s2 + i3 * s3;
        const int64_t ix   = i1 * s01 + i2 * s02 + i3 * s03;

        if (i0 < n_offs || i0 >= n_offs + n_dims) {
            // outside the rotated window: pass through unchanged
            dst[idst + i0 + 0] = x[ix + i0 + 0];
            dst[idst + i0 + 1] = x[ix + i0 + 1];
            return;
        }

        const int64_t iw = i0 - n_offs; // relative idx within the rotated window
        const float freq_factor = freq_factors ? freq_factors[iw / 2] : 1.0f;
        const float theta_base  = pos[i2] * sycl::pow(theta_scale, (float) (iw / 2));

        float cos_theta, sin_theta;
        rope_yarn(theta_base / freq_factor, freq_scale, corr_low, corr_high, iw, ext_factor, attn_factor,
                  cos_theta, sin_theta);

        if constexpr (is_neox) {
            const int64_t channel = i0 / 2 + n_offs / 2;
            const int64_t o0 = ix   + channel;
            const int64_t d0 = idst + channel;
            const float x0 = (float) x[o0];
            const float x1 = (float) x[o0 + n_dims / 2];
            dst[d0]              = (T) (x0 * cos_theta - x1 * sin_theta);
            dst[d0 + n_dims / 2] = (T) (x0 * sin_theta + x1 * cos_theta);
        } else {
            const int64_t o0 = ix   + i0;
            const int64_t d0 = idst + i0;
            const float x0 = (float) x[o0];
            const float x1 = (float) x[o0 + 1];
            dst[d0]     = (T) (x0 * cos_theta - x1 * sin_theta);
            dst[d0 + 1] = (T) (x0 * sin_theta + x1 * cos_theta);
        }
    });
}

template <typename T>
void dispatch_rope_mode(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool is_neox) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t nr   = ggml_nrows(src0);

    const int64_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const int64_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const int64_t s03 = src0->nb[3] / ggml_type_size(src0->type);
    const int64_t s1  = dst->nb[1]  / ggml_type_size(dst->type);
    const int64_t s2  = dst->nb[2]  / ggml_type_size(dst->type);
    const int64_t s3  = dst->nb[3]  / ggml_type_size(dst->type);

    const int n_dims     = ggml_get_op_params_i32(dst, 1);
    const int mode       = ggml_get_op_params_i32(dst, 2);
    const int n_ctx_orig = ggml_get_op_params_i32(dst, 4);
    const int n_offs     = ggml_get_op_params_i32(dst, 15);
    GGML_UNUSED(mode);

    const float freq_base   = ggml_get_op_params_f32(dst, 5);
    const float freq_scale  = ggml_get_op_params_f32(dst, 6);
    const float ext_factor  = ggml_get_op_params_f32(dst, 7);
    const float attn_factor = ggml_get_op_params_f32(dst, 8);
    const float beta_fast   = ggml_get_op_params_f32(dst, 9);
    const float beta_slow   = ggml_get_op_params_f32(dst, 10);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    const float theta_scale = sycl::pow(freq_base, -2.0f / n_dims);

    const float * freq_factors = src2 ? (const float *) src2->data : nullptr;
    const int32_t * pos        = (const int32_t *) src1->data;

    sycl::queue & q = ctx.stream();
    const T * x   = (const T *) src0->data;
    T *       out = (T *) dst->data;

    if (is_neox) {
        rope_kernel<T, true>(q, x, out, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs,
                              pos, freq_scale, ext_factor, attn_factor, corr_dims[0], corr_dims[1],
                              theta_scale, freq_factors, nr);
    } else {
        rope_kernel<T, false>(q, x, out, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs,
                               pos, freq_scale, ext_factor, attn_factor, corr_dims[0], corr_dims[1],
                               theta_scale, freq_factors, nr);
    }
}

} // namespace

void ggml_sycl_op_rope(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == src0->type);

    const int mode = ggml_get_op_params_i32(dst, 2);
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    if (src0->type == GGML_TYPE_F16) {
        dispatch_rope_mode<sycl::half>(ctx, dst, is_neox);
    } else {
        dispatch_rope_mode<float>(ctx, dst, is_neox);
    }
}

bool ggml_sycl_supports_rope(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    auto ok_type = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16; };
    if (!ok_type(src0->type) || op->type != src0->type) {
        return false;
    }
    const int mode = ggml_get_op_params_i32(op, 2);
    // Only the plain and NEOX pairing schemes are ported; mrope/imrope/vision (Qwen3-VL vision
    // tower) are residual work.
    return mode == 0 || mode == GGML_ROPE_TYPE_NEOX;
}
