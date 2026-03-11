#include "rope.hpp"

// ──────────────────────────────────────────────────────────────────
//  ROPE: Rotary Position Embedding (norm, neox, multi, vision)
//  Port from ggml-cuda/rope.cu
// ──────────────────────────────────────────────────────────────────

struct rope_corr_dims {
    float v[2];
};

struct mrope_sections {
    int v[4];
};

// ── helpers ──────────────────────────────────────────────────────

static inline float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / sycl::fmax(0.001f, high - low);
    return 1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y));
}

template <bool forward>
static inline void rope_yarn(
        const float theta_extrap, const float freq_scale,
        const rope_corr_dims corr_dims, const int64_t i0,
        const float ext_factor, float mscale,
        float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * sycl::native::log(1.0f / freq_scale);
    }
    cos_theta = sycl::native::cos(theta) * mscale;
    sin_theta = sycl::native::sin(theta) * mscale;
    if constexpr (!forward) {
        sin_theta *= -1.0f;
    }
}

// ── kernel: rope_norm ────────────────────────────────────────────
// Standard RoPE: adjacent pairs (x[i], x[i+1]) rotated together

template <bool forward, bool has_ff, typename T>
static void rope_norm_sycl(
        const T * x, T * dst,
        const int ne0, const int ne1,
        const int s1, const int s2,
        const int n_dims, const int nr,
        const int32_t * pos,
        const float freq_scale, const float freq_base,
        const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        sycl::queue & q) {

    GGML_ASSERT(ne0 % 2 == 0);
    const int block_size = SYCL_ROPE_BLOCK_SIZE;
    const int n_blocks_x = (ne0 + 2 * block_size - 1) / (2 * block_size);
    const float theta_scale = sycl::pow(freq_base, -2.0f / n_dims);
    const float theta_scale_log2 = sycl::log2(theta_scale);

    const sycl::range<3> block_dims(1, block_size, 1);
    const sycl::range<3> block_nums(1, n_blocks_x, nr);

    q.parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item) {
            const int i0 = 2 * (item.get_local_range(1) * item.get_group(1) + item.get_local_id(1));
            if (i0 >= ne0) return;

            const int row      = item.get_local_range(2) * item.get_group(2) + item.get_local_id(2);
            const int row0     = row % ne1;
            const int channel0 = row / ne1;

            const int idst = row * ne0 + i0;
            const int ix   = channel0 * s2 + row0 * s1 + i0;

            if (i0 >= n_dims) {
                dst[idst + 0] = x[ix + 0];
                dst[idst + 1] = x[ix + 1];
                return;
            }

            const float theta_base = pos[channel0] * sycl::native::exp2(theta_scale_log2 * (i0 / 2.0f));
            const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

            float cos_theta, sin_theta;
            rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                               ext_factor, attn_factor, cos_theta, sin_theta);

            const float x0 = static_cast<float>(x[ix + 0]);
            const float x1 = static_cast<float>(x[ix + 1]);

            dst[idst + 0] = static_cast<T>(x0 * cos_theta - x1 * sin_theta);
            dst[idst + 1] = static_cast<T>(x0 * sin_theta + x1 * cos_theta);
        });
}

// ── kernel: rope_neox ────────────────────────────────────────────
// NeoX-style: pairs at offset n_dims/2 (x[i], x[i + n_dims/2])

template <bool forward, bool has_ff, typename T>
static void rope_neox_sycl(
        const T * x, T * dst,
        const int ne0, const int ne1,
        const int s1, const int s2,
        const int n_dims, const int nr,
        const int32_t * pos,
        const float freq_scale, const float freq_base,
        const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        sycl::queue & q) {

    GGML_ASSERT(ne0 % 2 == 0);
    const int block_size = SYCL_ROPE_BLOCK_SIZE;
    const int n_blocks_x = (ne0 + 2 * block_size - 1) / (2 * block_size);
    const float theta_scale = sycl::pow(freq_base, -2.0f / n_dims);
    const float theta_scale_log2 = sycl::log2(theta_scale);

    const sycl::range<3> block_dims(1, block_size, 1);
    const sycl::range<3> block_nums(1, n_blocks_x, nr);

    q.parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item) {
            const int i0 = 2 * (item.get_local_range(1) * item.get_group(1) + item.get_local_id(1));
            if (i0 >= ne0) return;

            const int row      = item.get_local_range(2) * item.get_group(2) + item.get_local_id(2);
            const int row0     = row % ne1;
            const int channel0 = row / ne1;

            const int idst = row * ne0 + i0 / 2;
            const int ix   = channel0 * s2 + row0 * s1 + i0 / 2;

            if (i0 >= n_dims) {
                dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
                dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];
                return;
            }

            const float theta_base = pos[channel0] * sycl::native::exp2(theta_scale_log2 * (i0 / 2.0f));
            const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

            float cos_theta, sin_theta;
            rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                               ext_factor, attn_factor, cos_theta, sin_theta);

            const float x0 = static_cast<float>(x[ix + 0]);
            const float x1 = static_cast<float>(x[ix + n_dims / 2]);

            dst[idst + 0]          = static_cast<T>(x0 * cos_theta - x1 * sin_theta);
            dst[idst + n_dims / 2] = static_cast<T>(x0 * sin_theta + x1 * cos_theta);
        });
}

// ── kernel: rope_multi ───────────────────────────────────────────
// Multi-position RoPE (M-RoPE) for multi-modal / interleaved positions

template <bool forward, bool has_ff, typename T>
static void rope_multi_sycl(
        const T * x, T * dst,
        const int ne0, const int ne1, const int ne2,
        const int s1, const int s2,
        const int n_dims, const int nr,
        const int32_t * pos,
        const float freq_scale, const float freq_base,
        const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const mrope_sections sections,
        const bool is_imrope,
        sycl::queue & q) {

    GGML_ASSERT(ne0 % 2 == 0);
    const int block_size = SYCL_ROPE_BLOCK_SIZE;
    const int n_blocks_x = (ne0 + 2 * block_size - 1) / (2 * block_size);
    const float theta_scale = sycl::pow(freq_base, -2.0f / n_dims);
    const float theta_scale_log2 = sycl::log2(theta_scale);

    const sycl::range<3> block_dims(1, block_size, 1);
    const sycl::range<3> block_nums(1, n_blocks_x, nr);

    q.parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item) {
            const int i0 = 2 * (item.get_local_range(1) * item.get_group(1) + item.get_local_id(1));
            if (i0 >= ne0) return;

            const int row      = item.get_local_range(2) * item.get_group(2) + item.get_local_id(2);
            const int row0     = row % ne1;
            const int channel0 = row / ne1;

            const int idst = row * ne0 + i0 / 2;
            const int ix   = channel0 * s2 + row0 * s1 + i0 / 2;

            if (i0 >= n_dims) {
                dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
                dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];
                return;
            }

            const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
            const int sec_w = sections.v[1] + sections.v[0];
            const int sector = (i0 / 2) % sect_dims;

            const float theta_pow = sycl::native::exp2(theta_scale_log2 * (i0 / 2.0f));
            float theta_base = 0.0f;
            if (is_imrope) {
                if (sector % 3 == 1 && sector < 3 * sections.v[1]) {           // h
                    theta_base = pos[channel0 + ne2 * 1] * theta_pow;
                } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {    // w
                    theta_base = pos[channel0 + ne2 * 2] * theta_pow;
                } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {    // t
                    theta_base = pos[channel0] * theta_pow;
                } else {
                    theta_base = pos[channel0 + ne2 * 3] * theta_pow;
                }
            } else {
                if (sector < sections.v[0]) {
                    theta_base = pos[channel0] * theta_pow;
                } else if (sector >= sections.v[0] && sector < sec_w) {
                    theta_base = pos[channel0 + ne2 * 1] * theta_pow;
                } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
                    theta_base = pos[channel0 + ne2 * 2] * theta_pow;
                } else if (sector >= sec_w + sections.v[2]) {
                    theta_base = pos[channel0 + ne2 * 3] * theta_pow;
                }
            }

            const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

            float cos_theta, sin_theta;
            rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                               ext_factor, attn_factor, cos_theta, sin_theta);

            const float x0 = static_cast<float>(x[ix + 0]);
            const float x1 = static_cast<float>(x[ix + n_dims / 2]);

            dst[idst + 0]          = static_cast<T>(x0 * cos_theta - x1 * sin_theta);
            dst[idst + n_dims / 2] = static_cast<T>(x0 * sin_theta + x1 * cos_theta);
        });
}

// ── kernel: rope_vision ──────────────────────────────────────────
// Vision-model RoPE with 2-section position lookup

template <bool forward, bool has_ff, typename T>
static void rope_vision_sycl(
        const T * x, T * dst,
        const int ne0, const int ne1, const int ne2,
        const int s1, const int s2,
        const int n_dims, const int nr,
        const int32_t * pos,
        const float freq_scale, const float freq_base,
        const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const mrope_sections sections,
        sycl::queue & q) {

    GGML_ASSERT(ne0 % 2 == 0);
    const int block_size = SYCL_ROPE_BLOCK_SIZE;
    const int n_blocks_x = (ne0 + 2 * block_size - 1) / (2 * block_size);
    const float theta_scale = sycl::pow(freq_base, -2.0f / n_dims);
    const float theta_scale_log2 = sycl::log2(theta_scale);

    const sycl::range<3> block_dims(1, block_size, 1);
    const sycl::range<3> block_nums(1, n_blocks_x, nr);

    q.parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item) {
            const int i0 = 2 * (item.get_local_range(1) * item.get_group(1) + item.get_local_id(1));
            if (i0 >= ne0) return;

            const int row      = item.get_local_range(2) * item.get_group(2) + item.get_local_id(2);
            const int row0     = row % ne1;
            const int channel0 = row / ne1;

            const int idst = row * ne0 + i0 / 2;
            const int ix   = channel0 * s2 + row0 * s1 + i0 / 2;

            const int sect_dims = sections.v[0] + sections.v[1];
            const int sec_w     = sections.v[1] + sections.v[0];
            const int sector    = (i0 / 2) % sect_dims;

            float theta_base = 0.0f;
            if (sector < sections.v[0]) {
                const int p = sector;
                theta_base  = pos[channel0] * sycl::native::exp2(theta_scale_log2 * (float)p);
            } else if (sector >= sections.v[0] && sector < sec_w) {
                const int p = sector - sections.v[0];
                theta_base  = pos[channel0 + ne2] * sycl::native::exp2(theta_scale_log2 * (float)p);
            }

            const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

            float cos_theta, sin_theta;
            rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0,
                               ext_factor, attn_factor, cos_theta, sin_theta);

            const float x0 = static_cast<float>(x[ix + 0]);
            const float x1 = static_cast<float>(x[ix + n_dims]);

            dst[idst + 0]      = static_cast<T>(x0 * cos_theta - x1 * sin_theta);
            dst[idst + n_dims] = static_cast<T>(x0 * sin_theta + x1 * cos_theta);
        });
}

// ── host dispatch ────────────────────────────────────────────────

template <bool forward>
static void ggml_sycl_op_rope_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32 || dst->type  == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    GGML_ASSERT(src0->type == dst->type);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t nr   = ggml_nrows(src0);

    const int s01 = src0->nb[1] / ggml_type_size(src0->type);
    const int s02 = src0->nb[2] / ggml_type_size(src0->type);

    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    mrope_sections sections;

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int) * 4);

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }
    if (is_vision) {
        GGML_ASSERT(n_dims == ne00 / 2);
    }

    const int32_t * pos = (const int32_t *) src1->data;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    sycl::queue & q = *ctx.stream();

    // ── type dispatch macro ──
    #define DISPATCH_ROPE_FF(launch_fn, ...) \
        do { \
            if (src0->type == GGML_TYPE_F32) { \
                if (freq_factors == nullptr) { \
                    launch_fn<forward, false, float>((const float *)src0->data, (float *)dst->data, __VA_ARGS__); \
                } else { \
                    launch_fn<forward, true, float>((const float *)src0->data, (float *)dst->data, __VA_ARGS__); \
                } \
            } else { \
                if (freq_factors == nullptr) { \
                    launch_fn<forward, false, sycl::half>((const sycl::half *)src0->data, (sycl::half *)dst->data, __VA_ARGS__); \
                } else { \
                    launch_fn<forward, true, sycl::half>((const sycl::half *)src0->data, (sycl::half *)dst->data, __VA_ARGS__); \
                } \
            } \
        } while (0)

    if (is_neox) {
        DISPATCH_ROPE_FF(rope_neox_sycl,
            ne00, ne01, s01, s02,
            n_dims, nr, pos, freq_scale, freq_base,
            ext_factor, attn_factor, corr_dims, freq_factors, q);
    } else if (is_mrope && !is_vision) {
        DISPATCH_ROPE_FF(rope_multi_sycl,
            ne00, ne01, ne02, s01, s02,
            n_dims, nr, pos, freq_scale, freq_base,
            ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope, q);
    } else if (is_vision) {
        DISPATCH_ROPE_FF(rope_vision_sycl,
            ne00, ne01, ne02, s01, s02,
            n_dims, nr, pos, freq_scale, freq_base,
            ext_factor, attn_factor, corr_dims, freq_factors, sections, q);
    } else {
        DISPATCH_ROPE_FF(rope_norm_sycl,
            ne00, ne01, s01, s02,
            n_dims, nr, pos, freq_scale, freq_base,
            ext_factor, attn_factor, corr_dims, freq_factors, q);
    }

    #undef DISPATCH_ROPE_FF
}

void ggml_sycl_rope(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_rope_impl<true>(ctx, dst);
}

void ggml_sycl_rope_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_rope_impl<false>(ctx, dst);
}
