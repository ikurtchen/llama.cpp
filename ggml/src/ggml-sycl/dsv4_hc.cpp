#include "dsv4_hc.hpp"
#include "common.hpp"
#include <cmath>

#define SYCL_DSV4_HC_BLOCK_SIZE 256
#define DSV4_HC 4

static inline void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }
        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static inline void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }
        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static void dsv4_hc_comb_kernel(
        const float * mixes, const float * scale, const float * base, float * dst,
        int64_t n_tokens,
        int64_t sm0, int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0, int64_t sd1, int64_t sd2,
        float eps, int32_t n_iter,
        const sycl::nd_item<1> & item) {
    constexpr int comb_offset = 2*DSV4_HC;

    const int64_t it = (int64_t)item.get_global_id(0);
    if (it >= n_tokens) {
        return;
    }

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = sycl::fmax(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = sycl::exp(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

static void dsv4_hc_pre_kernel(
        const float * x, const float * weights, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1,
        int64_t sd0, int64_t sd1,
        const sycl::nd_item<1> & item) {

    const int64_t ir = (int64_t)item.get_global_id(0);
    const int64_t nr = n_embd * n_tokens;
    if (ir >= nr) {
        return;
    }
    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = x[i0 * sx0 + it * sx2] * weights[it * sw1];
    for (int64_t ih = 1; ih < hc; ++ih) {
        const float xv = x[i0 * sx0 + ih * sx1 + it * sx2];
        const float wv = weights[ih * sw0 + it * sw1];
        sum += xv * wv;
    }
    dst[i0 * sd0 + it * sd1] = sum;
}

static void dsv4_hc_post_kernel(
        const float * x, const float * residual, const float * post, const float * comb, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        const sycl::nd_item<1> & item) {

    const int64_t ir = (int64_t)item.get_global_id(0);
    const int64_t nr = n_embd * hc * n_tokens;
    if (ir >= nr) {
        return;
    }
    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0 * sx0 + it * sx1] * post[idst * sp0 + it * sp1];
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0 * sr0 + isrc * sr1 + it * sr2] * comb[idst * sc0 + isrc * sc1 + it * sc2];
    }
    dst[i0 * sd0 + idst * sd1 + it * sd2] = sum;
}

void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0]   == DSV4_HC);
    GGML_ASSERT(dst->ne[1]   == DSV4_HC);
    GGML_ASSERT(dst->ne[2]   == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0]  == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const float * mixes_d = (const float *) mixes->data;
    const float * scale_d = (const float *) scale->data;
    const float * base_d  = (const float *) base->data;
    float       * dst_d   = (float       *) dst->data;

    const int64_t n_tokens = mixes->ne[1];
    const float   eps      = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter   = ggml_get_op_params_i32(dst, 1);

    const int64_t sm0  = (int64_t)(nbm0 / sizeof(float));
    const int64_t sm1  = (int64_t)(nbm1 / sizeof(float));
    const int64_t ss0  = (int64_t)(nbs0 / sizeof(float));
    const int64_t sb0  = (int64_t)(nbb0 / sizeof(float));
    const int64_t sd0  = (int64_t)(nbd0 / sizeof(float));
    const int64_t sd1  = (int64_t)(nbd1 / sizeof(float));
    const int64_t sd2  = (int64_t)(nbd2 / sizeof(float));

    const int64_t num_blocks = (n_tokens + SYCL_DSV4_HC_BLOCK_SIZE - 1) / SYCL_DSV4_HC_BLOCK_SIZE;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    stream->parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_DSV4_HC_BLOCK_SIZE, SYCL_DSV4_HC_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            dsv4_hc_comb_kernel(
                mixes_d, scale_d, base_d, dst_d,
                n_tokens,
                sm0, sm1,
                ss0,
                sb0,
                sd0, sd1, sd2,
                eps, n_iter, item);
        });
}

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const float * x_d       = (const float *) x->data;
    const float * weights_d = (const float *) weights->data;
    float       * dst_d     = (float       *) dst->data;

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const int64_t sx0  = (int64_t)(nbx0 / sizeof(float));
    const int64_t sx1  = (int64_t)(nbx1 / sizeof(float));
    const int64_t sx2  = (int64_t)(nbx2 / sizeof(float));
    const int64_t sw0  = (int64_t)(nbw0 / sizeof(float));
    const int64_t sw1  = (int64_t)(nbw1 / sizeof(float));
    const int64_t sd0  = (int64_t)(nbd0 / sizeof(float));
    const int64_t sd1  = (int64_t)(nbd1 / sizeof(float));

    const int64_t nr = n_embd * n_tokens;
    const int64_t num_blocks = (nr + SYCL_DSV4_HC_BLOCK_SIZE - 1) / SYCL_DSV4_HC_BLOCK_SIZE;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    stream->parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_DSV4_HC_BLOCK_SIZE, SYCL_DSV4_HC_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            dsv4_hc_pre_kernel(
                x_d, weights_d, dst_d,
                n_embd, hc, n_tokens,
                sx0, sx1, sx2,
                sw0, sw1,
                sd0, sd1, item);
        });
}

void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);

    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const float * x_d        = (const float *) x->data;
    const float * residual_d = (const float *) residual->data;
    const float * post_d     = (const float *) post->data;
    const float * comb_d     = (const float *) comb->data;
    float       * dst_d      = (float       *) dst->data;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int64_t sx0  = (int64_t)(nbx0 / sizeof(float));
    const int64_t sx1  = (int64_t)(nbx1 / sizeof(float));
    const int64_t sr0  = (int64_t)(nbr0 / sizeof(float));
    const int64_t sr1  = (int64_t)(nbr1 / sizeof(float));
    const int64_t sr2  = (int64_t)(nbr2 / sizeof(float));
    const int64_t sp0  = (int64_t)(nbp0 / sizeof(float));
    const int64_t sp1  = (int64_t)(nbp1 / sizeof(float));
    const int64_t sc0  = (int64_t)(nbc0 / sizeof(float));
    const int64_t sc1  = (int64_t)(nbc1 / sizeof(float));
    const int64_t sc2  = (int64_t)(nbc2 / sizeof(float));
    const int64_t sd0  = (int64_t)(nbd0 / sizeof(float));
    const int64_t sd1  = (int64_t)(nbd1 / sizeof(float));
    const int64_t sd2  = (int64_t)(nbd2 / sizeof(float));

    const int64_t nr = n_embd * hc * n_tokens;
    const int64_t num_blocks = (nr + SYCL_DSV4_HC_BLOCK_SIZE - 1) / SYCL_DSV4_HC_BLOCK_SIZE;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    stream->parallel_for(
        sycl::nd_range<1>(num_blocks * SYCL_DSV4_HC_BLOCK_SIZE, SYCL_DSV4_HC_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            dsv4_hc_post_kernel(
                x_d, residual_d, post_d, comb_d, dst_d,
                n_embd, hc, n_tokens,
                sx0, sx1,
                sr0, sr1, sr2,
                sp0, sp1,
                sc0, sc1, sc2,
                sd0, sd1, sd2, item);
        });
}
