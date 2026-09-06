#include "dsv4-hc.hpp"

#include <sycl/sycl.hpp>

namespace {

constexpr int DSV4_HC = 4;
constexpr int DSV4_HC_COMB_OFFSET = 2*DSV4_HC;
constexpr size_t DSV4_HC_BLOCK_SIZE = 256;

inline void dsv4_hc_comb_norm_cols(float * comb, float eps) {
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

inline void dsv4_hc_comb_norm_rows(float * comb, float eps) {
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

} // namespace

void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;
    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb)
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb)
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb)
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb)

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    const int64_t sm0 = (int64_t) nbm0 / sizeof(float);
    const int64_t sm1 = (int64_t) nbm1 / sizeof(float);
    const int64_t ss0 = (int64_t) nbs0 / sizeof(float);
    const int64_t sb0 = (int64_t) nbb0 / sizeof(float);
    const int64_t sd0 = (int64_t) nbd0 / sizeof(float);
    const int64_t sd1 = (int64_t) nbd1 / sizeof(float);
    const int64_t sd2 = (int64_t) nbd2 / sizeof(float);

    const float * mixes_data = (const float *) mixes->data;
    const float * scale_data = (const float *) scale->data;
    const float * base_data  = (const float *) base->data;
    float * dst_data         = (float *) dst->data;

    sycl::queue & q = ctx.stream();
    const size_t global_size = ((size_t) n_tokens + DSV4_HC_BLOCK_SIZE - 1) / DSV4_HC_BLOCK_SIZE * DSV4_HC_BLOCK_SIZE;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(DSV4_HC_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) {
        const int64_t it = (int64_t) item.get_global_id(0);
        if (it >= n_tokens) {
            return;
        }

        const float scale_comb = scale_data[2*ss0];
        float comb[DSV4_HC*DSV4_HC];

        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            float maxv = -INFINITY;
            for (int idst = 0; idst < DSV4_HC; ++idst) {
                const int idx = idst + DSV4_HC*isrc;
                const float v = mixes_data[(DSV4_HC_COMB_OFFSET + idx) * sm0 + it * sm1] *
                                scale_comb + base_data[(DSV4_HC_COMB_OFFSET + idx) * sb0];
                comb[idx] = v;
                maxv = sycl::fmax(maxv, v);
            }

            float sum = 0.0f;
            for (int idst = 0; idst < DSV4_HC; ++idst) {
                const int idx = idst + DSV4_HC*isrc;
                const float v = sycl::exp(comb[idx] - maxv);
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
                dst_data[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
            }
        }
    });
}

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb)
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb)
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb)

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];
    const int64_t nr       = n_embd * n_tokens;
    const int64_t sx0 = (int64_t) nbx0 / sizeof(float);
    const int64_t sx1 = (int64_t) nbx1 / sizeof(float);
    const int64_t sx2 = (int64_t) nbx2 / sizeof(float);
    const int64_t sw0 = (int64_t) nbw0 / sizeof(float);
    const int64_t sw1 = (int64_t) nbw1 / sizeof(float);
    const int64_t sd0 = (int64_t) nbd0 / sizeof(float);
    const int64_t sd1 = (int64_t) nbd1 / sizeof(float);

    const float * x_data = (const float *) x->data;
    const float * w_data = (const float *) weights->data;
    float * dst_data     = (float *) dst->data;

    sycl::queue & q = ctx.stream();
    const size_t global_size = ((size_t) nr + DSV4_HC_BLOCK_SIZE - 1) / DSV4_HC_BLOCK_SIZE * DSV4_HC_BLOCK_SIZE;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(DSV4_HC_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) {
        const int64_t ir = (int64_t) item.get_global_id(0);
        if (ir >= nr) {
            return;
        }

        const int64_t i0 = ir % n_embd;
        const int64_t it = ir / n_embd;

        float sum = x_data[i0*sx0 + it*sx2] * w_data[it*sw1];
        for (int64_t ih = 1; ih < hc; ++ih) {
            sum += x_data[i0*sx0 + ih*sx1 + it*sx2] * w_data[ih*sw0 + it*sw1];
        }

        dst_data[i0*sd0 + it*sd1] = sum;
    });
}

void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb)
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb)
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb)
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb)
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb)

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];
    const int64_t nr       = n_embd * hc * n_tokens;
    const int64_t sx0 = (int64_t) nbx0 / sizeof(float);
    const int64_t sx1 = (int64_t) nbx1 / sizeof(float);
    const int64_t sr0 = (int64_t) nbr0 / sizeof(float);
    const int64_t sr1 = (int64_t) nbr1 / sizeof(float);
    const int64_t sr2 = (int64_t) nbr2 / sizeof(float);
    const int64_t sp0 = (int64_t) nbp0 / sizeof(float);
    const int64_t sp1 = (int64_t) nbp1 / sizeof(float);
    const int64_t sc0 = (int64_t) nbc0 / sizeof(float);
    const int64_t sc1 = (int64_t) nbc1 / sizeof(float);
    const int64_t sc2 = (int64_t) nbc2 / sizeof(float);
    const int64_t sd0 = (int64_t) nbd0 / sizeof(float);
    const int64_t sd1 = (int64_t) nbd1 / sizeof(float);
    const int64_t sd2 = (int64_t) nbd2 / sizeof(float);

    const float * x_data    = (const float *) x->data;
    const float * r_data    = (const float *) residual->data;
    const float * post_data = (const float *) post->data;
    const float * comb_data = (const float *) comb->data;
    float * dst_data        = (float *) dst->data;

    sycl::queue & q = ctx.stream();
    const size_t global_size = ((size_t) nr + DSV4_HC_BLOCK_SIZE - 1) / DSV4_HC_BLOCK_SIZE * DSV4_HC_BLOCK_SIZE;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(DSV4_HC_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) {
        const int64_t ir = (int64_t) item.get_global_id(0);
        if (ir >= nr) {
            return;
        }

        const int64_t i0   = ir % n_embd;
        const int64_t idst = (ir / n_embd) % hc;
        const int64_t it   = ir / (n_embd * hc);

        float sum = x_data[i0*sx0 + it*sx1] * post_data[idst*sp0 + it*sp1];
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += r_data[i0*sr0 + isrc*sr1 + it*sr2] * comb_data[idst*sc0 + isrc*sc1 + it*sc2];
        }

        dst_data[i0*sd0 + idst*sd1 + it*sd2] = sum;
    });
}

bool ggml_sycl_supports_dsv4_hc_comb(const ggml_tensor * op) {
    return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
           op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
}

bool ggml_sycl_supports_dsv4_hc_pre(const ggml_tensor * op) {
    return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32;
}

bool ggml_sycl_supports_dsv4_hc_post(const ggml_tensor * op) {
    return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
           op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32;
}
