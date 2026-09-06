// SYCL flash attention for the dense decoder path:
// - Q in F32 or F16
// - K/V in F16
// - output in F32
// - causal/additive mask and ALiBi
// - GQA
#include "fattn.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

static constexpr size_t FATTN_WG = 128;

struct fattn_params {
    const void * q;
    const void * k;
    const void * v;
    const void * mask;
    void *       dst;

    int64_t dk;
    int64_t dv;

    int64_t q_ne1;
    int64_t q_ne2;
    int64_t q_ne3;
    int64_t k_ne1;
    int64_t k_ne2;
    int64_t k_ne3;
    int64_t v_ne2;
    int64_t v_ne3;

    int64_t q_nb1;
    int64_t q_nb2;
    int64_t q_nb3;
    int64_t k_nb1;
    int64_t k_nb2;
    int64_t k_nb3;
    int64_t v_nb1;
    int64_t v_nb2;
    int64_t v_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;

    int64_t mask_ne2;
    int64_t mask_ne3;
    int64_t mask_nb1;
    int64_t mask_nb2;
    int64_t mask_nb3;

    int64_t rk2;
    int64_t rk3;
    int64_t rv2;
    int64_t rv3;

    uint32_t n_head_log2;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

template <typename T>
inline float ggml_sycl_fattn_to_f32(T v) {
    return (float) v;
}

inline float ggml_sycl_fattn_get_alibi_slope(float max_bias, uint32_t h, uint32_t n_head_log2, float m0, float m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    const float base = h < n_head_log2 ? m0 : m1;
    const int exph   = h < n_head_log2 ? (int) h + 1 : 2 * ((int) h - (int) n_head_log2) + 1;
    return sycl::pow(base, (float) exph);
}

inline uint32_t ggml_sycl_floor_pow2(uint32_t x) {
    uint32_t p = 1;
    while ((p << 1) <= x) {
        p <<= 1;
    }
    return p;
}

template <typename QType, typename MaskType, bool has_mask>
static void ggml_sycl_flash_attn_ext_launch(sycl::queue & q, const fattn_params & p) {
    const int64_t nrows = p.q_ne1 * p.q_ne2 * p.q_ne3;

    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> q_row(sycl::range<1>((size_t) p.dk), cgh);
        sycl::local_accessor<float, 1> state(sycl::range<1>(4), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nrows * FATTN_WG), sycl::range<1>(FATTN_WG)),
            [=](sycl::nd_item<1> item) {
                const int64_t row = (int64_t) item.get_group(0);
                const int lid = (int) item.get_local_id(0);

                const int64_t rows_per_batch = p.q_ne2 * p.q_ne1;
                const int64_t iq3 = row / rows_per_batch;
                const int64_t row_rem = row - iq3 * rows_per_batch;
                const int64_t iq2 = row_rem / p.q_ne1;
                const int64_t iq1 = row_rem - iq2 * p.q_ne1;

                const char * q_ptr = (const char *) p.q + iq1 * p.q_nb1 + iq2 * p.q_nb2 + iq3 * p.q_nb3;

                for (int64_t d = lid; d < p.dk; d += (int64_t) FATTN_WG) {
                    q_row[d] = ggml_sycl_fattn_to_f32(((const QType *) q_ptr)[d]);
                }

                if (lid == 0) {
                    state[0] = -std::numeric_limits<float>::infinity();
                    state[1] = 0.0f;
                }
                item.barrier(sycl::access::fence_space::local_space);

                const int64_t ik2 = iq2 / p.rk2;
                const int64_t ik3 = iq3 / p.rk3;
                const int64_t iv2 = iq2 / p.rv2;
                const int64_t iv3 = iq3 / p.rv3;

                const float slope = ggml_sycl_fattn_get_alibi_slope(
                    p.max_bias, (uint32_t) iq2, p.n_head_log2, p.m0, p.m1);

                const MaskType * mask_row = nullptr;
                if constexpr (has_mask) {
                    mask_row = (const MaskType *) ((const char *) p.mask +
                        iq1 * p.mask_nb1 +
                        (iq2 % p.mask_ne2) * p.mask_nb2 +
                        (iq3 % p.mask_ne3) * p.mask_nb3);
                }

                float acc0 = 0.0f;
                float acc1 = 0.0f;
                const int64_t d0 = lid;
                const int64_t d1 = lid + (int64_t) FATTN_WG;
                const bool has_d0 = d0 < p.dv;
                const bool has_d1 = d1 < p.dv;

                for (int64_t ic = 0; ic < p.k_ne1; ++ic) {
                    float mv = 0.0f;
                    if constexpr (has_mask) {
                        mv = slope * ggml_sycl_fattn_to_f32(mask_row[ic]);
                        if (sycl::isinf(mv) && mv < 0.0f) {
                            continue;
                        }
                    }

                    const sycl::half * k_row_ptr = (const sycl::half *) ((const char *) p.k +
                        ic * p.k_nb1 + ik2 * p.k_nb2 + ik3 * p.k_nb3);

                    float partial = 0.0f;
                    for (int64_t d = lid; d < p.dk; d += (int64_t) FATTN_WG) {
                        partial += q_row[d] * (float) k_row_ptr[d];
                    }

                    const float score = sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());

                    if (lid == 0) {
                        const float s = score * p.scale + mv;
                        const float mold = state[0];

                        float scale_old = 1.0f;
                        float weight = 1.0f;

                        if (s > mold) {
                            state[0] = s;
                            scale_old = sycl::exp(mold - s);
                        } else {
                            weight = sycl::exp(s - mold);
                        }

                        state[1] = state[1] * scale_old + weight;
                        state[2] = scale_old;
                        state[3] = weight;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    const sycl::half * v_row_ptr = (const sycl::half *) ((const char *) p.v +
                        ic * p.v_nb1 + iv2 * p.v_nb2 + iv3 * p.v_nb3);

                    const float scale_old = state[2];
                    const float weight = state[3];

                    if (has_d0) {
                        acc0 = acc0 * scale_old + weight * (float) v_row_ptr[d0];
                    }
                    if (has_d1) {
                        acc1 = acc1 * scale_old + weight * (float) v_row_ptr[d1];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lid == 0) {
                    state[2] = state[1] == 0.0f ? 0.0f : 1.0f / state[1];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float * dst_row_ptr = (float *) ((char *) p.dst + iq2 * p.dst_nb1 + iq1 * p.dst_nb2 + iq3 * p.dst_nb3);
                const float inv_sum = state[2];

                if (has_d0) {
                    dst_row_ptr[d0] = acc0 * inv_sum;
                }
                if (has_d1) {
                    dst_row_ptr[d1] = acc1 * inv_sum;
                }
            });
    });
}

static bool ggml_sycl_flash_attn_ext_supports_head_dim(int64_t dim) {
    return dim == 64 || dim == 80 || dim == 128;
}

static bool ggml_sycl_flash_attn_ext_supports_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

} // namespace

void ggml_sycl_op_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    GGML_ASSERT(ggml_sycl_supports_flash_attn_ext(dst));
    GGML_ASSERT(ctx.stream().get_device().get_info<sycl::info::device::max_work_group_size>() >= FATTN_WG);

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(scale));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));

    const uint32_t n_head = (uint32_t) q->ne[2];
    const uint32_t n_head_log2 = ggml_sycl_floor_pow2(n_head);

    fattn_params p = {};
    p.q = q->data;
    p.k = k->data;
    p.v = v->data;
    p.mask = m ? m->data : nullptr;
    p.dst = dst->data;

    p.dk = k->ne[0];
    p.dv = v->ne[0];

    p.q_ne1 = q->ne[1];
    p.q_ne2 = q->ne[2];
    p.q_ne3 = q->ne[3];
    p.k_ne1 = k->ne[1];
    p.k_ne2 = k->ne[2];
    p.k_ne3 = k->ne[3];
    p.v_ne2 = v->ne[2];
    p.v_ne3 = v->ne[3];

    p.q_nb1 = (int64_t) q->nb[1];
    p.q_nb2 = (int64_t) q->nb[2];
    p.q_nb3 = (int64_t) q->nb[3];
    p.k_nb1 = (int64_t) k->nb[1];
    p.k_nb2 = (int64_t) k->nb[2];
    p.k_nb3 = (int64_t) k->nb[3];
    p.v_nb1 = (int64_t) v->nb[1];
    p.v_nb2 = (int64_t) v->nb[2];
    p.v_nb3 = (int64_t) v->nb[3];
    p.dst_nb1 = (int64_t) dst->nb[1];
    p.dst_nb2 = (int64_t) dst->nb[2];
    p.dst_nb3 = (int64_t) dst->nb[3];

    if (m) {
        p.mask_ne2 = m->ne[2];
        p.mask_ne3 = m->ne[3];
        p.mask_nb1 = (int64_t) m->nb[1];
        p.mask_nb2 = (int64_t) m->nb[2];
        p.mask_nb3 = (int64_t) m->nb[3];
    } else {
        p.mask_ne2 = 1;
        p.mask_ne3 = 1;
        p.mask_nb1 = 0;
        p.mask_nb2 = 0;
        p.mask_nb3 = 0;
    }

    p.rk2 = q->ne[2] / k->ne[2];
    p.rk3 = q->ne[3] / k->ne[3];
    p.rv2 = q->ne[2] / v->ne[2];
    p.rv3 = q->ne[3] / v->ne[3];

    p.n_head_log2 = n_head_log2;
    p.scale = scale;
    p.max_bias = max_bias;
    p.m0 = std::pow(2.0f, -(max_bias) / n_head_log2);
    p.m1 = std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2);

    sycl::queue & queue = ctx.stream();

    if (q->type == GGML_TYPE_F16) {
        if (m == nullptr) {
            ggml_sycl_flash_attn_ext_launch<sycl::half, float, false>(queue, p);
        } else if (m->type == GGML_TYPE_F16) {
            ggml_sycl_flash_attn_ext_launch<sycl::half, sycl::half, true>(queue, p);
        } else {
            ggml_sycl_flash_attn_ext_launch<sycl::half, float, true>(queue, p);
        }
    } else {
        if (m == nullptr) {
            ggml_sycl_flash_attn_ext_launch<float, float, false>(queue, p);
        } else if (m->type == GGML_TYPE_F16) {
            ggml_sycl_flash_attn_ext_launch<float, sycl::half, true>(queue, p);
        } else {
            ggml_sycl_flash_attn_ext_launch<float, float, true>(queue, p);
        }
    }
}

bool ggml_sycl_supports_flash_attn_ext(const ggml_tensor * op) {
    if (op->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * m = op->src[3];
    const ggml_tensor * s = op->src[4];

    if (q == nullptr || k == nullptr || v == nullptr) {
        return false;
    }

    if (!ggml_sycl_flash_attn_ext_supports_type(q->type) || op->type != GGML_TYPE_F32) {
        return false;
    }

    if (k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false;
    }

    if (s != nullptr) {
        return false;
    }

    if (m != nullptr && m->type != GGML_TYPE_F16 && m->type != GGML_TYPE_F32) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) op->op_params + 1, sizeof(max_bias));
    memcpy(&logit_softcap, (const float *) op->op_params + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        return false;
    }

    if (ggml_flash_attn_ext_get_prec(op) != GGML_PREC_DEFAULT && ggml_flash_attn_ext_get_prec(op) != GGML_PREC_F32) {
        return false;
    }

    if (ggml_get_op_params_i32(op, 4) != 0) {
        return false;
    }

    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(op)) {
        return false;
    }

    if (q->nb[0] != ggml_type_size(q->type) || op->nb[0] != sizeof(float)) {
        return false;
    }

    if (k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t)) {
        return false;
    }

    if (m != nullptr && m->nb[0] != ggml_type_size(m->type)) {
        return false;
    }

    if (q->ne[0] != k->ne[0] || v->ne[0] != op->ne[0]) {
        return false;
    }

    if (!ggml_sycl_flash_attn_ext_supports_head_dim(q->ne[0]) || !ggml_sycl_flash_attn_ext_supports_head_dim(v->ne[0])) {
        return false;
    }

    if (q->ne[0] != v->ne[0]) {
        return false;
    }

    if (k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2] || k->ne[3] != v->ne[3]) {
        return false;
    }

    if (q->ne[2] % k->ne[2] != 0 || q->ne[3] % k->ne[3] != 0 || q->ne[2] % v->ne[2] != 0 || q->ne[3] % v->ne[3] != 0) {
        return false;
    }

    if (op->ne[1] != q->ne[2] || op->ne[2] != q->ne[1] || op->ne[3] != q->ne[3]) {
        return false;
    }

    if (max_bias > 0.0f && m == nullptr) {
        return false;
    }

    if (m != nullptr) {
        if (!ggml_is_contiguous(m) || m->ne[0] != k->ne[1] || m->ne[1] < q->ne[1]) {
            return false;
        }
    }

    return true;
}
