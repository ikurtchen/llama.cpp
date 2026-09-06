#include "gated_delta_net.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>

namespace {

template <int S_V, bool KDA>
static void gated_delta_net_f32(
        sycl::queue & q,
        const float * q_d,
        const float * k_d,
        const float * v_d,
        const float * g_d,
        const float * beta_d,
        const float * s_d,
        float * dst_d,
        int64_t H,
        int64_t n_tokens,
        int64_t n_seqs,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sv1,
        int64_t sv2,
        int64_t sv3,
        int64_t sb1,
        int64_t sb2,
        int64_t sb3,
        int64_t neqk1,
        int64_t rq3,
        float scale,
        int K) {
    const int64_t attn_stride = (int64_t) S_V * H;
    const int64_t state_slot_stride = (int64_t) S_V * S_V * H * n_seqs;
    const int64_t attn_elems = attn_stride * n_tokens * n_seqs;
    float * state_out_base = dst_d + attn_elems;

    q.parallel_for(
            sycl::nd_range<3>(sycl::range<3>((size_t) n_seqs, (size_t) H, (size_t) S_V * (size_t) S_V), sycl::range<3>(1, 1, (size_t) S_V)),
            [=](sycl::nd_item<3> item) {
                const int seq = (int) item.get_group(0);
                const int head = (int) item.get_group(1);
                const int col = (int) item.get_group(2);
                const int row = (int) item.get_local_id(2);
                const auto group = item.get_group();

                const int iq1 = head % (int) neqk1;
                const int iq3 = seq / (int) rq3;
                const int64_t state_head_base = ((int64_t) seq * H + head) * S_V * S_V + (int64_t) col * S_V;
                const int64_t attn_head_base = ((int64_t) seq * n_tokens * H + head) * S_V + col;

                float state_val = s_d[state_head_base + row];

                for (int64_t t = 0; t < n_tokens; ++t) {
                    const float * q_t = q_d + iq3 * sq3 + t * sq2 + iq1 * sq1;
                    const float * k_t = k_d + iq3 * sq3 + t * sq2 + iq1 * sq1;
                    const float * v_t = v_d + seq * sv3 + t * sv2 + head * sv1;

                    const int64_t gb_offset = seq * sb3 + t * sb2 + head * sb1;
                    const float * beta_t = beta_d + gb_offset;
                    const float * g_t = g_d + gb_offset * (KDA ? S_V : 1);

                    const float beta_val = *beta_t;
                    const float k_val = k_t[row];
                    const float q_val = q_t[row];
                    const float decay = KDA ? sycl::exp(g_t[row]) : sycl::exp(*g_t);

                    const float kv_partial = decay * state_val * k_val;
                    const float kv_col = sycl::reduce_over_group(group, kv_partial, sycl::plus<float>());
                    const float delta = (v_t[col] - kv_col) * beta_val;

                    state_val = decay * state_val + k_val * delta;

                    const float attn_partial = state_val * q_val;
                    const float attn_col = sycl::reduce_over_group(group, attn_partial, sycl::plus<float>());

                    if (row == 0) {
                        dst_d[attn_head_base + t * attn_stride] = attn_col * scale;
                    }

                    if (K > 1) {
                        const int target_slot = (int) n_tokens - 1 - (int) t;
                        if (target_slot >= 0 && target_slot < K) {
                            float * state_slot = state_out_base + (int64_t) target_slot * state_slot_stride;
                            state_slot[state_head_base + row] = state_val;
                        }
                    }
                }

                if (K == 1) {
                    state_out_base[state_head_base + row] = state_val;
                }
            });
}

} // namespace

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q = dst->src[0];
    ggml_tensor * src_k = dst->src[1];
    ggml_tensor * src_v = dst->src[2];
    ggml_tensor * src_g = dst->src[3];
    ggml_tensor * src_beta = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t, nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t, nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, src_beta, nb);

    const int64_t S_v = nev0;
    const int64_t H = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs = nev3;

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;
    const int64_t rq3 = nev3 / neq3;
    const bool kda = (src_g->ne[0] == S_v);
    const int K = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(src_q->type == GGML_TYPE_F32);
    GGML_ASSERT(src_k->type == GGML_TYPE_F32);
    GGML_ASSERT(src_v->type == GGML_TYPE_F32);
    GGML_ASSERT(src_g->type == GGML_TYPE_F32);
    GGML_ASSERT(src_beta->type == GGML_TYPE_F32);
    GGML_ASSERT(src_state->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    const int64_t sq1 = (int64_t) nbq1 / (int64_t) sizeof(float);
    const int64_t sq2 = (int64_t) nbq2 / (int64_t) sizeof(float);
    const int64_t sq3 = (int64_t) nbq3 / (int64_t) sizeof(float);
    const int64_t sv1 = (int64_t) nbv1 / (int64_t) sizeof(float);
    const int64_t sv2 = (int64_t) nbv2 / (int64_t) sizeof(float);
    const int64_t sv3 = (int64_t) nbv3 / (int64_t) sizeof(float);
    const int64_t sb1 = (int64_t) nbb1 / (int64_t) sizeof(float);
    const int64_t sb2 = (int64_t) nbb2 / (int64_t) sizeof(float);
    const int64_t sb3 = (int64_t) nbb3 / (int64_t) sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);
    sycl::queue & q = ctx.stream();

    if (kda) {
        switch (S_v) {
            case 16: gated_delta_net_f32<16, true>(q, (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
                                                   (const float *) src_g->data, (const float *) src_beta->data, (const float *) src_state->data,
                                                   (float *) dst->data, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                                                   neqk1, rq3, scale, K); break;
            case 32: gated_delta_net_f32<32, true>(q, (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
                                                   (const float *) src_g->data, (const float *) src_beta->data, (const float *) src_state->data,
                                                   (float *) dst->data, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                                                   neqk1, rq3, scale, K); break;
            case 64: gated_delta_net_f32<64, true>(q, (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
                                                   (const float *) src_g->data, (const float *) src_beta->data, (const float *) src_state->data,
                                                   (float *) dst->data, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                                                   neqk1, rq3, scale, K); break;
            case 128: gated_delta_net_f32<128, true>(q, (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
                                                     (const float *) src_g->data, (const float *) src_beta->data, (const float *) src_state->data,
                                                     (float *) dst->data, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                                                     neqk1, rq3, scale, K); break;
            default: GGML_ABORT("gated delta net requires head_size 16, 32, 64, or 128");
        }
    } else {
        switch (S_v) {
            case 16: gated_delta_net_f32<16, false>(q, (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
                                                    (const float *) src_g->data, (const float *) src_beta->data, (const float *) src_state->data,
                                                    (float *) dst->data, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                                                    neqk1, rq3, scale, K); break;
            case 32: gated_delta_net_f32<32, false>(q, (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
                                                    (const float *) src_g->data, (const float *) src_beta->data, (const float *) src_state->data,
                                                    (float *) dst->data, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                                                    neqk1, rq3, scale, K); break;
            case 64: gated_delta_net_f32<64, false>(q, (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
                                                    (const float *) src_g->data, (const float *) src_beta->data, (const float *) src_state->data,
                                                    (float *) dst->data, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                                                    neqk1, rq3, scale, K); break;
            case 128: gated_delta_net_f32<128, false>(q, (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
                                                      (const float *) src_g->data, (const float *) src_beta->data, (const float *) src_state->data,
                                                      (float *) dst->data, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                                                      neqk1, rq3, scale, K); break;
            default: GGML_ABORT("gated delta net requires head_size 16, 32, 64, or 128");
        }
    }
}

bool ggml_sycl_supports_gated_delta_net(const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * g = op->src[3];
    const ggml_tensor * beta = op->src[4];
    const ggml_tensor * state = op->src[5];
    if (!q || !k || !v || !g || !beta || !state) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32 ||
        beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t S_v = v->ne[0];
    return (S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128) && ggml_is_contiguous_rows(q) && ggml_is_contiguous_rows(k) &&
           ggml_is_contiguous_rows(v) && ggml_are_same_stride(q, k) && ggml_is_contiguous(g) && ggml_is_contiguous(beta) &&
           ggml_is_contiguous(state);
}
