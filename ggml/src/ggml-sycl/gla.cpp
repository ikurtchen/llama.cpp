#include "gla.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstring>
#include <cstdint>

namespace {

template <int HEAD_SIZE>
static void gated_linear_attn_f32(
        sycl::queue & q,
        int64_t B,
        int64_t T,
        int64_t C,
        int64_t H,
        float scale,
        const float * k,
        const float * v,
        const float * r,
        const float * td,
        const float * s,
        float * dst) {
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> k_local(sycl::range<1>(HEAD_SIZE), cgh);
        sycl::local_accessor<float, 1> r_local(sycl::range<1>(HEAD_SIZE), cgh);
        sycl::local_accessor<float, 1> td_local(sycl::range<1>(HEAD_SIZE), cgh);

        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t) B * (size_t) H * HEAD_SIZE), sycl::range<1>(HEAD_SIZE)),
                [=](sycl::nd_item<1> item) {
                    const int tid = (int) item.get_local_id(0);
                    const int bid = (int) item.get_group(0);
                    const int batch_i = bid / (int) H;
                    const int head_i = bid % (int) H;
                    const int state_size = (int) C * HEAD_SIZE;
                    const int n_seq_tokens = (int) (T / B);
                    const int state_head_base = batch_i * state_size + head_i * HEAD_SIZE * HEAD_SIZE;

                    float state[HEAD_SIZE];
                    for (int i = 0; i < HEAD_SIZE; ++i) {
                        state[i] = s[state_head_base + i * HEAD_SIZE + tid];
                    }

                    for (int t = batch_i * n_seq_tokens * (int) C + head_i * HEAD_SIZE + tid;
                         t < (batch_i + 1) * n_seq_tokens * (int) C + head_i * HEAD_SIZE + tid;
                         t += (int) C) {
                        sycl::group_barrier(item.get_group());
                        k_local[tid] = k[t];
                        r_local[tid] = r[t];
                        td_local[tid] = td[t];
                        sycl::group_barrier(item.get_group());

                        const float v_val = v[t];
                        float y = 0.0f;
                        for (int j = 0; j < HEAD_SIZE; ++j) {
                            const float kv = k_local[j] * v_val;
                            state[j] = state[j] * td_local[j] + kv;
                            y += r_local[j] * state[j];
                        }
                        dst[t] = y * scale;
                    }

                    for (int i = 0; i < HEAD_SIZE; ++i) {
                        dst[T * C + state_head_base + i * HEAD_SIZE + tid] = state[i];
                    }
                });
    });
}

} // namespace

void ggml_sycl_op_gated_linear_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const float * k_d  = static_cast<const float *>(dst->src[0]->data);
    const float * v_d  = static_cast<const float *>(dst->src[1]->data);
    const float * r_d  = static_cast<const float *>(dst->src[2]->data);
    const float * td_d = static_cast<const float *>(dst->src[3]->data);
    const float * s_d  = static_cast<const float *>(dst->src[4]->data);

    const int64_t B = dst->src[4]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    float * dst_d = static_cast<float *>(dst->data);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[2]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[3]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);

    sycl::queue & q = ctx.stream();
    const int64_t head_size = C / H;
    if (head_size == 64) {
        gated_linear_attn_f32<64>(q, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } else if (head_size == 128) {
        gated_linear_attn_f32<128>(q, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } else {
        GGML_ABORT("gated linear attn requires head_size 64 or 128");
    }
}

bool ggml_sycl_supports_gated_linear_attn(const ggml_tensor * op) {
    const ggml_tensor * k  = op->src[0];
    const ggml_tensor * v  = op->src[1];
    const ggml_tensor * r  = op->src[2];
    const ggml_tensor * td = op->src[3];
    const ggml_tensor * s  = op->src[4];
    if (!k || !v || !r || !td || !s) {
        return false;
    }
    if (k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 || r->type != GGML_TYPE_F32 || td->type != GGML_TYPE_F32 ||
        s->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (k->ne[1] == 0 || op->ne[0] % k->ne[1] != 0) {
        return false;
    }
    const int64_t head_size = op->ne[0] / k->ne[1];
    return head_size == 64 || head_size == 128;
}
