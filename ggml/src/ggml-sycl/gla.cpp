#include "gla.hpp"

template <int head_size>
static void gated_linear_attn_f32_sycl(
        const int B, const int T, const int C, const int H, const float scale,
        const float * k, const float * v, const float * r,
        const float * td, const float * s,
        float * dst,
        sycl::queue & stream) {

    const sycl::range<1> global(B * H * head_size);
    const sycl::range<1> local(head_size);

    stream.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> _k(sycl::range<1>(head_size), cgh);
        sycl::local_accessor<float, 1> _r(sycl::range<1>(head_size), cgh);
        sycl::local_accessor<float, 1> _td(sycl::range<1>(head_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) {
                const int tid = item.get_local_id(0);
                const int bid = item.get_group(0);

                const int batch_i = bid / H;
                const int head_i  = bid % H;
                const int state_size    = C * head_size;
                const int n_seq_tokens  = T / B;

                float state[head_size];

#pragma unroll
                for (int i = 0; i < head_size; i++) {
                    state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
                }

                for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
                     t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
                     t += C) {
                    sycl::group_barrier(item.get_group());
                    _k[tid]  = k[t];
                    _r[tid]  = r[t];
                    _td[tid] = td[t];
                    sycl::group_barrier(item.get_group());

                    const float _v = v[t];
                    float y = 0;
                    for (int j = 0; j < head_size; j += 4) {
                        float kv0 = _k[j + 0] * _v;
                        float kv1 = _k[j + 1] * _v;
                        float kv2 = _k[j + 2] * _v;
                        float kv3 = _k[j + 3] * _v;

                        state[j + 0] = state[j + 0] * _td[j + 0] + kv0;
                        state[j + 1] = state[j + 1] * _td[j + 1] + kv1;
                        state[j + 2] = state[j + 2] * _td[j + 2] + kv2;
                        state[j + 3] = state[j + 3] * _td[j + 3] + kv3;

                        y += _r[j + 0] * state[j + 0];
                        y += _r[j + 1] * state[j + 1];
                        y += _r[j + 2] * state[j + 2];
                        y += _r[j + 3] * state[j + 3];
                    }
                    dst[t] = y * scale;
                }

#pragma unroll
                for (int i = 0; i < head_size; i++) {
                    dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
                }
            }
        );
    });
}

void ggml_sycl_op_gated_linear_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const float * k_d  = (const float *) dst->src[0]->data;
    const float * v_d  = (const float *) dst->src[1]->data;
    const float * r_d  = (const float *) dst->src[2]->data;
    const float * td_d = (const float *) dst->src[3]->data;
    const float * s_d  = (const float *) dst->src[4]->data;

    const int64_t B = dst->src[4]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float scale;
    memcpy(&scale, (float *) dst->op_params, sizeof(float));

    float * dst_d = (float *) dst->data;

    sycl::queue & stream = *ctx.stream();

    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);

    if (C / H == 64) {
        gated_linear_attn_f32_sycl<64>(B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d, stream);
    } else {
        gated_linear_attn_f32_sycl<128>(B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d, stream);
    }
}
