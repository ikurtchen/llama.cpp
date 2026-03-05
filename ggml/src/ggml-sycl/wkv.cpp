#include "wkv.hpp"
#include "presets.hpp"

template <int head_size>
static void rwkv_wkv6_f32_sycl(
        const int B, const int T, const int C, const int H,
        const float * k, const float * v, const float * r,
        const float * tf, const float * td, const float * s,
        float * dst,
        sycl::queue & stream) {

    const sycl::range<1> global(B * H * head_size);
    const sycl::range<1> local(head_size);

    stream.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> _k(sycl::range<1>(head_size), cgh);
        sycl::local_accessor<float, 1> _r(sycl::range<1>(head_size), cgh);
        sycl::local_accessor<float, 1> _tf(sycl::range<1>(head_size), cgh);
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

                for (int i = 0; i < head_size; i++) {
                    state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
                }

                sycl::group_barrier(item.get_group());
                _tf[tid] = tf[head_i * head_size + tid];
                sycl::group_barrier(item.get_group());

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

                        y += _r[j + 0] * (_tf[j + 0] * kv0 + state[j + 0]);
                        y += _r[j + 1] * (_tf[j + 1] * kv1 + state[j + 1]);
                        y += _r[j + 2] * (_tf[j + 2] * kv2 + state[j + 2]);
                        y += _r[j + 3] * (_tf[j + 3] * kv3 + state[j + 3]);

                        state[j + 0] = state[j + 0] * _td[j + 0] + kv0;
                        state[j + 1] = state[j + 1] * _td[j + 1] + kv1;
                        state[j + 2] = state[j + 2] * _td[j + 2] + kv2;
                        state[j + 3] = state[j + 3] * _td[j + 3] + kv3;
                    }
                    dst[t] = y;
                }

                for (int i = 0; i < head_size; i++) {
                    dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
                }
            }
        );
    });
}

template <int head_size>
static void rwkv_wkv7_f32_sycl(
        const int B, const int T, const int C, const int H,
        const float * r, const float * w, const float * k,
        const float * v, const float * a, const float * b,
        const float * s,
        float * dst,
        sycl::queue & stream) {

    const sycl::range<1> global(B * H * head_size);
    const sycl::range<1> local(head_size);

    stream.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> _r(sycl::range<1>(head_size), cgh);
        sycl::local_accessor<float, 1> _w(sycl::range<1>(head_size), cgh);
        sycl::local_accessor<float, 1> _k(sycl::range<1>(head_size), cgh);
        sycl::local_accessor<float, 1> _a(sycl::range<1>(head_size), cgh);
        sycl::local_accessor<float, 1> _b(sycl::range<1>(head_size), cgh);

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

                for (int i = 0; i < head_size; i++) {
                    state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
                }

                for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
                     t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
                     t += C) {
                    sycl::group_barrier(item.get_group());
                    _r[tid] = r[t];
                    _w[tid] = w[t];
                    _k[tid] = k[t];
                    _a[tid] = a[t];
                    _b[tid] = b[t];
                    sycl::group_barrier(item.get_group());

                    float sa = 0;
                    for (int j = 0; j < head_size; j += 4) {
                        sa += _a[j + 0] * state[j + 0];
                        sa += _a[j + 1] * state[j + 1];
                        sa += _a[j + 2] * state[j + 2];
                        sa += _a[j + 3] * state[j + 3];
                    }

                    const float _v = v[t];
                    float y = 0;
                    for (int j = 0; j < head_size; j += 4) {
                        float kv0 = _k[j + 0] * _v;
                        float kv1 = _k[j + 1] * _v;
                        float kv2 = _k[j + 2] * _v;
                        float kv3 = _k[j + 3] * _v;

                        state[j + 0] = state[j + 0] * _w[j + 0] + kv0 + sa * _b[j + 0];
                        state[j + 1] = state[j + 1] * _w[j + 1] + kv1 + sa * _b[j + 1];
                        state[j + 2] = state[j + 2] * _w[j + 2] + kv2 + sa * _b[j + 2];
                        state[j + 3] = state[j + 3] * _w[j + 3] + kv3 + sa * _b[j + 3];

                        y += state[j + 0] * _r[j + 0];
                        y += state[j + 1] * _r[j + 1];
                        y += state[j + 2] * _r[j + 2];
                        y += state[j + 3] * _r[j + 3];
                    }
                    dst[t] = y;
                }

                for (int i = 0; i < head_size; i++) {
                    dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
                }
            }
        );
    });
}

void ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const float * k_d  = (const float *) dst->src[0]->data;
    const float * v_d  = (const float *) dst->src[1]->data;
    const float * r_d  = (const float *) dst->src[2]->data;
    const float * tf_d = (const float *) dst->src[3]->data;
    const float * td_d = (const float *) dst->src[4]->data;
    const float * s_d  = (const float *) dst->src[5]->data;

    const int64_t B = dst->src[5]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *) dst->data;

    sycl::queue & stream = *ctx.stream();

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == SYCL_WKV_BLOCK_SIZE || C / H == SYCL_WKV_BLOCK_SIZE * 2);

    if (C / H == SYCL_WKV_BLOCK_SIZE) {
        rwkv_wkv6_f32_sycl<SYCL_WKV_BLOCK_SIZE>(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d, stream);
    } else {
        rwkv_wkv6_f32_sycl<SYCL_WKV_BLOCK_SIZE * 2>(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d, stream);
    }
}

void ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const float * r_d = (const float *) dst->src[0]->data;
    const float * w_d = (const float *) dst->src[1]->data;
    const float * k_d = (const float *) dst->src[2]->data;
    const float * v_d = (const float *) dst->src[3]->data;
    const float * a_d = (const float *) dst->src[4]->data;
    const float * b_d = (const float *) dst->src[5]->data;
    const float * s_d = (const float *) dst->src[6]->data;

    const int64_t B = dst->src[6]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *) dst->data;

    sycl::queue & stream = *ctx.stream();

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == SYCL_WKV_BLOCK_SIZE || C / H == SYCL_WKV_BLOCK_SIZE * 2);

    if (C / H == SYCL_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32_sycl<SYCL_WKV_BLOCK_SIZE>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, stream);
    } else {
        rwkv_wkv7_f32_sycl<SYCL_WKV_BLOCK_SIZE * 2>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, stream);
    }
}
