#include "wkv.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace {

template <int HEAD_SIZE>
static void rwkv_wkv6_f32(
        sycl::queue & q,
        int64_t B,
        int64_t T,
        int64_t C,
        int64_t H,
        const float * k,
        const float * v,
        const float * r,
        const float * tf,
        const float * td,
        const float * s,
        float * dst) {
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> k_local(sycl::range<1>(HEAD_SIZE), cgh);
        sycl::local_accessor<float, 1> r_local(sycl::range<1>(HEAD_SIZE), cgh);
        sycl::local_accessor<float, 1> tf_local(sycl::range<1>(HEAD_SIZE), cgh);
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

                    sycl::group_barrier(item.get_group());
                    tf_local[tid] = tf[head_i * HEAD_SIZE + tid];
                    sycl::group_barrier(item.get_group());

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
                            y += r_local[j] * (tf_local[j] * kv + state[j]);
                            state[j] = state[j] * td_local[j] + kv;
                        }
                        dst[t] = y;
                    }

                    for (int i = 0; i < HEAD_SIZE; ++i) {
                        dst[T * C + state_head_base + i * HEAD_SIZE + tid] = state[i];
                    }
                });
    });
}

template <int HEAD_SIZE>
static void rwkv_wkv7_f32(
        sycl::queue & q,
        int64_t B,
        int64_t T,
        int64_t C,
        int64_t H,
        const float * r,
        const float * w,
        const float * k,
        const float * v,
        const float * a,
        const float * b,
        const float * s,
        float * dst) {
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> r_local(sycl::range<1>(HEAD_SIZE), cgh);
        sycl::local_accessor<float, 1> w_local(sycl::range<1>(HEAD_SIZE), cgh);
        sycl::local_accessor<float, 1> k_local(sycl::range<1>(HEAD_SIZE), cgh);
        sycl::local_accessor<float, 1> a_local(sycl::range<1>(HEAD_SIZE), cgh);
        sycl::local_accessor<float, 1> b_local(sycl::range<1>(HEAD_SIZE), cgh);

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
                        state[i] = s[state_head_base + tid * HEAD_SIZE + i];
                    }

                    for (int t = batch_i * n_seq_tokens * (int) C + head_i * HEAD_SIZE + tid;
                         t < (batch_i + 1) * n_seq_tokens * (int) C + head_i * HEAD_SIZE + tid;
                         t += (int) C) {
                        sycl::group_barrier(item.get_group());
                        r_local[tid] = r[t];
                        w_local[tid] = w[t];
                        k_local[tid] = k[t];
                        a_local[tid] = a[t];
                        b_local[tid] = b[t];
                        sycl::group_barrier(item.get_group());

                        float sa = 0.0f;
                        for (int j = 0; j < HEAD_SIZE; ++j) {
                            sa += a_local[j] * state[j];
                        }

                        const float v_val = v[t];
                        float y = 0.0f;
                        for (int j = 0; j < HEAD_SIZE; ++j) {
                            const float kv = k_local[j] * v_val;
                            state[j] = state[j] * w_local[j] + kv + sa * b_local[j];
                            y += state[j] * r_local[j];
                        }
                        dst[t] = y;
                    }

                    for (int i = 0; i < HEAD_SIZE; ++i) {
                        dst[T * C + state_head_base + tid * HEAD_SIZE + i] = state[i];
                    }
                });
    });
}

static bool rwkv_common_ok(const ggml_tensor * op, int state_src, int data_src) {
    const ggml_tensor * data = op->src[data_src];
    const ggml_tensor * state = op->src[state_src];
    if (!data || !state) {
        return false;
    }
    if (data->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (data->ne[1] == 0 || op->ne[0] % data->ne[1] != 0) {
        return false;
    }
    const int64_t head_size = op->ne[0] / data->ne[1];
    return head_size == 64 || head_size == 128;
}

} // namespace

void ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const float * k_d  = static_cast<const float *>(dst->src[0]->data);
    const float * v_d  = static_cast<const float *>(dst->src[1]->data);
    const float * r_d  = static_cast<const float *>(dst->src[2]->data);
    const float * tf_d = static_cast<const float *>(dst->src[3]->data);
    const float * td_d = static_cast<const float *>(dst->src[4]->data);
    const float * s_d  = static_cast<const float *>(dst->src[5]->data);

    const int64_t B = dst->src[5]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];
    const int64_t head_size = C / H;

    sycl::queue & q = ctx.stream();
    if (head_size == 64) {
        rwkv_wkv6_f32<64>(q, B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, static_cast<float *>(dst->data));
    } else if (head_size == 128) {
        rwkv_wkv6_f32<128>(q, B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, static_cast<float *>(dst->data));
    } else {
        GGML_ABORT("rwkv wkv6 requires head_size 64 or 128");
    }
}

void ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const float * r_d = static_cast<const float *>(dst->src[0]->data);
    const float * w_d = static_cast<const float *>(dst->src[1]->data);
    const float * k_d = static_cast<const float *>(dst->src[2]->data);
    const float * v_d = static_cast<const float *>(dst->src[3]->data);
    const float * a_d = static_cast<const float *>(dst->src[4]->data);
    const float * b_d = static_cast<const float *>(dst->src[5]->data);
    const float * s_d = static_cast<const float *>(dst->src[6]->data);

    const int64_t B = dst->src[6]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];
    const int64_t head_size = C / H;

    sycl::queue & q = ctx.stream();
    if (head_size == 64) {
        rwkv_wkv7_f32<64>(q, B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, static_cast<float *>(dst->data));
    } else if (head_size == 128) {
        rwkv_wkv7_f32<128>(q, B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, static_cast<float *>(dst->data));
    } else {
        GGML_ABORT("rwkv wkv7 requires head_size 64 or 128");
    }
}

bool ggml_sycl_supports_rwkv_wkv6(const ggml_tensor * op) {
    return rwkv_common_ok(op, 5, 0);
}

bool ggml_sycl_supports_rwkv_wkv7(const ggml_tensor * op) {
    return rwkv_common_ok(op, 6, 0);
}
