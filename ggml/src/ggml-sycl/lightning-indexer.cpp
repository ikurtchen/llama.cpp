#include "lightning-indexer.hpp"

#include "dequantize.hpp"

#include <sycl/sycl.hpp>

namespace {

constexpr int LIGHTNING_EMBD = 128;
constexpr int LIGHTNING_SUBGROUP_SIZE = 16;
constexpr int LIGHTNING_VALUES_PER_LANE = LIGHTNING_EMBD / LIGHTNING_SUBGROUP_SIZE;
constexpr int LIGHTNING_KV_PER_BLOCK = 8;
constexpr int LIGHTNING_BLOCK_SIZE = LIGHTNING_SUBGROUP_SIZE * LIGHTNING_KV_PER_BLOCK;

template <int TYPE_K>
inline float load_lightning_k_value(const char * row, int idx) {
    if constexpr (TYPE_K == GGML_TYPE_F32) {
        return ((const float *) row)[idx];
    } else if constexpr (TYPE_K == GGML_TYPE_F16) {
        return (float) ((const sycl::half *) row)[idx];
    } else if constexpr (TYPE_K == GGML_TYPE_BF16) {
        return GGML_BF16_TO_FP32(((const ggml_bf16_t *) row)[idx]);
    } else if constexpr (TYPE_K == GGML_TYPE_Q8_0) {
        return ggml_sycl_dequantize_q8_0_at<float>((const block_q8_0 *) row, idx);
    } else {
        return 0.0f;
    }
}

template <int N_HEAD, int TYPE_K>
class lightning_indexer_kernel;

template <int N_HEAD, int TYPE_K>
void lightning_indexer_dispatch(
        sycl::queue & q,
        const float * q_d, const char * k_d, const float * w_d, const sycl::half * m_d, float * dst_d,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        int64_t nem3) {
    const int64_t num_kv_blocks = (n_kv + LIGHTNING_KV_PER_BLOCK - 1) / LIGHTNING_KV_PER_BLOCK;
    const size_t total_groups = (size_t) n_stream * (size_t) n_batch * (size_t) num_kv_blocks;
    const size_t global_size = total_groups * LIGHTNING_BLOCK_SIZE;

    q.parallel_for<lightning_indexer_kernel<N_HEAD, TYPE_K>>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(LIGHTNING_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(LIGHTNING_SUBGROUP_SIZE)]] {
        const size_t group_id = item.get_group(0);
        const int64_t kv_block = (int64_t) (group_id % (size_t) num_kv_blocks);
        const int64_t tmp = (int64_t) (group_id / (size_t) num_kv_blocks);
        const int64_t i_batch = tmp % n_batch;
        const int64_t i_stream = tmp / n_batch;

        const sycl::sub_group sg = item.get_sub_group();
        const int lane = (int) sg.get_local_id()[0];
        const int subgroup_id = (int) item.get_local_linear_id() / LIGHTNING_SUBGROUP_SIZE;
        const int64_t i_kv = kv_block * LIGHTNING_KV_PER_BLOCK + subgroup_id;

        if (i_kv >= n_kv) {
            return;
        }

        const char * q_base = (const char *) q_d + i_batch * nbq2 + i_stream * nbq3;
        const char * k_row = k_d + i_kv * nbk2 + i_stream * nbk3;
        const float * w_base = (const float *) ((const char *) w_d + i_batch * nbw1 + i_stream * nbw3);
        const sycl::half * m_base = (const sycl::half *) ((const char *) m_d + i_batch * nbm1 + (i_stream % nem3) * nbm3);
        float * dst_base = (float *) ((char *) dst_d + i_batch * nb1 + i_stream * nb3);

        float k_reg[LIGHTNING_VALUES_PER_LANE];
#pragma unroll
        for (int j = 0; j < LIGHTNING_VALUES_PER_LANE; ++j) {
            const int embd = lane + j * LIGHTNING_SUBGROUP_SIZE;
            k_reg[j] = load_lightning_k_value<TYPE_K>(k_row, embd);
        }

        float score = 0.0f;
        for (int head = 0; head < N_HEAD; ++head) {
            const float * q_head = (const float *) (q_base + head * nbq1);
            float partial = 0.0f;
#pragma unroll
            for (int j = 0; j < LIGHTNING_VALUES_PER_LANE; ++j) {
                const int embd = lane + j * LIGHTNING_SUBGROUP_SIZE;
                partial += q_head[embd] * k_reg[j];
            }

            const float sum = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
            if (lane == 0) {
                score += (sum > 0.0f ? sum : 0.0f) * w_base[head];
            }
        }

        if (lane == 0) {
            dst_base[i_kv] = score + (float) m_base[i_kv];
        }
    });
}

template <int N_HEAD>
void lightning_indexer_dispatch_type(
        sycl::queue & q, const ggml_tensor * k,
        const float * q_d, const char * k_d, const float * w_d, const sycl::half * m_d, float * dst_d,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        int64_t nem3) {
    switch (k->type) {
        case GGML_TYPE_F32:
            lightning_indexer_dispatch<N_HEAD, GGML_TYPE_F32>(q, q_d, k_d, w_d, m_d, dst_d, n_stream, n_batch, n_kv, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
            break;
        case GGML_TYPE_F16:
            lightning_indexer_dispatch<N_HEAD, GGML_TYPE_F16>(q, q_d, k_d, w_d, m_d, dst_d, n_stream, n_batch, n_kv, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
            break;
        case GGML_TYPE_BF16:
            lightning_indexer_dispatch<N_HEAD, GGML_TYPE_BF16>(q, q_d, k_d, w_d, m_d, dst_d, n_stream, n_batch, n_kv, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
            break;
        case GGML_TYPE_Q8_0:
            lightning_indexer_dispatch<N_HEAD, GGML_TYPE_Q8_0>(q, q_d, k_d, w_d, m_d, dst_d, n_stream, n_batch, n_kv, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
            break;
        default:
            GGML_ABORT("%s: unsupported type %s", __func__, ggml_type_name(k->type));
    }
}

} // namespace

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(size_t, nbw, w, nb)
    GGML_TENSOR_LOCALS(size_t, nbm, m, nb)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int64_t n_embd = q->ne[0];
    const int64_t n_head = q->ne[1];
    const int64_t n_batch = q->ne[2];
    const int64_t n_stream = q->ne[3];
    const int64_t n_kv = k->ne[2];

    GGML_ASSERT(n_embd == LIGHTNING_EMBD);
    GGML_ASSERT(k->ne[0] == n_embd);
    GGML_ASSERT(k->ne[1] == 1);
    GGML_ASSERT(k->ne[3] == n_stream);
    GGML_ASSERT(w->ne[0] == n_head);
    GGML_ASSERT(w->ne[1] == n_batch);
    GGML_ASSERT(w->ne[3] == n_stream);
    GGML_ASSERT(m->ne[0] == n_kv);
    GGML_ASSERT(m->ne[1] == n_batch);
    GGML_ASSERT(n_stream % m->ne[3] == 0);

    const float * q_d = (const float *) q->data;
    const char * k_d = (const char *) k->data;
    const float * w_d = (const float *) w->data;
    const sycl::half * m_d = (const sycl::half *) m->data;
    float * dst_d = (float *) dst->data;

    sycl::queue & queue = ctx.stream();

    if (n_head == 64) {
        lightning_indexer_dispatch_type<64>(queue, k, q_d, k_d, w_d, m_d, dst_d, n_stream, n_batch, n_kv, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, m->ne[3]);
    } else if (n_head == 32) {
        lightning_indexer_dispatch_type<32>(queue, k, q_d, k_d, w_d, m_d, dst_d, n_stream, n_batch, n_kv, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, m->ne[3]);
    } else {
        GGML_ABORT("%s: unsupported n_head=%lld", __func__, (long long) n_head);
    }
}

bool ggml_sycl_supports_lightning_indexer(const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * w = op->src[2];
    const ggml_tensor * m = op->src[3];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)

    if (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || m->type != GGML_TYPE_F16) {
        return false;
    }

    if (neq0 != LIGHTNING_EMBD) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32) {
        return false;
    }

    if (k->ne[0] != LIGHTNING_EMBD || k->ne[1] != 1 || k->ne[2] != op->ne[0] || k->ne[3] != q->ne[3]) {
        return false;
    }

    if (w->ne[0] != q->ne[1] || w->ne[1] != q->ne[2] || w->ne[3] != q->ne[3]) {
        return false;
    }

    if (m->ne[0] != op->ne[0] || m->ne[1] != q->ne[2] || m->ne[3] == 0 || q->ne[3] % m->ne[3] != 0) {
        return false;
    }

    for (const ggml_tensor * t : { q, k }) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch (k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}
