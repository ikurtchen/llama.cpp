#include "lightning_indexer.hpp"
#include "common.hpp"
#include "dequantize.hpp"

constexpr int WARP_SIZE_K = 32;

// Optimization: use SYCL group collectives for warp reduction instead of manual shuffle loops.
// This replaces warp_reduce_sum with reduce_over_group for fewer barriers and better codegen.
template <int WARP_SIZE>
inline float warp_reduce_sum_group(const float val, const sycl::sub_group & sg) {
    (void)WARP_SIZE;
    return sycl::reduce_over_group(sg, val, sycl::plus<float>());
}

// Optimization: manual XOR butterfly reduction — profile vs reduce_over_group on B70
template <int WARP_SIZE>
inline float warp_reduce_sum_xor(float val, const sycl::sub_group & sg) {
    (void)sg;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += sycl::permute_group_by_xor(sg, val, offset);
    }
    return val;
}

// Dequantize 4 values from K at given element offset i0 into float4.
// Matches CUDA get_dequantize_V<type, float, 4>() calling convention.
template <ggml_type TYPE_K>
inline void dequantize_k_vec_4(const void * k_base, sycl::float4 * dst, int i0) {
    if constexpr (TYPE_K == GGML_TYPE_Q4_0) {
        constexpr int QK = QK4_0;
        const block_q4_0 * x = (const block_q4_0 *) k_base;

        const int ib  = i0 / QK;
        const int pos = i0 % QK;

        const float d = x[ib].d;
        if (pos < QK / 2) {
            dst->x() = d * ((float)((int8_t)((x[ib].qs[pos + 0] & 0xF) - 8)));
            dst->y() = d * ((float)((int8_t)((x[ib].qs[pos + 1] & 0xF) - 8)));
            dst->z() = d * ((float)((int8_t)((x[ib].qs[pos + 2] & 0xF) - 8)));
            dst->w() = d * ((float)((int8_t)((x[ib].qs[pos + 3] & 0xF) - 8)));
        } else {
            const int qs_off = pos - QK / 2;
            dst->x() = d * ((float)((int8_t)((x[ib].qs[qs_off + 0] >> 4) - 8)));
            dst->y() = d * ((float)((int8_t)((x[ib].qs[qs_off + 1] >> 4) - 8)));
            dst->z() = d * ((float)((int8_t)((x[ib].qs[qs_off + 2] >> 4) - 8)));
            dst->w() = d * ((float)((int8_t)((x[ib].qs[qs_off + 3] >> 4) - 8)));
        }
    } else if constexpr (TYPE_K == GGML_TYPE_Q4_1) {
        constexpr int QK = QK4_1;
        const block_q4_1 * x = (const block_q4_1 *) k_base;

        const int ib  = i0 / QK;
        const int iqs = i0 % (QK / 2);
        const int shift = (i0 % QK) / (QK / 2);

        int q;
        for (int l = 0; l < 4; ++l) {
            ((int8_t *)&q)[l] = x[ib].qs[iqs + l];
        }
        q >>= 4 * shift;
        q &= 0x0F0F0F0F;

        const sycl::half2 dm = *(const sycl::half2 *)&x[ib].dm;
        const float d = (float)dm[0];
        const float m = (float)dm[1];
        const int8_t * q8 = (const int8_t *)&q;
        dst->x() = d * (float)q8[0] + m;
        dst->y() = d * (float)q8[1] + m;
        dst->z() = d * (float)q8[2] + m;
        dst->w() = d * (float)q8[3] + m;
    } else if constexpr (TYPE_K == GGML_TYPE_Q5_0) {
        constexpr int QK = QK5_0;
        const block_q5_0 * x = (const block_q5_0 *) k_base;

        const int ib  = i0 / QK;
        const int iqs = i0 % (QK / 2);
        const int idq = i0 % QK;
        const int shift = (i0 % QK) / (QK / 2);

        int q;
        for (int l = 0; l < 4; ++l) {
            ((int8_t *)&q)[l] = x[ib].qs[iqs + l];
        }
        q >>= 4 * shift;
        q &= 0x0F0F0F0F;

        const uint32_t qh = *(const uint32_t *)x[ib].qh;
        for (int l = 0; l < 4; ++l) {
            q |= ((qh >> (idq + l)) & 1) << (8 * l + 4);
        }

        int sub = 0;
        for (int l = 0; l < 4; ++l) {
            int byte = (q >> (8 * l)) & 0xFF;
            byte = (byte - 16) & 0xFF;
            sub |= byte << (8 * l);
        }
        q = sub;

        const float d = (float)*(const sycl::half *)&x[ib].d;
        const int8_t * q8 = (const int8_t *)&q;
        dst->x() = d * (float)q8[0];
        dst->y() = d * (float)q8[1];
        dst->z() = d * (float)q8[2];
        dst->w() = d * (float)q8[3];
    } else if constexpr (TYPE_K == GGML_TYPE_Q5_1) {
        constexpr int QK = QK5_1;
        const block_q5_1 * x = (const block_q5_1 *) k_base;

        const int ib  = i0 / QK;
        const int iqs = i0 % (QK / 2);
        const int idq = i0 % QK;
        const int shift = (i0 % QK) / (QK / 2);

        int q;
        for (int l = 0; l < 4; ++l) {
            ((int8_t *)&q)[l] = x[ib].qs[iqs + l];
        }
        q >>= 4 * shift;
        q &= 0x0F0F0F0F;

        const uint32_t qh = *(const uint32_t *)x[ib].qh;
        for (int l = 0; l < 4; ++l) {
            q |= ((qh >> (idq + l)) & 1) << (8 * l + 4);
        }

        const sycl::half2 dm = *(const sycl::half2 *)&x[ib].dm;
        const float d = (float)dm[0];
        const float m = (float)dm[1];
        const int8_t * q8 = (const int8_t *)&q;
        dst->x() = d * (float)q8[0] + m;
        dst->y() = d * (float)q8[1] + m;
        dst->z() = d * (float)q8[2] + m;
        dst->w() = d * (float)q8[3] + m;
    } else if constexpr (TYPE_K == GGML_TYPE_Q8_0) {
        constexpr int QK = QK8_0;
        const block_q8_0 * x = (const block_q8_0 *) k_base;

        const int ib  = i0 / QK;
        const int iqs = i0 % QK;

        const float d = x[ib].d;
        dst->x() = (float)((int8_t)(x[ib].qs[iqs + 0])) * d;
        dst->y() = (float)((int8_t)(x[ib].qs[iqs + 1])) * d;
        dst->z() = (float)((int8_t)(x[ib].qs[iqs + 2])) * d;
        dst->w() = (float)((int8_t)(x[ib].qs[iqs + 3])) * d;
    } else if constexpr (TYPE_K == GGML_TYPE_F16) {
        const sycl::half * src = (const sycl::half *) k_base + i0;
        dst->x() = (float)src[0];
        dst->y() = (float)src[1];
        dst->z() = (float)src[2];
        dst->w() = (float)src[3];
    } else if constexpr (TYPE_K == GGML_TYPE_BF16) {
        const uint16_t * src = (const uint16_t *) k_base + i0;
        // bf16 to float: shift left 16 bits then reinterpret as float
        uint32_t w0 = (uint32_t)src[0] << 16;
        uint32_t w1 = (uint32_t)src[1] << 16;
        uint32_t w2 = (uint32_t)src[2] << 16;
        uint32_t w3 = (uint32_t)src[3] << 16;
        dst->x() = *(float *)&w0;
        dst->y() = *(float *)&w1;
        dst->z() = *(float *)&w2;
        dst->w() = *(float *)&w3;
    } else {
        // F32: direct copy
        static_assert(TYPE_K == GGML_TYPE_F32, "unsupported type");
        const float * src = (const float *) k_base + i0;
        dst->x() = src[0];
        dst->y() = src[1];
        dst->z() = src[2];
        dst->w() = src[3];
    }
}

// Vector-based lightning indexer kernel template.
// Each warp processes K_VECS_PER_WARP K-vectors. All warps in a block cooperate via shared memory.
// Uses sycl::reduce_over_group for warp-level reduction and [[intel::grf_size(256)]] large GRF
// to accommodate register pressure from k_reg_f[8] (32 floats) plus dequantize intermediates.
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int N_EMBD, int N_HEAD, ggml_type TYPE_K>
static void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const sycl::half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        int64_t nem3,
        float       * local_smem,
        const sycl::nd_item<3> & item) {

    (void)n_stream;
    (void)n_batch;
    (void)nb2;

    constexpr int K_VECS_PER_WARP   = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE_K;
    constexpr int N_HEAD_INNER      = N_HEAD / 4;
    constexpr int N_Q               = N_HEAD_INNER * (N_EMBD / 4);

    const int i_batch  = item.get_group(1);
    const int i_stream = item.get_group(2);
    const int i_warp   = item.get_local_id(1);
    const int i_lane   = item.get_local_id(2);
    const int tid      = i_warp * WARP_SIZE_K + i_lane;

    const int start_kv_block = item.get_group(0) * K_VECS_PER_BLOCK;
    const int start_kv       = start_kv_block + i_warp * K_VECS_PER_WARP;

    auto sg = item.get_sub_group();

    const char  * q_base = (const char  *)                 Q + i_batch * nbq2 + i_stream * nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch * nbw1 + i_stream * nbw3);

    float       * w_shared    = local_smem;
    sycl::float4 * q_shared_f = (sycl::float4 *)(local_smem + N_HEAD_INNER);
    float       * dst_shared  = local_smem + N_HEAD_INNER + N_Q * 4;

    // Phase 1: load K vectors into registers
    sycl::float4 k_reg_f[K_VECS_PER_WARP];
#pragma unroll
    for (int k = 0; k < K_VECS_PER_WARP; ++k) {
        int i_kv = start_kv + k;
        if (i_kv < n_kv) {
            const void * k_base = (const void *) ((const char *) K + i_kv * nbk2 + i_stream * nbk3);
            dequantize_k_vec_4<TYPE_K>(k_base, &k_reg_f[k], i_lane * 4);
        } else {
            k_reg_f[k] = sycl::float4{0, 0, 0, 0};
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

#pragma unroll
        for (int i_q = tid; i_q < N_Q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_q] = *(const sycl::float4 *) (q_base + i_head * nbq1 + i_embd * sizeof(sycl::float4));
        }

        item.barrier(sycl::access::fence_space::local_space);

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            const sycl::float4 q_vec = q_shared_f[i_head_inner * (N_EMBD / 4) + i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                qk[k] += q_vec.x() * k_reg_f[k].x();
                qk[k] += q_vec.y() * k_reg_f[k].y();
                qk[k] += q_vec.z() * k_reg_f[k].z();
                qk[k] += q_vec.w() * k_reg_f[k].w();
            }

            // OPTIMIZATION: use sycl::reduce_over_group instead of manual shuffle loop
#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum_group<WARP_SIZE_K>(qk[k], sg);
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const sycl::half * m_base = (const sycl::half *) ((const char *) M + i_batch * nbm1 + (i_stream % nem3) * nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch * nb1 + i_stream * nb3);
            dst_base[i_kv] = dst_shared[tid] + (float)m_base[i_kv];
        }
    }
}

// Macro to dispatch a typed kernel instantiation for the given N_EMBD/N_HEAD pair.
// OPTIMIZED: large GRF mode (256 registers) to eliminate spills from dequantize + reduce_over_group.
#define LIGHTNING_INDEXER_VEC_DISPATCH(n_embd, n_head, k_type)                                    \
    cgh.parallel_for(                                                                              \
        sycl::nd_range<3>(grid * block, block),                                                    \
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE_K)]                          \
                                   [[intel::grf_size(256)]] {                                      \
            lightning_indexer_kernel_vec<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, k_type>( \
                q_d, k_d, w_d, m_d, dst_d,                                                         \
                n_stream, n_batch, n_kv,                                                            \
                nb1, nb2, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3,        \
                get_pointer(smem_acc), item);                                                       \
        })

void ggml_sycl_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new_, w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int64_t n_embd   = q->ne[0];
    const int64_t n_head   = q->ne[1];
    const int64_t n_batch  = q->ne[2];
    const int64_t n_stream = q->ne[3];
    const int64_t n_kv     = k->ne[2];

    const float       * q_d   = (const float       *) q->data;
    const char        * k_d   = (const char        *) k->data;
    const float       * w_d   = (const float       *) w->data;
    const sycl::half  * m_d   = (const sycl::half  *) m->data;
    float             * dst_d = (      float       *) dst->data;

    // OPTIMIZED: K_VECS_PER_WARP=8 with large GRF (256 regs) to accommodate register pressure.
    // WARPS_PER_BLOCK=8 x WARP_SIZE=32 = 256 threads per block.
    // K_VECS_PER_BLOCK=64, so each block processes more KVs with fewer blocks.
    constexpr int K_VECS_PER_WARP  = 8;
    constexpr int WARPS_PER_BLOCK  = 8;
    constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

    constexpr int N_HEAD_INNER    = 64 / 4;
    constexpr int SMEM_Q_ELEMS    = N_HEAD_INNER * (128 / 4);
    constexpr int SMEM_FLOATS     = N_HEAD_INNER + SMEM_Q_ELEMS * 4 + K_VECS_PER_BLOCK;
    constexpr int SMEM_BYTES      = SMEM_FLOATS * (int)sizeof(float);

    sycl::range<3> block(1, WARPS_PER_BLOCK, WARP_SIZE_K);
    int64_t num_kv_blocks = (n_kv + K_VECS_PER_BLOCK - 1) / K_VECS_PER_BLOCK;
    sycl::range<3> grid(num_kv_blocks, n_batch, n_stream);

    const ggml_type k_type = k->type;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> smem_acc(sycl::range<1>(SMEM_FLOATS), cgh);

        if (n_embd == 128 && n_head == 64) {
            if (k_type == GGML_TYPE_F16) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 64, GGML_TYPE_F16);
            } else if (k_type == GGML_TYPE_Q4_0) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 64, GGML_TYPE_Q4_0);
            } else if (k_type == GGML_TYPE_Q4_1) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 64, GGML_TYPE_Q4_1);
            } else if (k_type == GGML_TYPE_Q5_0) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 64, GGML_TYPE_Q5_0);
            } else if (k_type == GGML_TYPE_Q5_1) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 64, GGML_TYPE_Q5_1);
            } else if (k_type == GGML_TYPE_Q8_0) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 64, GGML_TYPE_Q8_0);
            } else if (k_type == GGML_TYPE_BF16) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 64, GGML_TYPE_BF16);
            } else if (k_type == GGML_TYPE_F32) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 64, GGML_TYPE_F32);
            }
        } else if (n_embd == 128 && n_head == 32) {
            if (k_type == GGML_TYPE_F16) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 32, GGML_TYPE_F16);
            } else if (k_type == GGML_TYPE_Q4_0) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 32, GGML_TYPE_Q4_0);
            } else if (k_type == GGML_TYPE_Q4_1) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 32, GGML_TYPE_Q4_1);
            } else if (k_type == GGML_TYPE_Q5_0) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 32, GGML_TYPE_Q5_0);
            } else if (k_type == GGML_TYPE_Q5_1) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 32, GGML_TYPE_Q5_1);
            } else if (k_type == GGML_TYPE_Q8_0) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 32, GGML_TYPE_Q8_0);
            } else if (k_type == GGML_TYPE_BF16) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 32, GGML_TYPE_BF16);
            } else if (k_type == GGML_TYPE_F32) {
                LIGHTNING_INDEXER_VEC_DISPATCH(128, 32, GGML_TYPE_F32);
            }
        }
    });
}

bool ggml_sycl_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];

    if (!q || q->ne[0] != 128) return false;
    if (q->ne[1] != 32 && q->ne[1] != 64) return false;

    if (!k) return false;

    // IQ4_NL not supported (different dequantize path)
    if (k->type == GGML_TYPE_IQ4_NL) return false;

    return true;
}