#include "lightning_indexer.hpp"
#include "common.hpp"
#include "dequantize.hpp"

constexpr int WARP_SIZE_K = 32;

// Dequantize 4 values from K at given element offset i0 into float4.
// Matches CUDA get_dequantize_V<type, float, 4>() calling convention.
template <ggml_type TYPE_K>
static inline void dequantize_k_vec_4(const void * k_base, sycl::float4 * dst, int i0) {
    if constexpr (TYPE_K == GGML_TYPE_Q4_0) {
        constexpr int QK = QK4_0;
        const block_q4_0 * x = (const block_q4_0 *) k_base;

        const int ib  = i0 / QK;
        const int iqs = i0 % (QK / 2);

        const int vui0 = x[ib].qs[iqs + 0];
        const int vui1 = x[ib].qs[iqs + 1];

        const float d = x[ib].d;
        dst->x() = d * ((float)((int8_t)((vui0 & 0xF)      - 8)));
        dst->y() = d * ((float)((int8_t)((vui0 >> 4)        - 8)));
        dst->z() = d * ((float)((int8_t)((vui1 & 0xF)      - 8)));
        dst->w() = d * ((float)((int8_t)((vui1 >> 4)        - 8)));
    } else if constexpr (TYPE_K == GGML_TYPE_F16) {
        const sycl::half * src = (const sycl::half *) k_base + i0;
        dst->x() = (float)src[0];
        dst->y() = (float)src[1];
        dst->z() = (float)src[2];
        dst->w() = (float)src[3];
    } else {
        dst->x() = 0; dst->y() = 0; dst->z() = 0; dst->w() = 0;
    }
}

// Vector-based lightning indexer kernel.
// Each warp processes K_VECS_PER_WARP K-vectors. All warps in a block cooperate via shared memory.
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int N_EMBD, int N_HEAD>
static void lightning_indexer_kernel_vec_f16(
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

    const char  * q_base = (const char  *)                 Q + i_batch * nbq2 + i_stream * nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch * nbw1 + i_stream * nbw3);

    // Shared memory layout:
    //   w_shared[N_HEAD_INNER] at offset 0
    //   q_shared_f[N_HEAD_INNER * (N_EMBD/4)] at offset N_HEAD_INNER
    //   dst_shared[K_VECS_PER_BLOCK] at offset N_HEAD_INNER + N_HEAD_INNER * (N_EMBD/4)
    float       * w_shared    = local_smem;
    sycl::float4 * q_shared_f = (sycl::float4 *)(local_smem + N_HEAD_INNER);
    float       * dst_shared  = local_smem + N_HEAD_INNER + N_Q * 4;

    // Phase 1: load K vectors (F16) into registers
    sycl::float4 k_reg_f[K_VECS_PER_WARP];
#pragma unroll
    for (int k = 0; k < K_VECS_PER_WARP; ++k) {
        int i_kv = start_kv + k;
        if (i_kv < n_kv) {
            const void * k_base = (const void *) ((const char *) K + i_kv * nbk2 + i_stream * nbk3);
            dequantize_k_vec_4<GGML_TYPE_F16>(k_base, &k_reg_f[k], i_lane * 4);
        } else {
            k_reg_f[k] = sycl::float4{0, 0, 0, 0};
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // Phase 2: load weights and Q into shared memory
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

        // Phase 3: compute scores
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

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum<WARP_SIZE_K>(qk[k], item);
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    // Phase 4: store per-warp results to shared memory
    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    // Phase 5: write from shared memory to VRAM in coalesced manner
    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const sycl::half * m_base = (const sycl::half *) ((const char *) M + i_batch * nbm1 + (i_stream % nem3) * nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch * nb1 + i_stream * nb3);
            dst_base[i_kv] = dst_shared[tid] + (float)m_base[i_kv];
        }
    }
}

// Q4_0 specialization — differs only in K-loading phase
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int N_EMBD, int N_HEAD>
static void lightning_indexer_kernel_vec_q4_0(
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

    const char  * q_base = (const char  *)                 Q + i_batch * nbq2 + i_stream * nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch * nbw1 + i_stream * nbw3);

    float       * w_shared    = local_smem;
    sycl::float4 * q_shared_f = (sycl::float4 *)(local_smem + N_HEAD_INNER);
    float       * dst_shared  = local_smem + N_HEAD_INNER + N_Q * 4;

    // Phase 1: dequantize Q4_0 K vectors
    sycl::float4 k_reg_f[K_VECS_PER_WARP];
#pragma unroll
    for (int k = 0; k < K_VECS_PER_WARP; ++k) {
        int i_kv = start_kv + k;
        if (i_kv < n_kv) {
            const void * k_base = (const void *) ((const char *) K + i_kv * nbk2 + i_stream * nbk3);
            dequantize_k_vec_4<GGML_TYPE_Q4_0>(k_base, &k_reg_f[k], i_lane * 4);
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

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum<WARP_SIZE_K>(qk[k], item);
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

    constexpr int K_VECS_PER_WARP  = 8;
    constexpr int WARPS_PER_BLOCK  = 8;
    constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

    // Shared memory size: N_HEAD_INNER floats for weights + N_HEAD_INNER*(N_EMBD/4) float4s for Q + K_VECS_PER_BLOCK floats for dst
    constexpr int N_HEAD_INNER    = 64 / 4; // max N_HEAD=64 -> N_HEAD_INNER=16
    constexpr int SMEM_Q_ELEMS    = N_HEAD_INNER * (128 / 4); // 16 * 32 = 512 float4 elements
    constexpr int SMEM_FLOATS     = N_HEAD_INNER + SMEM_Q_ELEMS * 4 + K_VECS_PER_BLOCK; // 16 + 2048 + 64 = 2128
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
                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE_K)]] {
                        lightning_indexer_kernel_vec_f16<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 64>(
                            q_d, k_d, w_d, m_d, dst_d,
                            n_stream, n_batch, n_kv,
                            nb1, nb2, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3,
                            get_pointer(smem_acc), item);
                    });
            } else if (k_type == GGML_TYPE_Q4_0) {
                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE_K)]] {
                        lightning_indexer_kernel_vec_q4_0<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 64>(
                            q_d, k_d, w_d, m_d, dst_d,
                            n_stream, n_batch, n_kv,
                            nb1, nb2, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3,
                            get_pointer(smem_acc), item);
                    });
            }
        } else if (n_embd == 128 && n_head == 32) {
            if (k_type == GGML_TYPE_F16) {
                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE_K)]] {
                        lightning_indexer_kernel_vec_f16<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 32>(
                            q_d, k_d, w_d, m_d, dst_d,
                            n_stream, n_batch, n_kv,
                            nb1, nb2, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3,
                            get_pointer(smem_acc), item);
                    });
            } else if (k_type == GGML_TYPE_Q4_0) {
                cgh.parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE_K)]] {
                        lightning_indexer_kernel_vec_q4_0<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 32>(
                            q_d, k_d, w_d, m_d, dst_d,
                            n_stream, n_batch, n_kv,
                            nb1, nb2, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3,
                            get_pointer(smem_acc), item);
                    });
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
    const ggml_type k_type = k->type;
    if (k_type != GGML_TYPE_F16 && k_type != GGML_TYPE_Q4_0) {
        return false;
    }

    return true;
}